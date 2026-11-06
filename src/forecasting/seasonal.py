"""Seasonal (1-6 month lead) statistical forecasting with tercile probabilities.

Method (the "multiple linear regression baseline" of the specification)
  * Predictors: window means of ocean indices (R34, WIOD, EIOD, DMI, ...) over the ``predictor_months`` months
    ending at the initialization month.  Nothing after the initialization date is ever read.
  * Model: ridge-regularised multiple linear regression on standardised predictors.  For every target (station or
    grid cell) the predictor subset (0..``max_predictors`` predictors, 0 = climatology) is chosen by exact
    leave-one-out error computed on the TRAINING years only; a subset must beat climatology to be kept.
  * Probabilities: Gaussian predictive distribution (mean = regression forecast, sd = leave-one-out residual sd) turned
    into below / near / above-normal probabilities using terciles of the training years.  Precipitation is modelled on
    a square-root scale; terciles are unchanged by the monotone transform.
  * Hindcasts: rolling-origin (expanding window, only earlier years) or blocked (years within ``block_buffer`` of the
    test year removed) -- never random splits.  Terciles, climatology and selection all use training years only.

Terminology
  initialization date : last day of the initialization month (``init_month``); the forecast uses data up to here
  lead_months         : calendar months from the initialization month to the first month of the target season
                        (init May, JJAS -> 1; init Nov, FMAM -> 3)
  target period / valid time : first and last day of the target season
"""
from __future__ import annotations

import calendar
import itertools
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.preprocessing.station_processing import season_months

log = logging.getLogger("s2s.seasonal")

CATEGORIES = ["below", "normal", "above"]
TRANSFORMS = {
    "none": (lambda x: x, lambda x: x),
    "sqrt": (lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: np.clip(x, 0, None) ** 2),
    "log1p": (lambda x: np.log1p(np.clip(x, 0, None)), lambda x: np.expm1(x)),
}


def season_name(season) -> str:
    return season if isinstance(season, str) else "".join(calendar.month_abbr[m][0] for m in season_months(season))


# ---------------------------------------------------------------- forecast specification
@dataclass(frozen=True)
class ForecastSpec:
    season: object                 # name from settings.STATION_SEASONS or a list of month numbers
    init_month: int                # 1..12
    predictor_months: int = 3
    year_label: str = "start"      # how cross-year seasons are labelled: 'start' or 'end' year

    @property
    def months(self) -> list[int]:
        return season_months(self.season)

    @property
    def name(self) -> str:
        return season_name(self.season)

    @property
    def cross_year(self) -> bool:
        m = self.months
        return any(m[i] < m[i - 1] for i in range(1, len(m)))

    @property
    def lead_months(self) -> int:
        s = self.months[0]
        return ((s - self.init_month) % 12) or 12

    def start_year(self, season_year: int) -> int:
        return season_year - 1 if (self.cross_year and self.year_label == "end") else season_year

    def init_year(self, season_year: int) -> int:
        return self.start_year(season_year) if self.init_month < self.months[0] else self.start_year(season_year) - 1

    def init_date(self, season_year: int) -> pd.Timestamp:
        return pd.Timestamp(self.init_year(season_year), self.init_month, 1) + pd.offsets.MonthEnd(0)

    def window(self, season_year: int) -> list[pd.Timestamp]:
        end = pd.Timestamp(self.init_year(season_year), self.init_month, 1)
        return [end - pd.DateOffset(months=k) for k in range(self.predictor_months - 1, -1, -1)]

    def target_start(self, season_year: int) -> pd.Timestamp:
        return pd.Timestamp(self.start_year(season_year), self.months[0], 1)

    def target_end(self, season_year: int) -> pd.Timestamp:
        last = self.months[-1]
        y = self.start_year(season_year) + (1 if self.cross_year else 0)
        return pd.Timestamp(y, last, 1) + pd.offsets.MonthEnd(0)

    def label(self, season_year: int) -> str:
        return f"{self.name}_{season_year}"

    def describe(self, season_year: int) -> dict:
        return {"init_date": self.init_date(season_year).date().isoformat(), "lead_months": self.lead_months,
                "target_start": self.target_start(season_year).date().isoformat(),
                "target_end": self.target_end(season_year).date().isoformat(),
                "forecast_period": self.label(season_year)}


# ---------------------------------------------------------------- predictors (leak-free)
def window_means(monthly: pd.DataFrame, spec: ForecastSpec, years) -> pd.DataFrame:
    """Mean over the spec's predictor window for each season_year; NaN unless every month of the window exists.

    ``monthly`` is indexed by month start.  Only months inside [window start, initialization month] are touched.
    """
    monthly = monthly.copy(); monthly.index = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp()
    rows = {}
    for y in years:
        w = spec.window(int(y))
        part = monthly.reindex(w)
        rows[int(y)] = part.mean() if not part.isna().any().any() else part.mean().where(~part.isna().any())
    return pd.DataFrame.from_dict(rows, orient="index", columns=monthly.columns)


def build_predictors(indices: pd.DataFrame, spec: ForecastSpec, years, tendency: bool = False) -> pd.DataFrame:
    """Predictor table (rows = season_year, columns = indices) from a monthly index table."""
    X = window_means(indices, spec, years)
    if tendency and spec.predictor_months > 1:
        first = window_means(indices, ForecastSpec(spec.season, spec.init_month, 1, spec.year_label), years)
        # tendency = last month minus the window mean (shows whether the anomaly is growing)
        X = X.join((first - X).add_suffix("_tend"))
    return X


def recent_anomaly(monthly_target: pd.DataFrame, spec: ForecastSpec, years, min_ref_years: int = 10) -> pd.DataFrame:
    """Persistence baseline input: anomaly of the target in the predictor window, relative to EARLIER years only."""
    W = window_means(monthly_target, spec, years).sort_index()
    out = W.copy() * np.nan
    for i, y in enumerate(W.index):
        past = W.iloc[:i]
        ok = past.notna().sum() >= min_ref_years
        out.loc[y] = (W.loc[y] - past.mean()).where(ok)
    return out


def predictor_diagnostics(X: pd.DataFrame) -> dict:
    """Correlation matrix and variance-inflation factors (VIF > 5-10 signals multicollinearity)."""
    C = X.dropna().corr()
    inv = np.linalg.pinv(C.values + 1e-9 * np.eye(len(C)))
    vif = pd.Series(np.diag(inv), index=C.columns)
    return {"correlation": C, "vif": vif}


# ---------------------------------------------------------------- regression engine
def _loo(Xs: np.ndarray, Y: np.ndarray, alpha: float):
    """Exact leave-one-out RMSE (per column of Y) of ridge regression on standardised Xs (n x k, k may be 0)."""
    n = len(Y); Yc = Y - Y.mean(0)
    if Xs.shape[1] == 0:
        h = np.full(n, 1.0 / n); fit = np.zeros_like(Y)
    else:
        sd = Xs.std(0, ddof=1); sd[sd == 0] = 1.0
        Z = (Xs - Xs.mean(0)) / sd
        Ainv = np.linalg.inv(Z.T @ Z + alpha * np.eye(Z.shape[1]))
        fit = Z @ (Ainv @ Z.T @ Yc)
        h = 1.0 / n + np.einsum("ij,jk,ik->i", Z, Ainv, Z)
    loo = (Yc - fit) / (1.0 - h)[:, None]
    return np.sqrt((loo ** 2).mean(0))


def _predict(Xs: np.ndarray, Y: np.ndarray, xt: np.ndarray, alpha: float) -> np.ndarray:
    ym = Y.mean(0)
    if Xs.shape[1] == 0:
        return np.tile(ym, (len(xt), 1))
    sd = Xs.std(0, ddof=1); sd[sd == 0] = 1.0; mu = Xs.mean(0)
    Z = (Xs - mu) / sd
    B = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return ym + ((xt - mu) / sd) @ B


def _subsets(p: int, max_k: int):
    out = [()]
    for k in range(1, min(max_k, p) + 1):
        out += list(itertools.combinations(range(p), k))
    return out


def _select_and_predict(F, ft, Y, subsets, labels, alpha, tol):
    """Per target, pick the simplest feature subset whose exact LOO error is within ``tol`` of the best and that beats
    climatology (subsets[0] must be the empty set); predict at ``ft``.  Returns mean, sigma (LOO rmse), label."""
    R = np.vstack([_loo(F[:, list(sub)], Y, alpha) for sub in subsets])          # subsets x targets
    base = R[0]; m = Y.shape[1]
    chosen = np.zeros(m, dtype=int)
    for j in range(m):
        best = R[:, j].min()
        ok = np.where(R[:, j] <= best * (1 + tol))[0]
        k = min(ok, key=lambda q: (len(subsets[q]), R[q, j]))
        chosen[j] = k if R[k, j] < base[j] else 0
    mean = np.empty(m); sigma = np.empty(m); label = np.empty(m, dtype=object)
    for q in np.unique(chosen):
        jj = np.where(chosen == q)[0]; idx = list(subsets[q])
        mean[jj] = _predict(F[:, idx], Y[:, jj], ft[:, idx], alpha)[0]
        sigma[jj] = R[q, jj]; label[jj] = labels[q]
    return mean, sigma, label


# ---------------------------------------------------------------- forecasting methods
# Every method has the signature  f(X, Y, xt, names, o) -> (mean[m], sigma[m], label[m])  with complete X (n x p),
# complete Y (n x m, already transformed), the predictors xt (1 x p) of the forecast year and the options dict ``o``.
# sigma is an out-of-sample error scale (leave-one-out or out-of-bag).  Every method falls back to climatology for a
# target where it does not beat climatology out of sample.
def method_climatology(X, Y, xt, names, o):
    n, m = Y.shape
    return Y.mean(0), _loo(np.empty((n, 0)), Y, o["alpha"]), np.full(m, "climatology", dtype=object)


def method_ridge(X, Y, xt, names, o):
    cand = list(range(X.shape[1])) if o.get("candidates") is None else [names.index(c) for c in o["candidates"]]
    subsets = [tuple(cand[i] for i in sub) for sub in _subsets(len(cand), o["max_k"])]
    labels = ["climatology" if not sub else "+".join(names[i] for i in sub) for sub in subsets]
    return _select_and_predict(X, xt, Y, subsets, labels, o["alpha"], o["tol"])


def method_pcr(X, Y, xt, names, o):
    """EOF/PCA of the predictors followed by regression on the leading components (number chosen by LOO)."""
    mu = X.mean(0); sd = X.std(0, ddof=1); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    keep = int((S > 1e-8 * S[0]).sum()) if S[0] > 0 else 0
    kmax = min(keep, o.get("max_pc", 3))
    subsets = [tuple(range(k)) for k in range(kmax + 1)]
    labels = ["climatology" if k == 0 else f"PC1-{k}" if k > 1 else "PC1" for k in range(kmax + 1)]
    return _select_and_predict(Z @ Vt.T, ((xt - mu) / sd) @ Vt.T, Y, subsets, labels, o["alpha"], o["tol"])


def _cca_pred(Xtr, Ytr, Xte, kx, ky, kc):
    """Reduced-rank (CCA) prediction of Y from X via PCA truncation of both fields; returns len(Xte) x m."""
    mx = Xtr.mean(0); sx = Xtr.std(0, ddof=1); sx[sx == 0] = 1.0
    my = Ytr.mean(0); sy = Ytr.std(0, ddof=1); sy[sy == 0] = 1.0
    Zx = (Xtr - mx) / sx; Zy = (Ytr - my) / sy; n = len(Zx)
    _, Sx, Vx = np.linalg.svd(Zx, full_matrices=False); _, Sy, Vy = np.linalg.svd(Zy, full_matrices=False)
    if Sx[0] < 1e-12 or Sy[0] < 1e-12:
        return np.tile(my, (len(Xte), 1))
    kx = min(kx, int((Sx > 1e-8 * Sx[0]).sum())); ky = min(ky, int((Sy > 1e-8 * Sy[0]).sum()))
    Px, Py = Zx @ Vx[:kx].T, Zy @ Vy[:ky].T
    Cxx, Cyy, Cxy = Px.T @ Px / (n - 1), Py.T @ Py / (n - 1), Px.T @ Py / (n - 1)
    def powm(C, e):
        w, U = np.linalg.eigh(C); return (U * w ** e) @ U.T
    Cxx_h, Cyy_h, Cyy_s = powm(Cxx, -0.5), powm(Cyy, -0.5), powm(Cyy, 0.5)
    A, s, Bt = np.linalg.svd(Cxx_h @ Cxy @ Cyy_h)
    kc = min(kc, len(s))
    Brr = Cxx_h @ ((A[:, :kc] * s[:kc]) @ Bt[:kc]) @ Cyy_s
    return (((Xte - mx) / sx) @ Vx[:kx].T @ Brr) @ Vy[:ky] * sy + my


def method_cca(X, Y, xt, names, o):
    """Canonical correlation analysis between the predictors and all targets of the group (multi-target by nature)."""
    n, p = X.shape; m = Y.shape[1]
    combos = [(kx, ky, kc) for kx in range(1, min(3, p) + 1) for ky in range(1, min(3, m) + 1)
              for kc in range(1, min(kx, ky) + 1)]
    loo = np.empty((len(combos), n, m)); clim = np.empty((n, m))
    for i in range(n):
        tr = np.arange(n) != i
        clim[i] = Y[tr].mean(0)
        for c, (kx, ky, kc) in enumerate(combos):
            loo[c, i] = _cca_pred(X[tr], Y[tr], X[i:i + 1], kx, ky, kc)[0]
    sd2 = Y.var(0, ddof=1); sd2[sd2 == 0] = 1.0
    mse = ((loo - Y) ** 2).mean(1)                                   # combos x targets
    best = int(np.argmin((mse / sd2).sum(1)))
    rm_c = np.sqrt(mse[best]); rm_0 = np.sqrt(((clim - Y) ** 2).mean(0))
    use = rm_c < rm_0
    kx, ky, kc = combos[best]
    mean = np.where(use, _cca_pred(X, Y, xt, kx, ky, kc)[0], Y.mean(0))
    sigma = np.where(use, rm_c, rm_0)
    label = np.where(use, f"CCA({kx},{ky},{kc})", "climatology").astype(object)
    return mean, sigma, label


def method_rf(X, Y, xt, names, o):
    """Random forest (multi-output on standardised targets); error scale from out-of-bag residuals."""
    from sklearn.ensemble import RandomForestRegressor
    n, m = Y.shape
    my = Y.mean(0); sy = Y.std(0, ddof=1); sy[sy == 0] = 1.0
    Ys = (Y - my) / sy
    rf = RandomForestRegressor(n_estimators=o.get("rf_trees", 200), min_samples_leaf=o.get("rf_leaf", 3), bootstrap=True,
                               oob_score=True, random_state=o.get("seed", 42), n_jobs=1).fit(X, Ys)
    oob = np.asarray(rf.oob_prediction_).reshape(n, -1)
    err = np.sqrt(np.nanmean((Ys - oob) ** 2, axis=0)) * sy
    base = _loo(np.empty((n, 0)), Y, o["alpha"])
    use = err < base
    pred = rf.predict(xt).reshape(-1) * sy + my
    mean = np.where(use, pred, my)
    return mean, np.where(use, err, base), np.where(use, "RF", "climatology").astype(object)


METHODS = {"climatology": method_climatology, "ridge": method_ridge, "pcr": method_pcr, "cca": method_cca,
           "rf": method_rf}


def _fit_predict(method, Xtr, Ytr, xt, names, o, min_train):
    """Fit ``method`` for all targets (columns of Ytr; NaN gaps allowed) and predict at xt (1 x p)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHODS)}")
    m = Ytr.shape[1]
    mean = np.full(m, np.nan); sigma = np.full(m, np.nan); label = np.full(m, "", dtype=object)
    pats, inv = np.unique(np.isnan(Ytr), axis=1, return_inverse=True)       # targets grouped by missing-year pattern
    inv = np.asarray(inv).ravel()
    for g in range(pats.shape[1]):
        rows = ~pats[:, g]; cols = np.where(inv == g)[0]
        if rows.sum() < min_train:
            continue
        mean[cols], sigma[cols], label[cols] = METHODS[method](Xtr[rows], Ytr[np.ix_(rows, cols)], xt, names, o)
    return mean, sigma, label


def _probabilities(mean_t, sigma_t, t1_t, t2_t):
    with np.errstate(invalid="ignore", divide="ignore"):
        pb = ndtr((t1_t - mean_t) / sigma_t)
        pa = 1.0 - ndtr((t2_t - mean_t) / sigma_t)
    pb = np.where(sigma_t > 0, pb, np.nan); pa = np.where(sigma_t > 0, pa, np.nan)
    return pb, 1.0 - pb - pa, pa


def _one_step(Ytr_df, Xtr_df, xt, transform, method, o, min_train):
    fwd, inv = TRANSFORMS[transform]
    Ytr = Ytr_df.values.astype(float)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN targets (e.g. ocean cells) stay NaN
        t_orig = np.nanpercentile(Ytr, [100 / 3, 200 / 3], axis=0)
        clim = np.nanmean(Ytr, axis=0)
    mean_t, sigma_t, label = _fit_predict(method, Xtr_df.values, fwd(Ytr), xt, list(Xtr_df.columns), o, min_train)
    pb, pn, pa = _probabilities(mean_t, sigma_t, fwd(t_orig[0]), fwd(t_orig[1]))
    if method == "climatology":                                   # exactly 1/3 each by construction
        ok = np.isfinite(mean_t)
        pb, pn, pa = (np.where(ok, 1 / 3, np.nan) for _ in range(3))
    return {"forecast": inv(mean_t), "clim": clim, "sigma_t": sigma_t, "t1": t_orig[0], "t2": t_orig[1],
            "p_below": pb, "p_normal": pn, "p_above": pa, "selected": label}


def _opts(alpha, max_predictors, tol, candidates, seed=42, **extra):
    return {"alpha": alpha, "max_k": max_predictors, "tol": tol, "candidates": candidates, "seed": seed, **extra}


# ---------------------------------------------------------------- hindcast
@dataclass
class Hindcast:
    spec: ForecastSpec
    settings: dict
    obs: pd.DataFrame
    forecast: pd.DataFrame
    clim: pd.DataFrame
    sigma_t: pd.DataFrame
    t1: pd.DataFrame
    t2: pd.DataFrame
    p_below: pd.DataFrame
    p_normal: pd.DataFrame
    p_above: pd.DataFrame
    selected: pd.DataFrame
    persistence: pd.DataFrame | None = None
    transform: str = "none"

    @property
    def probs(self) -> np.ndarray:
        """(years, targets, 3) array: below, normal, above."""
        return np.stack([self.p_below.values, self.p_normal.values, self.p_above.values], axis=-1)

    @property
    def obs_category(self) -> np.ndarray:
        """Observed tercile category (0 below, 1 normal, 2 above) using TRAINING-year terciles; -1 if missing."""
        o = self.obs.values
        cat = np.where(o < self.t1.values, 0, np.where(o > self.t2.values, 2, 1))
        return np.where(np.isnan(o) | np.isnan(self.t1.values), -1, cat)


def _train_years(y, years, scheme, block_buffer):
    if scheme == "rolling":
        return [t for t in years if t < y]
    if scheme == "blocked":
        return [t for t in years if abs(t - y) > block_buffer]
    raise ValueError("scheme must be 'rolling' or 'blocked'")


def hindcast(Y: pd.DataFrame, X: pd.DataFrame, spec: ForecastSpec, *, scheme: str = "rolling", min_train: int = 15,
             block_buffer: int = 1, alpha: float = 1.0, max_predictors: int = 2, tol: float = 0.02,
             transform: str = "none", persist: pd.DataFrame | None = None, candidates=None,
             method: str = "ridge", seed: int = 42, **method_opts) -> Hindcast:
    """Out-of-sample hindcasts for every year with enough training data.

    Y: seasonal target values (index season_year, columns stations/cells).  X: predictors (same index).
    rolling: train on earlier years only (expanding window).  blocked: train on all years farther than
    ``block_buffer`` from the test year (skill estimate only; uses future years as training data).
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {list(TRANSFORMS)}")
    years = sorted(set(Y.index) & set(X.index)); Y = Y.loc[years]; X = X.loc[years]
    valid_x = X.notna().all(axis=1)
    out = {k: pd.DataFrame(np.nan, index=years, columns=Y.columns) for k in
           ["forecast", "clim", "sigma_t", "t1", "t2", "p_below", "p_normal", "p_above"]}
    out["selected"] = pd.DataFrame("", index=years, columns=Y.columns, dtype=object)
    for t in years:
        if not valid_x[t]:
            continue
        tr = [y for y in _train_years(t, years, scheme, block_buffer) if valid_x[y]]
        if len(tr) < min_train:
            continue
        r = _one_step(Y.loc[tr], X.loc[tr], X.loc[[t]].values, transform, method,
                      _opts(alpha, max_predictors, tol, candidates, seed, **method_opts), min_train)
        for k in ["forecast", "clim", "sigma_t", "t1", "t2", "p_below", "p_normal", "p_above"]:
            out[k].loc[t] = r[k]
        out["selected"].loc[t] = r["selected"]
    pers = None
    if persist is not None:
        P = persist.reindex(index=years, columns=Y.columns)
        pers = out["clim"] + P
    done = out["forecast"].notna().any(axis=1)
    sl = lambda d: d.loc[done]
    return Hindcast(spec, {"scheme": scheme, "min_train": min_train, "block_buffer": block_buffer, "alpha": alpha,
                           "max_predictors": max_predictors, "tol": tol, "transform": transform, "method": method},
                    sl(Y), sl(out["forecast"]), sl(out["clim"]), sl(out["sigma_t"]), sl(out["t1"]), sl(out["t2"]),
                    sl(out["p_below"]), sl(out["p_normal"]), sl(out["p_above"]), sl(out["selected"]),
                    None if pers is None else sl(pers), transform)


# ---------------------------------------------------------------- operational forecast
def forecast(Y: pd.DataFrame, X: pd.DataFrame, spec: ForecastSpec, target_year: int, *, min_train: int = 15,
             alpha: float = 1.0, max_predictors: int = 2, tol: float = 0.02, transform: str = "none",
             candidates=None, meta: pd.DataFrame | None = None, variable: str = "", method: str = "ridge",
             seed: int = 42, **method_opts) -> pd.DataFrame:
    """Forecast ``target_year`` from predictors available at the initialization date.

    Trains on every year of Y strictly before ``target_year`` (the target year's own observation is never used).
    ``meta`` (columns station, lat, lon) is joined to give the specification's station output format.
    """
    if target_year not in X.index or X.loc[target_year].isna().any():
        raise ValueError(f"Predictors for {spec.label(target_year)} are incomplete: need monthly indices through "
                         f"{spec.init_date(target_year).date()} (window of {spec.predictor_months} months).")
    train = [y for y in Y.index if y < target_year and y in X.index and X.loc[y].notna().all()]
    if len(train) < min_train:
        raise ValueError(f"Only {len(train)} training years before {target_year}; need >= {min_train}.")
    r = _one_step(Y.loc[train], X.loc[train], X.loc[[target_year]].values, transform, method,
                  _opts(alpha, max_predictors, tol, candidates, seed, **method_opts), min_train)
    df = pd.DataFrame({"Station": list(Y.columns), "Variable": variable,
                       "Forecast": r["forecast"], "Anomaly": r["forecast"] - r["clim"],
                       "Climatology": r["clim"], "P_below": r["p_below"], "P_normal": r["p_normal"],
                       "P_above": r["p_above"], "Tercile_lower": r["t1"], "Tercile_upper": r["t2"],
                       "Predictors": r["selected"], "N_train_years": len(train), "Method": method})
    d = spec.describe(target_year)
    df["Forecast_Period"] = d["forecast_period"]
    for k in ("init_date", "lead_months", "target_start", "target_end"):
        df[k] = d[k]
    if meta is not None:
        df = df.merge(meta[["station", "lat", "lon"]].rename(columns={"station": "Station"}), on="Station", how="left")
    first = ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"]
    first = [c for c in first if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


# ---------------------------------------------------------------- station tables
def station_matrix(seasonal: pd.DataFrame, variable: str, season_name_: str | None = None) -> pd.DataFrame:
    """Seasonal station table (from station_processing.seasonal_aggregate) -> years x stations matrix."""
    d = seasonal[seasonal["variable"] == variable]
    if season_name_ is not None:
        d = d[d["season"] == season_name_]
    return d.pivot(index="season_year", columns="station", values="value").sort_index()


def run_station_forecast(seasonal: pd.DataFrame, stations: pd.DataFrame, indices: pd.DataFrame, spec: ForecastSpec,
                         target_year: int, variable: str, *, transform: str = "none", scheme: str = "rolling",
                         monthly: pd.DataFrame | None = None, tendency: bool = False, methods=("ridge",),
                         ensemble_mode: str = "skill", ensemble_pool: str = "pooled", **kw):
    """Hindcast + forecast for every station of one variable.

    One method -> that method.  Several -> every method, climatology, and their combination (forecasting.ensemble).
    Returns (forecast_df, verification Hindcast, {name: Hindcast}).
    """
    Y = station_matrix(seasonal, variable, spec.name)
    years = sorted(set(Y.index) | {target_year})
    X = build_predictors(indices, spec, years, tendency)
    persist = None
    if monthly is not None:
        mv = monthly[monthly["variable"] == variable].pivot(index="month", columns="station", values="value")
        persist = recent_anomaly(mv, spec, Y.index)
    methods = list(methods)
    if len(methods) == 1:
        hc = hindcast(Y, X, spec, scheme=scheme, transform=transform, persist=persist, method=methods[0], **kw)
        fc = forecast(Y, X, spec, target_year, transform=transform, meta=stations, variable=variable, method=methods[0],
                      **{k: v for k, v in kw.items() if k != "block_buffer"})
        return fc, hc, {methods[0]: hc}
    from src.forecasting import ensemble
    return ensemble.run_multimodel(Y, X, spec, target_year, methods, transform=transform, scheme=scheme,
                                   mode=ensemble_mode, pool=ensemble_pool, persist=persist, meta=stations,
                                   variable=variable, **kw)


# ---------------------------------------------------------------- gridded fields
def forecast_gridded(field: np.ndarray, years, lat, lon, X: pd.DataFrame, spec: ForecastSpec, target_year: int,
                     **kw) -> dict:
    """Forecast a (year, lat, lon) seasonal field; returns arrays shaped (lat, lon) / (3, lat, lon)."""
    field = np.asarray(field, dtype=float)
    ny, nl, no = field.shape
    Y = pd.DataFrame(field.reshape(ny, nl * no), index=list(years))
    df = forecast(Y, X, spec, target_year, **kw)
    sh = (nl, no)
    prob = np.stack([df["P_below"].values, df["P_normal"].values, df["P_above"].values]).reshape(3, *sh)
    return {"forecast": df["Forecast"].values.reshape(sh), "anomaly": df["Anomaly"].values.reshape(sh),
            "probabilities": prob, "lat": np.asarray(lat), "lon": np.asarray(lon),
            "predictors": df["Predictors"].values.reshape(sh), "meta": spec.describe(target_year)}
