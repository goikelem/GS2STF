"""Subseasonal (weekly, 1-8 weeks ahead) forecasting for stations, with tercile probabilities.

Set-up
  A forecast is issued on an *initialization date* d using data up to the day BEFORE d.  The target is a window of
  whole weeks after d: ``lead_weeks=(2, 4)`` means days d+7 ... d+27 (weeks 2-4, one aggregated 3-week total / mean),
  ``(3, 6)`` weeks 3-6, ``(3, 3)`` the single week 3.  Run each lead separately for weekly resolution.
Samples and leak control
  For every earlier year the same window is built at the initialization date shifted by -k..+k weeks
  (``pool_weeks=k``), giving ~3k+... training samples instead of ~30 and a smoother tercile estimate.  A training sample
  is used only if its whole target window ended BEFORE the forecast's initialization date, and (hindcasts) only if
  it belongs to an earlier year: rolling-origin, never random splits.
Predictors (everything available at d)
  * ocean indices (R34, DMI, TNA, ...): mean of the last ``predictor_months`` COMPLETE months before d
  * persistence: the station's own anomaly over the last ``recent_weeks`` weeks before d
Targets are standardised per calendar shift (anomaly / sd of the training years at that shift; precipitation on a
square-root scale), so different times of the year are not mixed; terciles come from the pooled training anomalies.
Any method of forecasting.seasonal can be used (ridge, pcr, rf, climatology); every method falls back to climatology
when it does not beat it out of sample.  Precipitation is not forecast where more than ``max_zero_fraction`` of the
training windows are dry (terciles are not defined there).
Skill at weeks 2-6 is expected to be low: judge it by the hindcast RPSS and its bootstrap interval, not by the maps.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting import seasonal as S

log = logging.getLogger("s2s.subseasonal")
SUM_VARIABLES = {"precipitation"}
TRANSFORM = {"precipitation": "sqrt"}


@dataclass(frozen=True)
class SubseasonalSpec:
    init_month: int
    init_day: int
    lead_weeks: tuple = (2, 4)
    pool_weeks: int = 1
    recent_weeks: int = 2
    predictor_months: int = 3

    def __post_init__(self):
        lo, hi = self.lead_weeks
        if not (1 <= lo <= hi <= 12):
            raise ValueError("lead_weeks must satisfy 1 <= first <= last <= 12")
        if self.pool_weeks < 0 or self.recent_weeks < 1 or self.predictor_months < 1:
            raise ValueError("pool_weeks >= 0, recent_weeks >= 1, predictor_months >= 1 required")
        pd.Timestamp(2001, self.init_month, self.init_day)                    # validates the date

    def init_date(self, year: int, shift: int = 0) -> pd.Timestamp:
        d = self.init_day if not (self.init_month == 2 and self.init_day == 29) else 28
        return pd.Timestamp(year, self.init_month, d) + pd.Timedelta(days=7 * shift)

    def target_window(self, year: int, shift: int = 0):
        lo, hi = self.lead_weeks
        d = self.init_date(year, shift)
        return d + pd.Timedelta(days=7 * (lo - 1)), d + pd.Timedelta(days=7 * hi - 1)

    def recent_window(self, year: int, shift: int = 0):
        d = self.init_date(year, shift)
        return d - pd.Timedelta(days=7 * self.recent_weeks), d - pd.Timedelta(days=1)

    def sst_months(self, year: int, shift: int = 0):
        """Last ``predictor_months`` months that are complete before the initialization date."""
        last = self.init_date(year, shift).to_period("M").to_timestamp() - pd.DateOffset(months=1)
        return [last - pd.DateOffset(months=k) for k in range(self.predictor_months - 1, -1, -1)]

    @property
    def lead_label(self) -> str:
        lo, hi = self.lead_weeks
        return f"{lo}" if lo == hi else f"{lo}-{hi}"

    def describe(self, year: int) -> dict:
        s, e = self.target_window(year)
        lo, hi = self.lead_weeks
        return {"forecast_period": f"W{self.lead_label}_{self.init_date(year).date()}",
                "init_date": self.init_date(year).date().isoformat(), "lead_weeks": self.lead_label,
                "target_start": s.date().isoformat(), "target_end": e.date().isoformat(),
                "resolution": "weekly" if lo == hi else f"{hi - lo + 1}-week window"}


# ---------------------------------------------------------------- data preparation
def prepare_daily(obs: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Tidy station observations -> complete daily table (date x station); missing days stay NaN."""
    d = obs[obs["variable"] == variable]
    if d.empty:
        raise ValueError(f"No '{variable}' observations")
    w = d.pivot_table(index="date", columns="station", values="value", aggfunc="first").sort_index()
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))


def window_value(daily: pd.DataFrame, start, end, how: str, min_frac: float = 0.9) -> np.ndarray:
    """Window total (``sum``: mean x length, so up to 1-min_frac missing days are tolerated) or mean; NaN if too many
    days are missing or the window is outside the record."""
    m = daily.shape[1]
    if start < daily.index[0] or end > daily.index[-1]:
        return np.full(m, np.nan)
    w = daily.loc[start:end].values
    n = w.shape[0]; valid = np.isfinite(w).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(w, axis=0)
    val = mean * n if how == "sum" else mean
    return np.where(valid >= min_frac * n, val, np.nan)


@dataclass
class Samples:
    meta: pd.DataFrame       # year, shift, init, target_end
    Y: np.ndarray            # (n, stations) target window total / mean
    R: np.ndarray            # (n, stations) recent window total / mean (persistence predictor)
    X: pd.DataFrame          # (n, predictors) ocean predictors available before the initialization date
    stations: list
    variable: str


def build_samples(daily: pd.DataFrame, indices: pd.DataFrame, spec: SubseasonalSpec, years, variable: str,
                  min_frac: float = 0.9) -> Samples:
    how = "sum" if variable in SUM_VARIABLES else "mean"
    ind = indices.copy(); ind.index = pd.DatetimeIndex(ind.index).to_period("M").to_timestamp()
    meta, Y, R, X = [], [], [], []
    for y in years:
        for s in range(-spec.pool_weeks, spec.pool_weeks + 1):
            ts, te = spec.target_window(y, s); rs, re = spec.recent_window(y, s)
            meta.append({"year": int(y), "shift": s, "init": spec.init_date(y, s), "target_end": te})
            Y.append(window_value(daily, ts, te, how, min_frac)); R.append(window_value(daily, rs, re, how, min_frac))
            part = ind.reindex(spec.sst_months(y, s))
            X.append(part.mean().where(~part.isna().any()))
    return Samples(pd.DataFrame(meta), np.array(Y), np.array(R), pd.DataFrame(X).reset_index(drop=True),
                   list(daily.columns), variable)


# ---------------------------------------------------------------- one station, one test date
def _stats(T, shifts, rows):
    """Per-shift mean / sd of transformed values over training ``rows``."""
    out = {}
    for s in np.unique(shifts[rows]):
        v = T[rows & (shifts == s)]; v = v[np.isfinite(v)]
        if len(v) >= 5:
            out[s] = (v.mean(), v.std(ddof=1) if v.std(ddof=1) > 0 else 1.0)
    return out


def _fit_station(sm: Samples, j: int, tr: np.ndarray, te: int, method: str, o: dict, transform: str, min_train: int,
                 use_persistence: bool, max_zero_fraction: float):
    """tr: boolean mask of training samples, te: row of the forecast sample.  Returns a result dict or None."""
    fwd, inv = S.TRANSFORMS[transform]
    shifts = sm.meta["shift"].values
    y_raw = sm.Y[:, j]
    if sm.variable in SUM_VARIABLES:
        yv = y_raw[tr & np.isfinite(y_raw)]
        if len(yv) == 0 or np.mean(yv <= 0) > max_zero_fraction:
            return {"label": "dry-season (not forecastable)"}
    Yt, Rt = fwd(y_raw), fwd(sm.R[:, j])
    sy, sr = _stats(Yt, shifts, tr), _stats(Rt, shifts, tr)
    if 0 not in sy:
        return None
    m0, s0 = sy[0]
    Z = np.full(len(Yt), np.nan); P = np.full(len(Yt), np.nan)
    for s, (mu, sd) in sy.items():
        k = shifts == s; Z[k] = (Yt[k] - mu) / sd
    for s, (mu, sd) in sr.items():
        k = shifts == s; P[k] = (Rt[k] - mu) / sd
    feats = sm.X.copy(); names = list(feats.columns)
    persist_ok = use_persistence and 0 in sr and np.isfinite(P[te])
    if persist_ok:
        feats["persist"] = P; names.append("persist")
    F = feats.values
    rows = tr & np.isfinite(Z) & np.isfinite(F).all(axis=1)
    xt = F[te:te + 1]
    if rows.sum() < min_train or not np.isfinite(xt).all():
        return None
    og = {**o, "groups": sm.meta["year"].values[rows]}          # leave whole years out when estimating errors
    mz, sg, lab = S._fit_predict(method, F[rows], Z[rows][:, None], xt, names, og, min_train)
    if not np.isfinite(mz[0]):
        return None
    t1z, t2z = np.percentile(Z[rows], [100 / 3, 200 / 3])
    pb, pn, pa = S._probabilities(mz, sg, t1z, t2z)
    if method == "climatology":
        pb = pn = pa = np.array([1 / 3])
    clim = float(np.mean(y_raw[tr & (shifts == 0) & np.isfinite(y_raw)]))
    return {"forecast": float(S.retransform_mean(transform, m0 + s0 * mz[0], s0 * sg[0])), "clim": clim, "t1": float(inv(m0 + s0 * t1z)),
            "t2": float(inv(m0 + s0 * t2z)), "pb": float(pb[0]), "pn": float(pn[0]), "pa": float(pa[0]),
            "label": str(lab[0]), "n_years": int(sm.meta.loc[rows, "year"].nunique())}


def _train_mask(sm: Samples, year: int, init0: pd.Timestamp) -> np.ndarray:
    return ((sm.meta["year"].values < year) & (sm.meta["target_end"].values < np.datetime64(init0))).astype(bool)


def _options(alpha, max_predictors, tol, min_gain, seed, extra):
    return {"alpha": alpha, "max_k": max_predictors, "tol": tol, "candidates": None, "seed": seed, "min_gain": min_gain, **extra}


# ---------------------------------------------------------------- hindcast and forecast
def hindcast(sm: Samples, spec: SubseasonalSpec, method: str = "ridge", *, min_train_years: int = 10, alpha: float = 1.0,
             max_predictors: int = 2, tol: float = 0.02, min_gain: float = S.MIN_GAIN, seed: int = 42,
             use_persistence: bool = True, max_zero_fraction: float = 0.25, **method_opts) -> S.Hindcast:
    """Rolling-origin hindcast of the spec's initialization date: every test year is forecast from earlier years only."""
    transform = TRANSFORM.get(sm.variable, "none")
    o = _options(alpha, max_predictors, tol, min_gain, seed, method_opts)
    years = sorted(set(sm.meta.loc[sm.meta["shift"] == 0, "year"]))
    cols = sm.stations
    keys = ["forecast", "clim", "t1", "t2", "p_below", "p_normal", "p_above"]
    out = {k: pd.DataFrame(np.nan, index=years, columns=cols) for k in keys}
    sel = pd.DataFrame("", index=years, columns=cols, dtype=object)
    obs = pd.DataFrame(np.nan, index=years, columns=cols)
    for y in years:
        te = int(np.where((sm.meta["year"].values == y) & (sm.meta["shift"].values == 0))[0][0])
        tr = _train_mask(sm, y, spec.init_date(y))
        if sm.meta.loc[tr, "year"].nunique() < min_train_years:
            continue
        obs.loc[y] = sm.Y[te]
        for j, st in enumerate(cols):
            r = _fit_station(sm, j, tr, te, method, o, transform, min_train_years * (1 + 2 * spec.pool_weeks) // 2 or 5,
                             use_persistence, max_zero_fraction)
            if r is None or "forecast" not in r:
                continue
            for k, rk in zip(keys, ["forecast", "clim", "t1", "t2", "pb", "pn", "pa"]):
                out[k].loc[y, st] = r[rk]
            sel.loc[y, st] = r["label"]
    done = out["forecast"].notna().any(axis=1)
    sl = lambda d: d.loc[done]
    nan = sl(out["forecast"]) * np.nan
    return S.Hindcast(spec, {"scheme": "rolling", "method": method, "min_train_years": min_train_years,
                             "max_predictors": max_predictors, "min_gain": min_gain, "transform": transform},
                      sl(obs), sl(out["forecast"]), sl(out["clim"]), nan, sl(out["t1"]), sl(out["t2"]),
                      sl(out["p_below"]), sl(out["p_normal"]), sl(out["p_above"]), sl(sel), None, "sqrt" if transform == "sqrt" else "none")


def forecast(sm: Samples, spec: SubseasonalSpec, target_year: int, method: str = "ridge", *, meta: pd.DataFrame | None = None,
             min_train_years: int = 10, alpha: float = 1.0, max_predictors: int = 2, tol: float = 0.02,
             min_gain: float = S.MIN_GAIN, seed: int = 42, use_persistence: bool = True, max_zero_fraction: float = 0.25,
             **method_opts) -> pd.DataFrame:
    """Operational forecast for the initialization date of ``target_year`` (data before that date only)."""
    transform = TRANSFORM.get(sm.variable, "none")
    o = _options(alpha, max_predictors, tol, min_gain, seed, method_opts)
    hit = np.where((sm.meta["year"].values == target_year) & (sm.meta["shift"].values == 0))[0]
    if len(hit) == 0:
        raise ValueError(f"Samples were not built for year {target_year}")
    te = int(hit[0]); tr = _train_mask(sm, target_year, spec.init_date(target_year))
    if sm.meta.loc[tr, "year"].nunique() < min_train_years:
        raise ValueError(f"Only {sm.meta.loc[tr, 'year'].nunique()} training years before {spec.init_date(target_year).date()}; "
                         f"need >= {min_train_years}")
    if sm.X.iloc[te].isna().any():
        raise ValueError("Ocean predictors are incomplete for this initialization date: need complete monthly values for "
                         f"{[m.strftime('%Y-%m') for m in spec.sst_months(target_year)]}")
    rows = []
    for j, st in enumerate(sm.stations):
        r = _fit_station(sm, j, tr, te, method, o, transform, min_train_years * (1 + 2 * spec.pool_weeks) // 2 or 5,
                         use_persistence, max_zero_fraction)
        nan = float("nan")
        if r is None or "forecast" not in r:
            reason = (r or {}).get("label", "insufficient data")
            rows.append({"Station": st, "Forecast": nan, "Anomaly": nan, "Climatology": nan, "P_below": nan, "P_normal": nan,
                         "P_above": nan, "Tercile_lower": nan, "Tercile_upper": nan, "Predictors": reason, "N_train_years": 0})
            continue
        rows.append({"Station": st, "Forecast": r["forecast"], "Anomaly": r["forecast"] - r["clim"], "Climatology": r["clim"],
                     "P_below": r["pb"], "P_normal": r["pn"], "P_above": r["pa"], "Tercile_lower": r["t1"],
                     "Tercile_upper": r["t2"], "Predictors": r["label"], "N_train_years": r["n_years"]})
    df = pd.DataFrame(rows)
    df.insert(1, "Variable", sm.variable); df["Method"] = method
    d = spec.describe(target_year)
    for k in ("forecast_period", "init_date", "lead_weeks", "target_start", "target_end", "resolution"):
        df["Forecast_Period" if k == "forecast_period" else k] = d[k]
    if meta is not None:
        df = df.merge(meta[["station", "lat", "lon"]].rename(columns={"station": "Station"}), on="Station", how="left")
    first = [c for c in ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"] if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def run_station_subseasonal(obs: pd.DataFrame, stations: pd.DataFrame, indices: pd.DataFrame, spec: SubseasonalSpec,
                            target_year: int, variable: str, *, methods=("ridge",), ensemble_mode: str = "skill",
                            ensemble_pool: str = "pooled", min_frac: float = 0.9, **kw):
    """Hindcast + forecast for one variable and one lead window.  Several methods -> climatology, every method and
    their skill-weighted combination (forecasting.ensemble).  Returns (forecast table, Hindcast, {name: Hindcast})."""
    daily = prepare_daily(obs, variable)
    years = list(range(daily.index[0].year, target_year + 1))
    sm = build_samples(daily, indices, spec, years, variable, min_frac)
    methods = list(methods)
    if len(methods) == 1:
        hc = hindcast(sm, spec, methods[0], **kw)
        return forecast(sm, spec, target_year, methods[0], meta=stations, **kw), hc, {methods[0]: hc}
    from src.forecasting import ensemble
    return ensemble.run_multimodel(None, None, spec, target_year, methods, mode=ensemble_mode, pool=ensemble_pool,
                                   hindcast_fn=lambda m: hindcast(sm, spec, m, **kw),
                                   forecast_fn=lambda m: forecast(sm, spec, target_year, m, meta=stations, **kw))
