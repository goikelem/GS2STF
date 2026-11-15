"""Integration of dynamical-model (NMME, SEAS5, C3S, ...) seasonal guidance.

The model is NOT run here: you supply its forecasts (and its re-forecast archive for past years) as CSV files that you
produced from the model output, all issued at the SAME initialization month as the forecast you are making.

Two ways to use the guidance
  1. index predictors  (``--dynamical-indices FILE``): the model's forecast of ocean indices for the target season
     (e.g. Nino3.4 / DMI), one row per season_year, e.g.
         season_year,init_month,R34,DMI
         1993,5,0.4,-0.1  ...  2026,5,1.1,0.3
     The columns enter the regression as extra predictors named DYN_<column>.  This is the "forecast SST" input: the
     value is what the model predicted at the initialization date, never the observed SST of the verification period.
  2. station guidance (``--dynamical-stations FILE``): the model's forecast of the station variable itself (the seasonal
     total / mean at the station, e.g. sampled from the model grid), long format
         Station,season_year,init_month,value
     It becomes an extra ensemble member, "dynamical": observation regressed on the model forecast (a calibration /
     model-output-statistics step that removes mean and scale bias), fitted on EARLIER years only, kept only if it beats
     climatology out of sample, with tercile probabilities from the calibrated regression.  The skill-weighted ensemble
     then decides how much to trust it from its hindcast record.
Leak control: the init_month column (when present) must equal the forecast's init month, otherwise loading fails; the
re-forecast value for year t is by construction information available at the initialization date of year t.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.forecasting import seasonal as S

log = logging.getLogger("s2s.dynamical")


def _check_init(df: pd.DataFrame, spec, path) -> None:
    if "init_month" in df.columns:
        bad = sorted(set(df["init_month"].dropna().astype(int)) - {spec.init_month})
        if bad:
            raise ValueError(f"{path}: init_month {bad} differs from the forecast's init month {spec.init_month}; supply model "
                             f"forecasts issued in the same month (mixing lead times would make the predictor inconsistent)")
    else:
        log.warning("%s has no init_month column: assuming all rows were issued in month %d", path, spec.init_month)


def load_index_forecasts(path, spec) -> pd.DataFrame:
    """Model index forecasts -> predictor table (index season_year, columns DYN_<name>)."""
    df = pd.read_csv(path)
    if "season_year" not in df.columns:
        raise ValueError(f"{path}: needs a season_year column")
    _check_init(df, spec, path)
    cols = [c for c in df.columns if c not in ("season_year", "init_month")]
    if not cols:
        raise ValueError(f"{path}: no predictor columns")
    if df["season_year"].duplicated().any():
        raise ValueError(f"{path}: duplicate season_year rows")
    out = df.set_index(df["season_year"].astype(int))[cols].apply(pd.to_numeric, errors="coerce")
    out.columns = [f"DYN_{c}" for c in cols]
    return out.sort_index()


def load_station_forecasts(path, spec, variable: str | None = None, value_column: str | None = None) -> pd.DataFrame:
    """Model station forecasts -> matrix (season_year x station).

    If the file has a Variable column only the rows of ``variable`` are used; without it the whole file is taken to be
    the forecast of ``variable`` (the caller decides which variable that is)."""
    df = pd.read_csv(path)
    vcol = next((c for c in df.columns if c.lower() == "variable"), None)
    if vcol is not None and variable is not None:
        from src.data_ingestion.station_csv_loader import normalise_variable
        df = df[df[vcol].map(normalise_variable) == variable]
        if df.empty:
            raise ValueError(f"{path}: no rows for variable '{variable}'")
    low = {c.lower(): c for c in df.columns}
    sc = low.get("station"); yc = low.get("season_year")
    vc = value_column or next((low[k] for k in ("value", "forecast") if k in low), None)
    if not (sc and yc and vc):
        raise ValueError(f"{path}: needs Station, season_year and value (or Forecast) columns; found {list(df.columns)}")
    _check_init(df, spec, path)
    df = df.assign(**{sc: df[sc].astype(str).str.strip(), vc: pd.to_numeric(df[vc], errors="coerce")})
    if df.duplicated([sc, yc]).any():
        raise ValueError(f"{path}: duplicate (Station, season_year) rows")
    return df.pivot(index=yc, columns=sc, values=vc).sort_index().astype(float)


# ---------------------------------------------------------------- calibrated station member
def _station_step(Ytr, Mtr, mt, transform, alpha, tol, min_gain, min_train):
    """One station, one forecast year: calibrate obs on the model forecast; returns dict or None."""
    fwd, inv = S.TRANSFORMS[transform]
    ok = np.isfinite(Ytr) & np.isfinite(Mtr)
    if ok.sum() < min_train or not np.isfinite(mt):
        return None
    y, m = Ytr[ok], Mtr[ok]
    F, ft = fwd(m)[:, None], np.array([[float(fwd(np.array(mt)))]])
    mean_t, sg, lab = S._select_and_predict(F, ft, fwd(y)[:, None], [(), (0,)], ["climatology", "dynamical"], alpha, tol, min_gain)
    t1, t2 = np.percentile(y, [100 / 3, 200 / 3])
    pb, pn, pa = S._probabilities(mean_t, sg, fwd(t1), fwd(t2))
    return {"forecast": float(S.retransform_mean(transform, mean_t[0], sg[0])), "clim": float(y.mean()), "t1": float(t1), "t2": float(t2),
            "sigma_t": float(sg[0]), "pb": float(pb[0]), "pn": float(pn[0]), "pa": float(pa[0]), "label": str(lab[0]),
            "n": int(ok.sum())}


def hindcast(Y: pd.DataFrame, M: pd.DataFrame, spec, *, transform: str = "none", min_train: int = 15, alpha: float = 1.0,
             tol: float = 0.02, min_gain: float = S.MIN_GAIN) -> S.Hindcast:
    """Rolling-origin hindcast of the calibrated dynamical member (training years strictly before each test year)."""
    cols = list(Y.columns); M = M.reindex(columns=cols)
    years = sorted(set(Y.index) & set(M.index)); Y, M = Y.loc[years], M.loc[years]
    keys = ["forecast", "clim", "sigma_t", "t1", "t2", "p_below", "p_normal", "p_above"]
    out = {k: pd.DataFrame(np.nan, index=years, columns=cols) for k in keys}
    sel = pd.DataFrame("", index=years, columns=cols, dtype=object)
    for t in years:
        tr = [y for y in years if y < t]
        if len(tr) < min_train:
            continue
        for st in cols:
            r = _station_step(Y.loc[tr, st].values, M.loc[tr, st].values, M.loc[t, st], transform, alpha, tol, min_gain, min_train)
            if r is None:
                continue
            for k, rk in zip(keys, ["forecast", "clim", "sigma_t", "t1", "t2", "pb", "pn", "pa"]):
                out[k].loc[t, st] = r[rk]
            sel.loc[t, st] = r["label"]
    done = out["forecast"].notna().any(axis=1)
    sl = lambda d: d.loc[done]
    return S.Hindcast(spec, {"scheme": "rolling", "method": "dynamical", "min_train": min_train, "min_gain": min_gain,
                             "transform": transform}, sl(Y), sl(out["forecast"]), sl(out["clim"]), sl(out["sigma_t"]),
                      sl(out["t1"]), sl(out["t2"]), sl(out["p_below"]), sl(out["p_normal"]), sl(out["p_above"]), sl(sel),
                      None, transform)


def forecast(Y: pd.DataFrame, M: pd.DataFrame, spec, target_year: int, *, transform: str = "none", min_train: int = 15,
             alpha: float = 1.0, tol: float = 0.02, min_gain: float = S.MIN_GAIN, meta=None, variable: str = "") -> pd.DataFrame:
    """Forecast table (same layout as seasonal.forecast) for the calibrated dynamical member."""
    if target_year not in M.index:
        raise ValueError(f"No dynamical-model forecast for season_year {target_year}")
    cols = list(Y.columns); tr = [y for y in Y.index if y < target_year]
    rows = []
    nan = float("nan")
    for st in cols:
        mt = M.loc[target_year, st] if st in M.columns else nan
        r = _station_step(Y.loc[tr, st].values, M.reindex(index=tr)[st].values if st in M.columns else np.full(len(tr), nan),
                          mt, transform, alpha, tol, min_gain, min_train)
        if r is None:
            rows.append({"Station": st, "Forecast": nan, "Anomaly": nan, "Climatology": nan, "P_below": nan, "P_normal": nan,
                         "P_above": nan, "Tercile_lower": nan, "Tercile_upper": nan, "Predictors": "no model forecast / too short",
                         "N_train_years": 0})
        else:
            rows.append({"Station": st, "Forecast": r["forecast"], "Anomaly": r["forecast"] - r["clim"], "Climatology": r["clim"],
                         "P_below": r["pb"], "P_normal": r["pn"], "P_above": r["pa"], "Tercile_lower": r["t1"],
                         "Tercile_upper": r["t2"], "Predictors": r["label"], "N_train_years": r["n"]})
    df = pd.DataFrame(rows); df.insert(1, "Variable", variable); df["Method"] = "dynamical"
    d = spec.describe(target_year)
    df["Forecast_Period"] = d["forecast_period"]
    for k in ("init_date", "lead_months", "target_start", "target_end"):
        df[k] = d[k]
    if meta is not None:
        df = df.merge(meta[["station", "lat", "lon"]].rename(columns={"station": "Station"}), on="Station", how="left")
    first = [c for c in ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"] if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]
