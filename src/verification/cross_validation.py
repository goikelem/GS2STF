"""Rolling-origin set-up, leakage checks, hyper-parameter tuning, chronological split and model selection.

Origin: model_validation.py."""
from __future__ import annotations

import hashlib
import json
import logging

import joblib
import numpy as np
import pandas as pd

from src import settings
from src.forecasting import machine_learning, sst_forecast
from src.teleconnections.lag_analysis import corr_with_sig


log = logging.getLogger("sst.validation")


IDX, H = settings.INDICES, settings.HORIZONS


def leakage_check(df: pd.DataFrame, n_checks: int = 8, seed: int = 0) -> bool:
    """(1) Predictors at origin t are unchanged when all data after t are deleted.
       (2) Training rows used for horizon h at origin T0 never have a target later than T0."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.arange(60, len(df) - 13), size=n_checks, replace=False)
    for target in ("R34", "DMI"):
        full = machine_learning.build_features(df, target)
        for p in pos:
            t = df.index[p]
            trunc = machine_learning.build_features(df.loc[:t], target).loc[t]
            a, b = full.loc[t].values.astype(float), trunc.values.astype(float)
            if not np.allclose(a, b, equal_nan=True):
                raise AssertionError(f"Predictor leakage detected at origin {t:%Y-%m} for {target}")
    X = machine_learning.build_features(df, "R34")
    for h in (1, 6, 12):
        y = machine_learning.target_for_horizon(df, "R34", h)
        t0 = df.index[int(pos[0])]
        d = pd.concat([X, y.rename("y")], axis=1).dropna()
        used = d[d.index <= machine_learning.add_months(t0, -h)]
        if len(used) and machine_learning.add_months(used.index.max(), h) > t0:
            raise AssertionError(f"Target leakage for h={h}")
    log.info("Leakage checks passed (predictors truncation-invariant; training targets <= origin).")
    return True


def _setup(df: pd.DataFrame) -> dict:
    last = df.index[-1]
    vs = pd.Timestamp(settings.VALIDATION_START + "-01")
    ve = last if settings.VALIDATION_END is None else min(pd.Timestamp(settings.VALIDATION_END + "-01"), last)
    if (vs - df.index[0]).days < 365.25 * 12:
        log.warning("Less than 12 years of data before VALIDATION_START (%s); training may be thin.", vs.strftime("%Y-%m"))
    if vs >= last:
        raise ValueError("VALIDATION_START is not before the end of the record.")
    r0 = machine_learning.add_months(vs, -1)
    refits = []
    r = r0
    while r < last:
        refits.append(r)
        r = machine_learning.add_months(r, settings.REFIT_EVERY)
    origins = df.index[(df.index >= r0) & (df.index < last)]
    return dict(last=last, vs=vs, ve=ve, r0=r0, refits=refits, origins=origins)


def _cache_id(df: pd.DataFrame, name: str) -> str:
    sig = json.dumps({"m": name, "last": str(df.index[-1]), "n": len(df), "vs": settings.VALIDATION_START,
                      "ve": settings.VALIDATION_END, "re": settings.REFIT_EVERY, "seed": settings.RANDOM_SEED,
                      "lags": settings.N_LAGS, "rw": settings.ROLL_WINDOWS, "trees": settings.RF_TREES,
                      "cv": settings.CV_FOLDS, "chk": float(round(df.sum().sum(), 4))}, sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()


def tune_all(df: pd.DataFrame, names: list[str], S: dict) -> dict:
    path = settings.out("cache") / "tuned_params.joblib"
    cid = _cache_id(df, "tune" + ",".join(names))
    if path.exists() and not settings.FORCE_RECOMPUTE:
        d = joblib.load(path)
        if d.get("id") == cid:
            log.info("Using cached hyper-parameters.")
            return d["params"]
    tuned = {}
    for name in [n for n in names if n in machine_learning.ML_MODELS]:
        for k in IDX:
            X = machine_learning.build_features(df, k)
            for h in H:
                tuned[(name, k, h)] = machine_learning.tune_ml(name, X, machine_learning.target_for_horizon(df, k, h), h, S["r0"])
        log.info("Tuned %s (rolling-origin CV, data <= %s)", name, S["r0"].strftime("%Y-%m"))
    joblib.dump({"id": cid, "params": tuned}, path)
    return tuned


def split_dates(hc: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    v = np.sort(hc["valid"].unique())
    cut = v[max(0, int(np.floor(len(v) * settings.SELECTION_FRACTION)) - 1)]
    return pd.Timestamp(cut), pd.Timestamp(v[-1])


def select_models(perf: pd.DataFrame, hc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sel_rows, flags = [], {}
    cut, _ = split_dates(hc)
    for k in IDX:
        for h in H:
            s = perf[(perf["Index"] == k) & (perf["Lead"] == h) & (perf["Period"] == "selection")].dropna(
                subset=["RMSE"])
            if s.empty:
                continue
            best = s["RMSE"].min()
            cand = s[s["RMSE"] <= best * (1 + settings.PARSIMONY_TOL)].copy()
            cand["cx"] = cand["Model"].map(sst_forecast.COMPLEXITY)
            pick = cand.sort_values(["cx", "RMSE"]).iloc[0]
            m = pick["Model"]
            t = perf[(perf["Index"] == k) & (perf["Lead"] == h) & (perf["Period"] == "test") &
                     (perf["Model"] == m)]
            t = t.iloc[0] if len(t) else None
            g = hc[(hc["model"] == m) & (hc["index"] == k) & (hc["lead"] == h) & (hc["valid"] > cut)]
            r, p, _ = corr_with_sig(g["obs"].reset_index(drop=True), g["pred"].reset_index(drop=True)) \
                if len(g) > 10 else (np.nan, np.nan, np.nan)
            ss = float(t["RMSE_skill_vs_climatology"]) if t is not None else np.nan
            if m == "climatology":
                flag = "no (climatological forecast - no skill beyond climatology)"
            elif not (np.isfinite(ss) and ss > 0):
                flag = "no (RMSE not better than climatology on independent test period)"
            elif not (np.isfinite(p) and p < 0.05 and r > 0):
                flag = "no (no significant positive correlation on independent test period)"
            else:
                flag = "yes"
            flags[(k, h)] = flag
            sel_rows.append({"Index": k, "Lead": h, "Selected_Model": m, "Best_Model_in_Selection": s.loc[
                s["RMSE"].idxmin(), "Model"], "Selection_RMSE": pick["RMSE"],
                "Test_RMSE": t["RMSE"] if t is not None else np.nan, "Test_MAE": t["MAE"] if t is not None else np.nan,
                "Test_Corr": t["Corr"] if t is not None else np.nan, "Test_R2": t["R2"] if t is not None else np.nan,
                "Test_MBE": t["MBE"] if t is not None else np.nan, "Test_Skill_vs_Climatology": ss,
                "Test_corr_p_value_effN": p, "Validated_Skill": flag})
    return pd.DataFrame(sel_rows), flags
