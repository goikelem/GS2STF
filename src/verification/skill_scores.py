"""Deterministic skill metrics and hindcast evaluation.

Origin: model_validation.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import settings
from src.verification.cross_validation import split_dates


IDX, H = settings.INDICES, settings.HORIZONS


def metrics(obs, pred, mape_min_abs: float = 0.5) -> dict:
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    n = len(o)
    if n < 3:
        return dict(N=n, RMSE=np.nan, MAE=np.nan, MAPE_pct=np.nan, Corr=np.nan, R2=np.nan, MBE=np.nan)
    err = p - o
    sst = np.sum((o - o.mean()) ** 2)
    big = np.abs(o) >= mape_min_abs     # MAPE is meaningless near zero anomalies -> only |obs| >= 0.5 degC
    return dict(N=n, RMSE=float(np.sqrt(np.mean(err ** 2))), MAE=float(np.mean(np.abs(err))),
                MAPE_pct=float(100 * np.mean(np.abs(err[big] / o[big]))) if big.sum() >= 5 else np.nan,
                Corr=float(np.corrcoef(o, p)[0, 1]) if np.std(p) > 0 and np.std(o) > 0 else np.nan,
                R2=float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
                MBE=float(np.mean(err)))      # MBE = mean(forecast - observation); >0 means over-forecast


def evaluate(hc: pd.DataFrame) -> pd.DataFrame:
    cut, _ = split_dates(hc)
    hc = hc.assign(period=np.where(hc["valid"] <= cut, "selection", "test"))
    rows = []
    for (m, k, h), g in hc.groupby(["model", "index", "lead"]):
        for per, gg in list(g.groupby("period")) + [("all", g)]:
            r = metrics(gg["obs"], gg["pred"])
            rows.append({"Model": m, "Index": k, "Lead": h, "Period": per, **r})
    out = pd.DataFrame(rows)
    # skill score relative to the seasonal-climatology benchmark
    ref = (out[out["Model"] == "climatology"][["Index", "Lead", "Period", "RMSE"]]
           .rename(columns={"RMSE": "RMSE_clim"}))
    out = out.merge(ref, on=["Index", "Lead", "Period"], how="left")
    out["RMSE_skill_vs_climatology"] = 1 - out["RMSE"] / out["RMSE_clim"]
    return out.drop(columns="RMSE_clim")
