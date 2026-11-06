"""Multi-model combination and hindcast-based method selection for seasonal forecasts.

Members are Hindcast objects made with the same predictors / years / validation scheme (see forecasting.seasonal),
typically ridge, PCR, CCA, random forest and climatology.  A forecast for year t is combined with weights that use
ONLY the members' hindcast skill in earlier years (ranked probability score history), so the combination itself is
verified out of sample:

  mode 'equal'  : plain average of the members' probabilities and forecasts
  mode 'skill'  : weights proportional to 1 / RPS^2 of each member over the history (climatology competes as a member,
                  which shrinks the ensemble toward climatology when nothing is skilful)
  mode 'select' : the single member with the lowest historical RPS ("hindcast-based model selection")

pool='pooled' scores members over all targets together (stable); pool='per_target' scores each station / cell
separately (flexible but noisier with ~30 seasons).  With fewer than ``min_history`` earlier hindcast seasons all
members get equal weight.  Probabilities are the weighted average (linear opinion pool); ``Ensemble_spread`` is the
weighted spread of the member forecasts.  Only the rolling-origin scheme supports 'skill' / 'select' (a blocked
hindcast uses future years for training, so its history is not a valid basis for weights).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.forecasting import seasonal as S
from src.verification import probabilistic_scores as V

log = logging.getLogger("s2s.ensemble")
MODES = ("equal", "skill", "select")


def rps_matrix(h: S.Hindcast) -> np.ndarray:
    """RPS per (year, target); NaN where the forecast or observation is missing."""
    pr, cat = h.probs, h.obs_category
    ok = (cat >= 0) & ~np.isnan(pr).any(-1)
    out = np.full(cat.shape, np.nan)
    out[ok] = V.rps(pr[ok], cat[ok])
    return out


def _weights(rps_hist: np.ndarray, avail: np.ndarray, mode: str, min_history: int, pool: str) -> np.ndarray:
    """rps_hist: (members, years, targets) history; avail: (members, targets) bool.  Returns (members, targets)."""
    K, _, m = rps_hist.shape
    w = avail.astype(float)
    if mode != "equal":
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if pool == "pooled":
                valid = ~np.isnan(rps_hist).all(axis=2)                       # members x years with any score
                n_hist = valid.all(axis=0).sum() if K else 0
                sc = np.nanmean(np.where(valid[:, :, None], rps_hist, np.nan), axis=(1, 2))
                sc = np.repeat(sc[:, None], m, 1)
                enough = np.full(m, n_hist >= min_history)
            else:
                cnt = (~np.isnan(rps_hist)).sum(1)                              # members x targets
                sc = np.nanmean(rps_hist, axis=1)
                enough = (cnt >= min_history).all(axis=0)
        sc = np.where(avail & np.isfinite(sc), sc, np.nan)
        if mode == "skill":
            ww = np.where(np.isfinite(sc), 1.0 / np.maximum(sc, 1e-6) ** 2, 0.0)
        else:
            ww = np.zeros_like(sc)
            has = np.isfinite(sc).any(0)
            best = np.nanargmin(np.where(np.isfinite(sc), sc, np.inf), axis=0)
            ww[best[has], np.where(has)[0]] = 1.0
        w = np.where(enough[None, :], ww, w)
        w = np.where(w.sum(0)[None, :] > 0, w, avail.astype(float))
    tot = w.sum(0, keepdims=True)
    return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)


def _mix(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted nan-aware combination over axis 0; values (K, ...), w (K, m) broadcast on the target axis."""
    ww = w.reshape(w.shape + (1,) * (values.ndim - w.ndim)) if values.ndim > w.ndim else w
    v = np.where(np.isnan(values), 0.0, values)
    tot = np.where(np.isnan(values), 0.0, ww).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (v * ww).sum(0) / tot, np.nan)


def combine_hindcasts(members: dict, mode: str = "skill", pool: str = "pooled", min_history: int = 5):
    """Out-of-sample multi-model hindcast.  Returns (Hindcast, weights) with weights = years x members (mean over targets)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    names = list(members)
    if mode != "equal" and any(h.settings["scheme"] != "rolling" for h in members.values()):
        raise ValueError("'skill' / 'select' need rolling-origin hindcasts; use mode='equal' with the blocked scheme")
    years = sorted(set.intersection(*[set(h.forecast.index) for h in members.values()]))
    H = members
    ref = H[names[0]]
    cols = ref.obs.columns
    F = np.stack([H[k].forecast.loc[years].values for k in names])             # K, T, m
    P = np.stack([H[k].probs[[list(H[k].forecast.index).index(y) for y in years]] for k in names])   # K, T, m, 3
    R = np.stack([rps_matrix(H[k])[[list(H[k].forecast.index).index(y) for y in years]] for k in names])
    out_f = np.full(F.shape[1:], np.nan); out_p = np.full(P.shape[1:], np.nan)
    spread = np.full(F.shape[1:], np.nan); W = np.zeros((len(years), len(names)))
    top = np.full(F.shape[1:], "", dtype=object)
    for i, y in enumerate(years):
        avail = ~np.isnan(P[:, i, :, :]).any(-1)                               # members x targets
        w = _weights(R[:, :i, :], avail, mode, min_history, pool)
        out_f[i] = _mix(F[:, i, :], w)
        out_p[i] = _mix(P[:, i, :, :], w)
        mu = out_f[i]
        spread[i] = np.sqrt(np.nan_to_num(_mix((F[:, i, :] - mu[None]) ** 2, w)))
        W[i] = w.mean(1)
        top[i] = np.array(names, dtype=object)[w.argmax(0)]
    sl = lambda d: d.loc[years]
    def frame(a): return pd.DataFrame(a, index=years, columns=cols)
    ens = S.Hindcast(ref.spec, {**ref.settings, "method": f"ensemble[{mode},{pool}]", "members": names},
                     sl(ref.obs), frame(out_f), sl(ref.clim), frame(np.full(out_f.shape, np.nan)), sl(ref.t1), sl(ref.t2),
                     frame(out_p[..., 0]), frame(out_p[..., 1]), frame(out_p[..., 2]),
                     pd.DataFrame(top, index=years, columns=cols),
                     None, ref.transform)
    ens.spread = frame(spread)
    return ens, pd.DataFrame(W, index=years, columns=names)


def combine_forecasts(member_fc: dict, member_hc: dict, target_year: int, mode: str = "skill", pool: str = "pooled",
                      min_history: int = 5) -> pd.DataFrame:
    """Combine operational forecast tables (one per member).  Weights use the members' hindcast skill in seasons
    strictly BEFORE ``target_year`` (a hindcast of the target season itself is never used)."""
    names = list(member_fc)
    base = member_fc[names[0]].copy()
    cols = list(base["Station"])
    for k in names:
        if list(member_fc[k]["Station"]) != cols:
            raise ValueError("members must forecast the same stations in the same order")
    R = np.stack([rps_matrix(member_hc[k])[np.asarray(member_hc[k].forecast.index) < target_year] for k in names])   # K, T, m
    avail = np.stack([~member_fc[k][["P_below", "P_normal", "P_above"]].isna().any(axis=1).values for k in names])
    w = _weights(R, avail, mode, min_history, pool)
    F = np.stack([member_fc[k]["Forecast"].values for k in names])
    P = np.stack([member_fc[k][["P_below", "P_normal", "P_above"]].values for k in names])
    f = _mix(F, w); p = _mix(P, w)
    base["Forecast"] = f; base["Anomaly"] = f - base["Climatology"].values
    for j, c in enumerate(["P_below", "P_normal", "P_above"]):
        base[c] = p[:, j]
    base["Ensemble_spread"] = np.sqrt(np.nan_to_num(_mix((F - f[None]) ** 2, w)))
    base["Method"] = f"ensemble[{mode},{pool}]"
    base["Predictors"] = [", ".join(f"{k}({member_fc[k]['Predictors'].iloc[j]})" for k in names if w[names.index(k), j] > 0)
                          for j in range(len(cols))]
    for i, k in enumerate(names):
        base[f"Forecast_{k}"] = F[i]; base[f"Weight_{k}"] = w[i]
    return base


def run_multimodel(Y, X, spec, target_year, methods=("ridge", "pcr", "cca", "rf"), *, transform="none",
                   scheme="rolling", mode="skill", pool="pooled", min_history=5, persist=None, meta=None,
                   variable="", **kw):
    """Hindcast + operational forecast for several methods and their combination.

    Returns (forecast table, final Hindcast, {name: Hindcast} incl. 'climatology' and 'ensemble').
    """
    names = ["climatology"] + [m for m in methods if m != "climatology"]
    hcs = {m: S.hindcast(Y, X, spec, scheme=scheme, transform=transform, method=m,
                         persist=persist if m == names[0] else None, **kw) for m in names}
    fkw = {k: v for k, v in kw.items() if k != "block_buffer"}
    fcs = {m: S.forecast(Y, X, spec, target_year, transform=transform, meta=meta, variable=variable, method=m, **fkw)
           for m in names}
    eff_mode = mode if scheme == "rolling" else "equal"
    if eff_mode != mode:
        log.warning("blocked hindcast: ensemble weights fall back to 'equal'")
    ens_h, _ = combine_hindcasts({m: hcs[m] for m in names}, eff_mode, pool, min_history)
    fc = combine_forecasts(fcs, hcs, target_year, eff_mode, pool, min_history)
    hcs["ensemble"] = ens_h
    return fc, ens_h, hcs
