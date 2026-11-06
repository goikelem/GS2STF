"""Deterministic and probabilistic verification of seasonal hindcasts (tercile categories).

Scores (all out-of-sample, see forecasting.seasonal.hindcast)
  deterministic : correlation, RMSE, MAE, bias, RMSE skill vs climatology (and vs persistence if supplied)
  categorical   : Heidke skill score of the most-probable category
  probabilistic : RPSS and Brier skill score vs climatological (1/3,1/3,1/3), Gaussian CRPS and CRPSS,
                  ROC area, reliability table
Skill scores are positive when the forecast beats climatology.  ``bootstrap`` resamples seasons (years) with
replacement, optionally in blocks, to give confidence intervals of the pooled scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

CLIM = np.full(3, 1.0 / 3.0)


# ---------------------------------------------------------------- scalar scores on (N,) / (N,3) arrays
def rps(probs: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Ranked probability score per forecast (3 ordered categories)."""
    obs = np.eye(3)[cat]
    return ((np.cumsum(probs, -1) - np.cumsum(obs, -1)) ** 2).sum(-1)


def brier(p: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((p - outcome) ** 2))


def heidke(fc_cat: np.ndarray, obs_cat: np.ndarray) -> float:
    """3-category Heidke skill score: (hits - expected) / (N - expected), expected by chance given the marginals."""
    n = len(fc_cat)
    if n == 0:
        return np.nan
    hits = np.sum(fc_cat == obs_cat)
    exp = sum(np.sum(fc_cat == k) * np.sum(obs_cat == k) for k in range(3)) / n
    return float((hits - exp) / (n - exp)) if n != exp else np.nan


def crps_gaussian(obs, mu, sigma):
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma^2)."""
    z = (obs - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def roc_auc(p: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney form); 0.5 = no discrimination."""
    pos, neg = p[outcome == 1], p[outcome == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def reliability_table(p: np.ndarray, outcome: np.ndarray, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> pd.DataFrame:
    """Forecast-probability bins vs observed relative frequency (plot as a reliability diagram)."""
    b = np.digitize(p, bins) - 1
    rows = []
    for k in range(len(bins) - 1):
        m = b == k
        if m.any():
            rows.append({"bin_low": bins[k], "bin_high": min(bins[k + 1], 1.0), "mean_forecast_prob": p[m].mean(),
                         "observed_frequency": outcome[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Hindcast level
def _flat(h):
    """Valid (year, target) pairs flattened: obs, forecast, clim, sigma(original units), probs, cat, year index."""
    obs, fc, cl = h.obs.values, h.forecast.values, h.clim.values
    cat = h.obs_category; pr = h.probs
    ok = (cat >= 0) & ~np.isnan(fc) & ~np.isnan(pr).any(-1)
    yr = np.repeat(np.arange(obs.shape[0])[:, None], obs.shape[1], 1)
    return {"obs": obs[ok], "fc": fc[ok], "clim": cl[ok], "probs": pr[ok], "cat": cat[ok], "year": yr[ok],
            "mask": ok}


def deterministic_scores(obs, fc, clim, persist=None) -> dict:
    ok = ~(np.isnan(obs) | np.isnan(fc) | np.isnan(clim))
    o, f, c = obs[ok], fc[ok], clim[ok]
    if len(o) < 3:
        return {"n": int(len(o))}
    rmse = float(np.sqrt(np.mean((f - o) ** 2))); rmse_c = float(np.sqrt(np.mean((c - o) ** 2)))
    out = {"n": int(len(o)), "correlation": float(np.corrcoef(f, o)[0, 1]) if f.std() > 0 and o.std() > 0 else np.nan,
           "rmse": rmse, "mae": float(np.mean(np.abs(f - o))), "bias": float(np.mean(f - o)),
           "rmse_climatology": rmse_c, "rmse_skill": 1 - rmse / rmse_c if rmse_c > 0 else np.nan}
    if persist is not None:
        p = persist[ok]; good = ~np.isnan(p)
        if good.sum() >= 3:
            rp = float(np.sqrt(np.mean((p[good] - o[good]) ** 2)))
            out["rmse_persistence"] = rp
            out["rmse_skill_vs_persistence"] = 1 - float(np.sqrt(np.mean((f[good] - o[good]) ** 2))) / rp if rp > 0 else np.nan
    return out


def probabilistic_scores(probs, cat, obs=None, fc=None, sigma=None) -> dict:
    n = len(cat)
    if n < 3:
        return {"n_prob": int(n)}
    rp, rp_c = rps(probs, cat).mean(), rps(np.tile(CLIM, (n, 1)), cat).mean()
    out = {"n_prob": int(n), "rpss": float(1 - rp / rp_c) if rp_c > 0 else np.nan,
           "hss": heidke(np.argmax(probs, -1), cat)}
    for k, nm in enumerate(["below", "normal", "above"]):
        o = (cat == k).astype(float)
        bs, bc = brier(probs[:, k], o), brier(np.full(n, 1 / 3), o)
        out[f"brier_ss_{nm}"] = float(1 - bs / bc) if bc > 0 else np.nan
        out[f"roc_auc_{nm}"] = roc_auc(probs[:, k], o)
    if obs is not None and sigma is not None:
        c = crps_gaussian(obs, fc, sigma).mean()
        out["crps"] = float(c)
    return out


def summarize(h) -> dict:
    """Per-target and pooled verification tables for a Hindcast.  Returns {'per_target', 'pooled', 'reliability'}."""
    rows = []
    for j, name in enumerate(h.obs.columns):
        o, f, c = h.obs.values[:, j], h.forecast.values[:, j], h.clim.values[:, j]
        pr, cat = h.probs[:, j, :], h.obs_category[:, j]
        ok = (cat >= 0) & ~np.isnan(pr).any(-1)
        row = {"station": name}
        row.update(deterministic_scores(o, f, c, None if h.persistence is None else h.persistence.values[:, j]))
        sg = h.sigma_t.values[:, j]
        row.update(probabilistic_scores(pr[ok], cat[ok], *( (o[ok], f[ok], sg[ok]) if h.transform == "none" else (None, None, None))))
        rows.append(row)
    per = pd.DataFrame(rows)
    d = _flat(h)
    pooled = {"n_targets": int(h.obs.shape[1]), "n_years": int(h.obs.shape[0])}
    pooled.update(probabilistic_scores(d["probs"], d["cat"]))
    pooled["mean_target_correlation"] = float(per["correlation"].mean()) if "correlation" in per else np.nan
    pooled["mean_target_rmse_skill"] = float(per["rmse_skill"].mean()) if "rmse_skill" in per else np.nan
    rel = pd.concat([reliability_table(d["probs"][:, k], (d["cat"] == k).astype(float)).assign(category=nm)
                     for k, nm in enumerate(["below", "normal", "above"])], ignore_index=True) if len(d["cat"]) else pd.DataFrame()
    return {"per_target": per, "pooled": pd.Series(pooled), "reliability": rel}


def bootstrap(h, n_boot: int = 1000, block: int = 1, level: float = 0.90, seed: int = 0) -> pd.DataFrame:
    """Confidence intervals of pooled RPSS, HSS and mean RMSE skill by resampling years (in blocks of ``block``)."""
    rng = np.random.default_rng(seed)
    ny = h.obs.shape[0]
    probs, cat = h.probs, h.obs_category
    o, f, c = h.obs.values, h.forecast.values, h.clim.values
    starts = np.arange(max(ny - block + 1, 1))
    k = int(np.ceil(ny / block))
    stats = {"rpss": [], "hss": [], "rmse_skill": []}
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, ny)) for s in rng.choice(starts, k)])[:ny]
        p, ct = probs[idx].reshape(-1, 3), cat[idx].ravel()
        ok = (ct >= 0) & ~np.isnan(p).any(-1)
        if ok.sum() < 3:
            continue
        r = probabilistic_scores(p[ok], ct[ok])
        stats["rpss"].append(r["rpss"]); stats["hss"].append(r["hss"])
        oo, ff, cc = o[idx].ravel(), f[idx].ravel(), c[idx].ravel()
        m = ~(np.isnan(oo) | np.isnan(ff) | np.isnan(cc))
        if m.sum() >= 3:
            e, ec = np.sqrt(np.mean((ff[m] - oo[m]) ** 2)), np.sqrt(np.mean((cc[m] - oo[m]) ** 2))
            stats["rmse_skill"].append(1 - e / ec if ec > 0 else np.nan)
    a = (1 - level) / 2
    return pd.DataFrame({k: {"mean": np.nanmean(v), f"lo_{int(level*100)}": np.nanquantile(v, a),
                             f"hi_{int(level*100)}": np.nanquantile(v, 1 - a), "n_boot": len(v)}
                         for k, v in stats.items() if len(v)}).T


def compare(hcs: dict, ref: str = "climatology", n_boot: int = 500, block: int = 2, level: float = 0.90, seed: int = 0) -> pd.DataFrame:
    """Method comparison table: pooled RPSS (with bootstrap CI), Heidke, mean correlation / RMSE skill, and the paired
    RPSS difference to ``ref`` (same resampled years for every method, so differences are meaningful).
    Years are resampled in blocks of ``block``; rows are sorted by RPSS."""
    names = list(hcs)
    years = sorted(set.intersection(*[set(h.forecast.index) for h in hcs.values()]))
    pos = {k: [list(h.forecast.index).index(y) for y in years] for k, h in hcs.items()}
    cat = {k: h.obs_category[pos[k]] for k, h in hcs.items()}
    pr = {k: h.probs[pos[k]] for k, h in hcs.items()}
    ny = len(years)
    rng = np.random.default_rng(seed)
    starts = np.arange(max(ny - block + 1, 1)); nb = int(np.ceil(ny / block))
    def rpss_of(k, idx):
        p, c = pr[k][idx].reshape(-1, 3), cat[k][idx].ravel()
        ok = (c >= 0) & ~np.isnan(p).any(-1)
        if ok.sum() < 3:
            return np.nan
        return 1 - rps(p[ok], c[ok]).mean() / rps(np.tile(CLIM, (ok.sum(), 1)), c[ok]).mean()
    boots = {k: [] for k in names}
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s0, min(s0 + block, ny)) for s0 in rng.choice(starts, nb)])[:ny]
        for k in names:
            boots[k].append(rpss_of(k, idx))
    a = (1 - level) / 2; lv = int(level * 100)
    rows = []
    for k in names:
        pooled = summarize_subset(hcs[k], pos[k])
        b = np.array(boots[k]); d = b - np.array(boots[ref]) if ref in boots else None
        if k == "climatology":                                   # a slowly drifting mean has no meaningful correlation
            pooled = {**pooled, "mean_correlation": np.nan}
        row = {"method": k, "n_seasons": ny, "rpss": rpss_of(k, np.arange(ny)), f"rpss_lo_{lv}": np.nanquantile(b, a),
               f"rpss_hi_{lv}": np.nanquantile(b, 1 - a), **pooled}
        if d is not None and k != ref:
            row[f"rpss_diff_vs_{ref}"] = float(np.nanmean(d))
            row[f"diff_lo_{lv}"] = np.nanquantile(d, a); row[f"diff_hi_{lv}"] = np.nanquantile(d, 1 - a)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rpss", ascending=False).reset_index(drop=True)


def summarize_subset(h, rows) -> dict:
    """Hss, mean per-target correlation and RMSE skill on the given hindcast rows."""
    o, f, c = h.obs.values[rows], h.forecast.values[rows], h.clim.values[rows]
    p, ct = h.probs[rows].reshape(-1, 3), h.obs_category[rows].ravel()
    ok = (ct >= 0) & ~np.isnan(p).any(-1)
    corr = [deterministic_scores(o[:, j], f[:, j], c[:, j]).get("correlation", np.nan) for j in range(o.shape[1])]
    sk = [deterministic_scores(o[:, j], f[:, j], c[:, j]).get("rmse_skill", np.nan) for j in range(o.shape[1])]
    return {"hss": heidke(np.argmax(p[ok], -1), ct[ok]) if ok.sum() >= 3 else np.nan,
            "mean_correlation": float(np.nanmean(corr)) if len(corr) else np.nan,
            "mean_rmse_skill": float(np.nanmean(sk)) if len(sk) else np.nan}
