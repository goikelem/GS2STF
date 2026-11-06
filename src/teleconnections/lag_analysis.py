"""Autocorrelation, effective-sample-size correlation, lead-lag and seasonal correlations between indices.

Origin: statistical_analysis.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import settings


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n, c0 = len(x), np.dot(x, x) / len(x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / n / c0 for k in range(1, nlags + 1)])


def pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Partial autocorrelation by the Durbin-Levinson recursion."""
    r = acf(x, nlags)
    p = np.zeros(nlags + 1)
    p[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = p[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - np.sum(phi[k - 1, 1:k] * r[k - 1:0:-1])
        den = 1 - np.sum(phi[k - 1, 1:k] * r[1:k])
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        p[k] = phi[k, k]
    return p


def eff_n(x: pd.Series, y: pd.Series | None = None) -> float:
    """Effective sample size accounting for lag-1 autocorrelation (Bretherton et al. 1999)."""
    n = min(x.dropna().size, (y if y is not None else x).dropna().size)
    r1x = acf(x.values, 1)[1]
    r1y = acf((y if y is not None else x).values, 1)[1]
    neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    return float(np.clip(neff, 3, n))


def corr_with_sig(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Pearson r, two-sided p using effective N, and effective N."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 10:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
    ne = eff_n(d.iloc[:, 0], d.iloc[:, 1])
    t = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), ne - 2)), ne


def season_series(s: pd.Series, months: list[int]) -> pd.Series:
    """Season means indexed by year; a season containing Dec+Jan (e.g. DJF) is labelled by the Jan year."""
    d = s.dropna().to_frame("v")
    d["m"] = d.index.month
    d["y"] = d.index.year + ((d["m"] == 12) & (1 in months)).astype(int)
    d = d[d["m"].isin(months)]
    g = d.groupby("y")["v"].agg(["mean", "count"])
    return g.loc[g["count"] == len(months), "mean"]


def correlation_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    p = r.copy()
    for a in df.columns:
        for b in df.columns:
            r.loc[a, b], p.loc[a, b], _ = corr_with_sig(df[a], df[b])
    return r.round(3), p.round(4)


def lead_lag(df: pd.DataFrame, maxlag: int = 24) -> pd.DataFrame:
    """corr(X(t), Y(t+lag)); lag>0 means X leads Y by `lag` months."""
    rows = []
    for x in df.columns:
        for y in df.columns:
            if x == y:
                continue
            for lag in range(-maxlag, maxlag + 1):
                xs, ys = df[x], df[y].shift(-lag)
                r, p, ne = corr_with_sig(xs, ys)
                rows.append({"X": x, "Y": y, "Lag_months": lag, "Correlation": r, "p_value": p, "N_eff": ne})
    return pd.DataFrame(rows).round(4)


def best_lags(ll: pd.DataFrame) -> pd.DataFrame:
    idx = ll.groupby(["X", "Y"])["Correlation"].apply(lambda s: s.abs().idxmax())
    best = ll.loc[idx.values].copy()
    best["Interpretation"] = np.where(best["Lag_months"] > 0, best["X"] + " leads " + best["Y"],
                                      np.where(best["Lag_months"] < 0, best["Y"] + " leads " + best["X"],
                                               "simultaneous"))
    return best.reset_index(drop=True)


def seasonal_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Correlation matrices of seasonal means for JJAS, OND, DJF, MAM + lagged ENSO->IOD season pairs."""
    rows, mats = [], {}
    for sname in ("JJAS", "OND", "DJF", "MAM"):
        m = pd.DataFrame({k: season_series(df[k], settings.SEASONS[sname]) for k in df.columns}).dropna()
        mats[sname] = m.corr().round(3)
        for i, a in enumerate(df.columns):
            for b in list(df.columns)[i + 1:]:
                r, p, ne = corr_with_sig(m[a], m[b])
                rows.append({"Season": sname, "Index_A": a, "Index_B": b, "r": r, "p_value": p, "N_years": len(m)})
    # ENSO precursor -> IOD season (same calendar year; DJF leads by labelling with Jan year)
    pairs = [("R34", "JJAS", "DMI", "OND"), ("R34", "MAM", "DMI", "JJAS"), ("R34", "DJF", "DMI", "OND"),
             ("R34", "JJAS", "WIOD", "OND"), ("R34", "JJAS", "EIOD", "OND"), ("DMI", "JJAS", "R34", "DJF")]
    for a, sa, b, sb in pairs:
        xa = season_series(df[a], settings.SEASONS[sa])
        xb = season_series(df[b], settings.SEASONS[sb])
        if sa == "DJF" or (sb == "DJF" and sa != "DJF"):
            pass
        d = pd.concat([xa, xb], axis=1).dropna()
        r, p, ne = corr_with_sig(d.iloc[:, 0], d.iloc[:, 1])
        rows.append({"Season": f"{a}({sa}) -> {b}({sb})", "Index_A": a, "Index_B": b, "r": r, "p_value": p,
                     "N_years": len(d)})
    return pd.DataFrame(rows).round(4), mats
