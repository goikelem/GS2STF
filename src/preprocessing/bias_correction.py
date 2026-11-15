"""Bias correction of a model / gridded product against observations.

Typical uses: a satellite or reanalysis product (CHIRPS, ERA5) against station records, or dynamical-model
hindcasts against observations.  Correctors are fitted on a TRAINING period only and then applied elsewhere; use
``rolling_correct`` for hindcasts so that every year is corrected with parameters learned from earlier years only.

Methods
  LinearScaling    additive (temperature), multiplicative (precipitation) or mean-and-variance matching
  QuantileMapping  empirical quantile mapping with additive or multiplicative tails (outside the training range the
                   correction at the nearest training quantile is kept, never extrapolated linearly)
Both accept a ``groups`` label per sample (e.g. calendar month) through ``fit_grouped`` / ``apply_grouped``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


class LinearScaling:
    """kind: 'additive' (obs mean - model mean), 'multiplicative' (obs mean / model mean) or 'variance'
    (match mean and standard deviation)."""

    def __init__(self, kind: str = "additive"):
        if kind not in ("additive", "multiplicative", "variance"):
            raise ValueError("kind must be 'additive', 'multiplicative' or 'variance'")
        self.kind = kind

    def fit(self, obs, mod):
        o, m = _pair(obs, mod)
        if len(o) < 3:
            raise ValueError("need at least 3 paired values")
        self.mo, self.mm = o.mean(), m.mean()
        self.so, self.sm = o.std(ddof=1), m.std(ddof=1)
        return self

    def apply(self, mod):
        x = np.asarray(mod, dtype=float)
        if self.kind == "additive":
            return x + (self.mo - self.mm)
        if self.kind == "multiplicative":
            return x * (self.mo / self.mm if abs(self.mm) > EPS else 1.0)
        scale = self.so / self.sm if self.sm > EPS else 1.0
        return (x - self.mm) * scale + self.mo


class QuantileMapping:
    """Empirical quantile mapping model -> observed distribution.

    kind 'additive' keeps the offset obs-model at the extreme quantiles beyond the training range (temperature);
    'multiplicative' keeps the ratio (precipitation, never negative).
    """

    def __init__(self, n_quantiles: int = 50, kind: str = "additive"):
        if kind not in ("additive", "multiplicative"):
            raise ValueError("kind must be 'additive' or 'multiplicative'")
        self.n, self.kind = n_quantiles, kind

    def fit(self, obs, mod):
        o, m = _pair(obs, mod)
        if len(o) < 10:
            raise ValueError("quantile mapping needs at least 10 paired values")
        p = np.linspace(0.0, 1.0, min(self.n, len(o)))
        self.qo, self.qm = np.quantile(o, p), np.maximum.accumulate(np.quantile(m, p))
        return self

    def apply(self, mod):
        x = np.asarray(mod, dtype=float)
        # ties in the model quantiles: use the first occurrence so np.interp stays well defined
        qm, first = np.unique(self.qm, return_index=True); qo = self.qo[first]
        out = np.interp(x, qm, qo) if len(qm) > 1 else np.full_like(x, qo[0])
        lo, hi = x < qm[0], x > qm[-1]
        if self.kind == "additive":
            out = np.where(lo, x + (qo[0] - qm[0]), out); out = np.where(hi, x + (qo[-1] - qm[-1]), out)
        else:
            rl = qo[0] / qm[0] if abs(qm[0]) > EPS else 1.0; rh = qo[-1] / qm[-1] if abs(qm[-1]) > EPS else 1.0
            out = np.where(lo, x * rl, out); out = np.where(hi, x * rh, out); out = np.maximum(out, 0.0)
        return np.where(np.isfinite(x), out, np.nan)


def _pair(obs, mod):
    o, m = np.asarray(obs, dtype=float).ravel(), np.asarray(mod, dtype=float).ravel()
    if o.shape != m.shape:
        raise ValueError("obs and model must have the same length (paired samples)")
    ok = np.isfinite(o) & np.isfinite(m)
    return o[ok], m[ok]


def fit_grouped(factory, obs, mod, groups) -> dict:
    """One corrector per group label (e.g. calendar month): ``factory()`` returns an unfitted corrector."""
    obs, mod, groups = np.asarray(obs, float), np.asarray(mod, float), np.asarray(groups)
    return {g: factory().fit(obs[groups == g], mod[groups == g]) for g in np.unique(groups)}


def apply_grouped(correctors: dict, mod, groups) -> np.ndarray:
    mod, groups = np.asarray(mod, float), np.asarray(groups)
    out = np.full(mod.shape, np.nan)
    for g, c in correctors.items():
        k = groups == g
        if k.any():
            out[k] = c.apply(mod[k])
    return out


def rolling_correct(mod: pd.DataFrame, obs: pd.DataFrame, factory, min_train: int = 15) -> pd.DataFrame:
    """Leak-free correction of every row (e.g. season_year) of ``mod`` (rows x targets) using only EARLIER rows.

    Rows with fewer than ``min_train`` earlier paired values stay NaN.  ``mod`` and ``obs`` share index and columns.
    """
    if not mod.index.equals(obs.index) or not mod.columns.equals(obs.columns):
        raise ValueError("mod and obs must have identical index and columns")
    out = pd.DataFrame(np.nan, index=mod.index, columns=mod.columns)
    idx = list(mod.index)
    for i, t in enumerate(idx):
        prev = idx[:i]
        for c in mod.columns:
            o, m = obs.loc[prev, c].values, mod.loc[prev, c].values
            ok = np.isfinite(o) & np.isfinite(m)
            if ok.sum() >= min_train and np.isfinite(mod.loc[t, c]):
                out.loc[t, c] = float(factory().fit(o[ok], m[ok]).apply(np.array([mod.loc[t, c]]))[0])
    return out
