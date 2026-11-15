"""Generalized linear model for rainfall occurrence: binomial GLM (logit link) of the seasonal wet-day fraction.

The response is the fraction of days in the season with rain >= a threshold (e.g. 1 mm), the number of trials is the
season length in days.  Fitted by iteratively reweighted least squares with a ridge penalty on the slopes, so a few
years of data and collinear ocean predictors stay stable (a wet-day fraction that is constant in a few years would
otherwise make the maximum-likelihood fit diverge).

Predictor subset (0..``max_k``, 0 = climatology) is chosen per station by exact leave-one-out refits on the TRAINING years,
and a subset must cut climatology's leave-one-out error by ``min_gain`` to be kept - the same rule as every other method.
The forecast fraction stays inside (0, 1) by construction; the error scale used for tercile probabilities is the
leave-one-out RMSE of the chosen model.

Registered as method ``glm`` in forecasting.seasonal; it needs ``trials=<season length in days>`` in the method options
(the station workflow sets it) and a response that is already a fraction (transform ``none``).  It refits the model
n x subsets times per station, so use it for stations, not for large grids.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import expit, logit

CLIP = 1e-6


def fit_binomial_glm(Z: np.ndarray, y: np.ndarray, trials: float, alpha: float = 1.0, max_iter: int = 30, tol: float = 1e-9):
    """Ridge-penalised binomial GLM (logit link) by IRLS.  Z: (n, k) standardised predictors, y: fractions in [0, 1].

    Returns (intercept, slopes).  The penalty is alpha x the mean IRLS weight on the slopes (the intercept is free).
    """
    n, k = Z.shape
    ybar = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    b = np.zeros(k + 1); b[0] = logit(ybar)
    A = np.hstack([np.ones((n, 1)), Z])
    for _ in range(max_iter):
        eta = A @ b
        p = np.clip(expit(eta), CLIP, 1 - CLIP)
        w = trials * p * (1 - p)
        z = eta + (y - p) / (p * (1 - p))
        P = np.diag([0.0] + [alpha * w.mean()] * k)
        new = np.linalg.solve(A.T @ (A * w[:, None]) + P, A.T @ (w * z))
        done = np.max(np.abs(new - b)) < tol
        b = new
        if done or k == 0:
            break
    return b[0], b[1:]


def _std(X):
    sd = X.std(0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
    sd = np.where(sd == 0, 1.0, sd)
    return X.mean(0), sd


def _predict_fraction(Xtr, ytr, Xte, trials, alpha):
    if Xtr.shape[1] == 0:
        return np.full(len(Xte), float(ytr.mean()))
    mu, sd = _std(Xtr)
    b0, b = fit_binomial_glm((Xtr - mu) / sd, ytr, trials, alpha)
    return expit(b0 + ((Xte - mu) / sd) @ b)


def _loo_rmse(X, y, trials, alpha):
    n = len(y); err = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        err[i] = y[i] - _predict_fraction(X[tr], y[tr], X[i:i + 1], trials, alpha)[0]
    return float(np.sqrt(np.mean(err ** 2)))


def _subsets(p, max_k):
    out = [()]
    for k in range(1, min(max_k, p) + 1):
        out += list(itertools.combinations(range(p), k))
    return out


def method_glm(X, Y, xt, names, o):
    """Method interface of forecasting.seasonal: X (n x p), Y (n x m) wet-day fractions, xt (1 x p)."""
    if "trials" not in o:
        raise ValueError("method 'glm' needs the number of trials per row (season length in days): pass trials=<days>; it is "
                         "set automatically for the wet-day-fraction workflow (--wet-day-threshold)")
    if np.nanmin(Y) < 0 or np.nanmax(Y) > 1:
        raise ValueError("method 'glm' models fractions in [0, 1] (use transform 'none')")
    trials, alpha = float(o["trials"]), o["alpha"]
    cand = list(range(X.shape[1])) if o.get("candidates") is None else [names.index(c) for c in o["candidates"]]
    subsets = [tuple(cand[i] for i in s) for s in _subsets(len(cand), o["max_k"])]
    n, m = Y.shape
    mean = np.empty(m); sigma = np.empty(m); label = np.empty(m, dtype=object)
    for j in range(m):
        y = Y[:, j]
        R = np.array([_loo_rmse(X[:, list(s)], y, trials, alpha) for s in subsets])
        best = R.min()
        ok = np.where(R <= best * (1 + o["tol"]))[0]
        k = min(ok, key=lambda q: (len(subsets[q]), R[q]))
        if not R[k] < R[0] * (1 - o["min_gain"]):
            k = 0
        idx = list(subsets[k])
        mean[j] = _predict_fraction(X[:, idx], y, xt[:, idx], trials, alpha)[0]
        sigma[j] = R[k]
        label[j] = "climatology" if not idx else "glm:" + "+".join(names[i] for i in idx)
    return mean, sigma, label
