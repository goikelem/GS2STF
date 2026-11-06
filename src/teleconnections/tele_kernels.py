"""NumPy kernels for teleconnection analysis (lagged correlation, FDR, MCA, composites, analogues). No xarray dependency.

Origin: tele_kernels.py."""
from __future__ import annotations

import numpy as np
from scipy import stats


def monthly_anomalies(x: np.ndarray, months: np.ndarray, ref_mask: np.ndarray, standardize: bool = False):
    """Subtract the reference-period monthly climatology (optionally divide by its std)."""
    out = np.full(x.shape, np.nan)
    for m in range(1, 13):
        t = months == m
        ref = x[t & ref_mask]
        if ref.shape[0] == 0:
            continue
        mu = np.nanmean(ref, axis=0)
        a = x[t] - mu
        if standardize:
            sd = np.nanstd(ref, axis=0, ddof=1)
            a = np.where(sd > 0, a / sd, np.nan)
        out[t] = a
    return out


def lag1_autocorr(a: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each column (NaN-aware), clipped to [-0.99, 0.99]."""
    x0, x1 = a[:-1], a[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    x0, x1 = np.where(ok, x0, 0.0), np.where(ok, x1, 0.0)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m0, m1 = x0.sum(0) / n, x1.sum(0) / n
        c = (np.where(ok, x0 - m0, 0) * np.where(ok, x1 - m1, 0)).sum(0)
        v = np.sqrt((np.where(ok, x0 - m0, 0) ** 2).sum(0) * (np.where(ok, x1 - m1, 0) ** 2).sum(0))
        r = c / v
    return np.clip(r, -0.99, 0.99)


def corr_columns(X: np.ndarray, y: np.ndarray, min_n: int = 10):
    """Pearson r between every column of X (n, N) and y (n,), pair-wise complete. Returns (r, n)."""
    ok = np.isfinite(X) & np.isfinite(y)[:, None]
    n = ok.sum(axis=0)
    Xo, yo = np.where(ok, X, 0.0), np.where(ok, y[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = Xo.sum(0) / n, yo.sum(0) / n
        dx, dy = np.where(ok, X - mx, 0.0), np.where(ok, y[:, None] - my, 0.0)
        r = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
    return np.where(n >= min_n, r, np.nan), n


def effective_n(n, r1x, r1y):
    """Effective sample size for two autocorrelated series (Bretherton et al. 1999)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ne = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    return np.clip(ne, 3, n)


def corr_pvalue(r, neff):
    """Two-sided p-value of r with ``neff`` effective degrees of freedom (t-test)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt((neff - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), np.maximum(neff - 2, 1))


def fdr_bh(p: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg (FDR) significance mask; NaN p-values are never significant."""
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    ok = np.isfinite(flat)
    m = ok.sum()
    sig = np.zeros(flat.shape, dtype=bool)
    if m == 0:
        return sig.reshape(p.shape)
    idx = np.where(ok)[0]
    order = idx[np.argsort(flat[idx])]
    thresh = alpha * np.arange(1, m + 1) / m
    passed = flat[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        sig[order[: kmax + 1]] = True
    return sig.reshape(p.shape)


def shift_rows(X: np.ndarray, lag: int) -> np.ndarray:
    """Row t of the result holds X[t - lag] (predictor leads the predictand by ``lag`` rows)."""
    if lag == 0:
        return X.copy()
    out = np.full(X.shape, np.nan)
    out[lag:] = X[:-lag]
    return out


def lagged_correlation(X: np.ndarray, y: np.ndarray, lags, row_mask: np.ndarray | None = None,
                       min_n: int = 10, fdr_alpha: float = 0.10) -> dict:
    """Correlate predictor fields X (T, N) at lag L with predictand y (T,) at the target rows.

    For every lag: r, n, n_eff, p (autocorrelation adjusted) and an FDR significance mask.
    Also returns r at maximum |r| across lags and the lag at which it occurs.
    ``row_mask`` selects the target months (e.g. the rainy season) of the predictand.
    """
    r1x = lag1_autocorr(X)
    r1y = lag1_autocorr(y[:, None])[0]
    sel = np.ones(len(y), bool) if row_mask is None else row_mask
    res = {"lags": list(lags), "r": [], "n": [], "n_eff": [], "p": [], "sig": []}
    for L in lags:
        XL = shift_rows(X, L)[sel]
        r, n = corr_columns(XL, y[sel], min_n)
        # n_eff from the full-series lag-1 autocorrelations scaled to the sub-sample size
        ne = effective_n(n.astype(float), r1x, r1y)
        p = corr_pvalue(r, ne)
        res["r"].append(r); res["n"].append(n); res["n_eff"].append(ne); res["p"].append(p)
        res["sig"].append(fdr_bh(p, fdr_alpha))
    R = np.array(res["r"])
    with np.errstate(invalid="ignore"):
        allnan = np.all(~np.isfinite(R), axis=0)
        R0 = np.where(np.isfinite(R), np.abs(R), -1)
        k = np.argmax(R0, axis=0)
    rmax = np.take_along_axis(R, k[None, :], axis=0)[0]
    res["r_max"] = np.where(allnan, np.nan, rmax)
    res["lag_of_max"] = np.where(allnan, np.nan, np.asarray(list(lags))[k]).astype(float)
    return res


def lagged_correlation_vs_index(F: np.ndarray, idx: np.ndarray, lags, row_mask=None, min_n: int = 10,
                                fdr_alpha: float = 0.10) -> dict:
    """Correlate every column of F (T, N) at time t with the 1-D index at time t - L.

    Same output structure as :func:`lagged_correlation` (here the index is the predictor)."""
    r1f = lag1_autocorr(F)
    r1i = lag1_autocorr(idx[:, None])[0]
    sel = np.ones(len(idx), bool) if row_mask is None else row_mask
    res = {"lags": list(lags), "r": [], "n": [], "n_eff": [], "p": [], "sig": []}
    for L in lags:
        ishift = shift_rows(idx[:, None], L)[:, 0]
        r, n = corr_columns(F[sel], ishift[sel], min_n)
        ne = effective_n(n.astype(float), r1f, r1i)
        p = corr_pvalue(r, ne)
        res["r"].append(r); res["n"].append(n); res["n_eff"].append(ne); res["p"].append(p)
        res["sig"].append(fdr_bh(p, fdr_alpha))
    R = np.array(res["r"])
    allnan = np.all(~np.isfinite(R), axis=0)
    k = np.argmax(np.where(np.isfinite(R), np.abs(R), -1), axis=0)
    res["r_max"] = np.where(allnan, np.nan, np.take_along_axis(R, k[None, :], axis=0)[0])
    res["lag_of_max"] = np.where(allnan, np.nan, np.asarray(list(lags))[k]).astype(float)
    return res


def pair_statistics(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42, ci: float = 0.95) -> dict:
    """Correlation (3 types), OLS regression, R2, RMSE, MAE, bias, bootstrap CI and permutation p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return {"n": n}
    rng = np.random.default_rng(seed)
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    lr = stats.linregress(x, y)
    pred = lr.intercept + lr.slope * x
    boots = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        boots[i] = np.corrcoef(x[j], y[j])[0, 1] if np.std(x[j]) > 0 and np.std(y[j]) > 0 else np.nan
    lo, hi = np.nanpercentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    perm = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(n_boot)])
    p_perm = (1 + np.sum(perm >= abs(pr[0]))) / (n_boot + 1)
    return {"n": n, "pearson_r": pr[0], "pearson_p": pr[1], "spearman_rho": sr[0], "spearman_p": sr[1],
            "kendall_tau": kt[0], "kendall_p": kt[1], "slope": lr.slope, "intercept": lr.intercept,
            "r2": lr.rvalue ** 2, "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "mae": float(np.mean(np.abs(y - pred))), "bias": float(np.mean(pred - y)),
            "pearson_r_ci_lo": lo, "pearson_r_ci_hi": hi, "perm_p": p_perm}


def _center(A):
    return A - np.nanmean(A, axis=0, keepdims=True)


def _svals_rank(Rx, Ry):
    return np.linalg.svd(Rx @ Ry.T, compute_uv=False)


def mca(X: np.ndarray, Y: np.ndarray, n_modes: int = 3, n_perm: int = 999, seed: int = 42,
        wx: np.ndarray | None = None, wy: np.ndarray | None = None) -> dict:
    """Maximum covariance analysis between X (years, nx) and Y (years, ny).

    Columns containing any NaN are excluded (reported in ``valid_x/valid_y``).  ``wx/wy`` are
    optional column weights (e.g. sqrt(cos lat)) applied before the SVD.  Significance of
    each mode's squared covariance fraction and singular value is estimated by permuting the
    year order of Y (the QR trick makes each permutation an n x n SVD).

    Returns singular values, squared covariance fractions (SCF), expansion coefficients
    (U, V), spatial patterns (left/right singular vectors on the full grid, NaN where
    excluded), homogeneous correlation maps, the correlation between paired ECs, and
    permutation p-values.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows (years)")
    vx = np.isfinite(X).all(axis=0) & (np.nanstd(X, axis=0) > 0)
    vy = np.isfinite(Y).all(axis=0) & (np.nanstd(Y, axis=0) > 0)
    Xc, Yc = _center(X[:, vx]), _center(Y[:, vy])
    if wx is not None:
        Xc = Xc * np.asarray(wx)[vx]
    if wy is not None:
        Yc = Yc * np.asarray(wy)[vy]
    # thin QR of the transposed data: X^T = Qx Rx ; Y^T = Qy Ry  ->  X^T Y = Qx (Rx Ry^T) Qy^T
    Qx, Rx = np.linalg.qr(Xc.T)
    Qy, Ry = np.linalg.qr(Yc.T)
    Us, S, Vst = np.linalg.svd(Rx @ Ry.T)
    k = int(min(n_modes, len(S)))
    A = Qx @ Us[:, :k]          # (nx_valid, k) left patterns
    B = Qy @ Vst[:k].T          # (ny_valid, k) right patterns
    U, V = Xc @ A, Yc @ B       # expansion coefficients (n, k)
    scf = S ** 2 / np.sum(S ** 2)
    ec_r = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)])
    # homogeneous maps: correlation of each EC with its own (unweighted) field
    hx, hy = _fast_corr(X[:, vx], U), _fast_corr(Y[:, vy], V)
    rng = np.random.default_rng(seed)
    perm_s = np.empty((n_perm, k))
    perm_scf = np.empty((n_perm, k))
    for b in range(n_perm):
        sp = _svals_rank(Rx, Ry[:, rng.permutation(n)])
        perm_s[b] = sp[:k]
        perm_scf[b] = (sp ** 2 / np.sum(sp ** 2))[:k]
    p_s = (1 + (perm_s >= S[:k]).sum(0)) / (n_perm + 1)
    p_scf = (1 + (perm_scf >= scf[:k]).sum(0)) / (n_perm + 1)

    def expand(M, valid, ncol):                      # M is (k, n_valid)
        out = np.full((k, ncol), np.nan)
        out[:, valid] = M
        return out
    return {"singular_values": S[:k], "scf": scf[:k], "scf_all": scf, "U": U, "V": V, "ec_corr": ec_r,
            "pattern_x": expand(A.T, vx, X.shape[1]), "pattern_y": expand(B.T, vy, Y.shape[1]),
            "hom_x": expand(hx, vx, X.shape[1]), "hom_y": expand(hy, vy, Y.shape[1]),
            "p_singular": p_s, "p_scf": p_scf, "valid_x": vx, "valid_y": vy}


def _fast_corr(F, E):
    """Correlation of each column of E (n,k) with every column of F (n,N): returns (k, N)."""
    Fs = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), np.nan)
    Es = (E - E.mean(0)) / E.std(0)
    return (Es.T @ Fs) / F.shape[0]


def phase_from_index(values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """'El Nino' / 'La Nina' / 'Neutral' (or 'missing') from an ENSO index per year."""
    v = np.asarray(values, float)
    out = np.where(v >= threshold, "El Nino", np.where(v <= -threshold, "La Nina", "Neutral")).astype(object)
    out[~np.isfinite(v)] = "missing"
    return out


def composite_difference(Y: np.ndarray, in_a: np.ndarray, in_b: np.ndarray, fdr_alpha: float = 0.10):
    """Mean(Y | A) - mean(Y | B) per cell with Welch t-test p and FDR mask.  Y: (years, N)."""
    A, B = Y[in_a], Y[in_b]
    diff = np.nanmean(A, axis=0) - np.nanmean(B, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t, p = stats.ttest_ind(A, B, axis=0, equal_var=False, nan_policy="omit")
    return diff, np.asarray(p, float), fdr_bh(np.asarray(p, float), fdr_alpha)


def pattern_similarity(target: np.ndarray, library: np.ndarray, weights: np.ndarray | None = None):
    """Weighted pattern correlation and RMSE between ``target`` (N,) and each row of ``library`` (n, N)."""
    w = np.ones(target.shape) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(target) & np.isfinite(library).all(axis=0) & (w > 0)
    t, L, w = target[ok], library[:, ok], w[ok]
    ws = w / w.sum()
    tm = (ws * t).sum()
    Lm = (L * ws).sum(axis=1, keepdims=True)
    cov = ((t - tm) * (L - Lm) * ws).sum(axis=1)
    sd = np.sqrt((ws * (t - tm) ** 2).sum() * (ws * (L - Lm) ** 2).sum(axis=1))
    r = cov / sd
    rmse = np.sqrt((ws * (L - t) ** 2).sum(axis=1))
    return r, rmse


def rank_analogues(target, library, years, weights=None, top_k=5, exclude_year=None):
    """Rank candidate years by pattern correlation (ties broken by RMSE)."""
    r, rmse = pattern_similarity(target, library, weights)
    years = np.asarray(years)
    cand = np.arange(len(years)) if exclude_year is None else np.where(years != exclude_year)[0]
    order = cand[np.lexsort((rmse[cand], -np.nan_to_num(r[cand], nan=-9)))]
    top = order[:top_k]
    return [{"rank": i + 1, "year": int(years[j]), "pattern_corr": float(r[j]), "rmse": float(rmse[j])}
            for i, j in enumerate(top)], top


def loo_skill(predictor_fields: np.ndarray, y_series: np.ndarray, y_fields: np.ndarray | None = None,
              top_k: int = 5, weights=None, method: str = "analogue") -> dict:
    """Leave-one-year-out hindcast skill of an analogue composite or a linear regression.

    predictor_fields: (years, N) predictors (for 'regression' pass (years, 1) or (years, p)).
    y_series: (years,) regional rainfall anomaly. Returns r, RMSE, MAE, bias, and MSE skill score
    against LOO climatology (positive = better than climatology).
    """
    n = len(y_series)
    pred = np.full(n, np.nan)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        if method == "analogue":
            _, top = rank_analogues(predictor_fields[i], predictor_fields[tr], tr, weights, top_k)
            pred[i] = np.nanmean(y_series[tr][top]) if len(top) else np.nan
        else:
            Xtr = np.column_stack([np.ones(len(tr)), predictor_fields[tr]])
            ok = np.isfinite(Xtr).all(1) & np.isfinite(y_series[tr])
            beta, *_ = np.linalg.lstsq(Xtr[ok], y_series[tr][ok], rcond=None)
            xi = np.concatenate([[1.0], np.atleast_1d(predictor_fields[i])])
            pred[i] = xi @ beta if np.isfinite(xi).all() else np.nan
    clim = np.array([np.nanmean(np.delete(y_series, i)) for i in range(n)])
    ok = np.isfinite(pred) & np.isfinite(y_series)
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    e, ec = pred[ok] - y_series[ok], clim[ok] - y_series[ok]
    return {"n": int(ok.sum()), "r": float(np.corrcoef(pred[ok], y_series[ok])[0, 1]),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "mse_skill_score": float(1 - np.mean(e ** 2) / np.mean(ec ** 2)), "hindcast": pred}
