"""Unit tests for teleconnection kernels (python tests/test_teleconnection.py  or  pytest tests)."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import teleconnection as tc  # noqa: E402

RNG = np.random.default_rng(1)


def test_pearson_map_matches_scipy():
    X = RNG.normal(size=(60, 5)); y = X[:, 0] * 0.5 + RNG.normal(size=60)
    X[3, 2] = np.nan
    r, n = tc.pearson_map(X, y)
    for j in range(5):
        ok = np.isfinite(X[:, j])
        assert abs(r[j] - stats.pearsonr(X[ok, j], y[ok])[0]) < 1e-10 and n[j] == ok.sum()


def test_effective_n():
    assert abs(tc.effective_n(100, 0.5, 0.5) - 60.0) < 1e-9
    assert tc.effective_n(100, -0.5, 0.5) == 100        # negative product not inflated


def test_fdr_bh_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, np.nan])
    rej = tc.fdr_bh(p, 0.05)
    assert rej[:2].all() and not rej[2:].any()           # BH at alpha=0.05: only the two smallest


def test_lagged_correlation_recovers_lag():
    T = 400
    X = RNG.normal(size=(T, 4))
    y = np.zeros(T)
    y[3:] = X[:-3, 0] + 0.5 * RNG.normal(size=T - 3)      # predictor leads by 3 months
    months = np.tile(np.arange(1, 13), T // 12 + 1)[:T]
    res = tc.lagged_correlation(X, y, np.ones(T, bool), range(0, 7))
    r, lag, p = tc.best_lag(res)
    assert lag[0] == 3 and r[0] > 0.8 and p[0] < 1e-6
    assert abs(r[1]) < 0.3


def test_mca_recovers_planted_mode_and_significance():
    n, px, py = 40, 30, 25
    t = RNG.normal(size=n)
    patx, paty = RNG.normal(size=px), RNG.normal(size=py)
    X = np.outer(t, patx) + RNG.normal(size=(n, px))
    Y = np.outer(t, paty) + RNG.normal(size=(n, py))
    m = tc.mca_permutation(X, Y, nperm=200, nmodes=3)
    assert m["r_ab"][0] > 0.8 and m["scf"][0] > 0.5 and m["p_value"][0] < 0.02
    assert abs(np.corrcoef(m["hom_x"][0], patx)[0, 1]) > 0.7
    assert abs(m["scf"].sum() - m["scf"][:3].sum()) < 1e-12
    Yn = RNG.normal(size=(n, py))
    mn = tc.mca_permutation(X, Yn, nperm=200, nmodes=1)
    assert mn["p_value"][0] > 0.05


def test_mca_matches_direct_svd():
    n, px, py = 25, 12, 9
    X = RNG.normal(size=(n, px)); Y = 0.5 * X[:, :1] @ RNG.normal(size=(1, py)) + RNG.normal(size=(n, py))
    wx, wy = RNG.uniform(0.5, 1.5, px), RNG.uniform(0.5, 1.5, py)
    m = tc.mca(X, Y, wx, wy, nmodes=3)
    Xa, Ya = (X - X.mean(0)) * wx, (Y - Y.mean(0)) * wy
    U, s, Vt = np.linalg.svd(Xa.T @ Ya / (n - 1), full_matrices=False)
    assert np.allclose(m["singular"], s[:3])
    a = Xa @ U[:, :3]
    assert np.allclose(np.abs(m["a"]), np.abs(a), atol=1e-9)
    assert np.allclose(np.abs(m["b"]), np.abs(Ya @ Vt[:3].T), atol=1e-9)


def test_eof_variance_fractions():
    t = RNG.normal(size=50)
    X = np.outer(t, RNG.normal(size=20)) * 3 + RNG.normal(size=(50, 20))
    e = tc.eof(X, nmodes=3)
    assert e["var_frac"][0] > 0.5 and (np.diff(e["var_frac"]) <= 1e-12).all()


def test_composite_detects_difference():
    A = RNG.normal(size=(45, 6))
    member = np.zeros(45, bool); member[:8] = True
    A[member, 0] += 2.5
    d, p, nm, no = tc.composite(A, member)
    assert nm == 8 and no == 37 and p[0] < 0.01 and d[0] > 1.5 and p[1:].min() > 0.001


def test_seasonal_aggregate_djf_labels_and_sum():
    years = np.repeat(np.arange(2000, 2004), 12); months = np.tile(np.arange(1, 13), 4)
    x = np.arange(48, dtype=float)[:, None]
    ys, a = tc.seasonal_aggregate(x, years, months, [12, 1, 2], "mean")
    # DJF 2001 = Dec 2000 (idx 11) + Jan 2001 (12) + Feb 2001 (13)
    i = list(ys).index(2001)
    assert abs(a[i, 0] - np.mean([11, 12, 13])) < 1e-12
    assert 2000 not in ys or np.isnan(a[list(ys).index(2000), 0])   # incomplete DJF 2000 dropped
    ys2, s = tc.seasonal_aggregate(x, years, months, [6, 7, 8, 9], "sum")
    assert abs(s[0, 0] - (5 + 6 + 7 + 8)) < 1e-12


def test_correlation_methods_ci():
    x = RNG.normal(size=80); y = 0.6 * x + 0.8 * RNG.normal(size=80)
    c = tc.correlation_methods(x, y, nboot=500)
    assert c["ci_low"] < c["pearson"] < c["ci_high"] and c["pearson_p"] < 1e-5
    assert abs(c["spearman"] - stats.spearmanr(x, y)[0]) < 1e-12 and c["neff"] <= c["n"]


def test_regression_and_map():
    x = RNG.normal(size=60); Y = np.column_stack([2 * x + RNG.normal(size=60) * 0.1, RNG.normal(size=60)])
    s, r, p = tc.regression_map(Y, x)
    assert abs(s[0] - 2) < 0.1 and p[0] < 1e-10
    reg = tc.linear_regression(x, 3 * x + 1)
    assert abs(reg["slope"] - 3) < 1e-9 and reg["r2"] > 0.999999


def test_analogue_skill_planted_vs_random():
    n, N = 60, 40
    w = np.ones(N)
    mode = RNG.normal(size=N)
    amp = RNG.normal(size=n)
    F = np.outer(amp, mode) + 0.5 * RNG.normal(size=(n, N))
    rain = 2 * amp + 0.3 * RNG.normal(size=n)
    yrs = np.arange(1981, 1981 + n)
    pred = tc.analogue_loo_hindcast(F, yrs, rain, w, k=5)
    sk = tc.forecast_skill(rain, pred)
    assert sk["pearson"] > 0.6 and sk["rmsess_vs_climatology"] > 0
    pred_r = tc.analogue_loo_hindcast(RNG.normal(size=(n, N)), yrs, rain, w, k=5)
    assert tc.forecast_skill(rain, pred_r)["pearson"] < 0.35
    top, c = tc.rank_analogues(F, yrs, F[10], w, exclude_year=yrs[10], k=3)
    assert yrs[10] not in top and (np.diff(c) <= 1e-12).all()


def test_loo_linear():
    x = RNG.normal(size=50); y = 2 * x + RNG.normal(size=50) * 0.3
    sk = tc.forecast_skill(y, tc.loo_linear_hindcast(x, y))
    assert sk["pearson"] > 0.9 and sk["rmsess_vs_climatology"] > 0.5


def _seeded(fn):
    import functools

    @functools.wraps(fn)
    def wrapper():
        global RNG
        RNG = np.random.default_rng(123)       # every test gets the same fresh stream: order-independent
        return fn()
    return wrapper


for _n, _f in list(globals().items()):
    if _n.startswith("test_") and callable(_f):
        globals()[_n] = _seeded(_f)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
