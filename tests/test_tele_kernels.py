"""Unit tests for tele_kernels (run: python tests/test_tele_kernels.py or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.teleconnections import tele_kernels as tk  # noqa: E402

RNG = np.random.default_rng(7)


def test_corr_columns_matches_numpy():
    X = RNG.normal(size=(60, 5)); y = X[:, 2] * 0.7 + RNG.normal(size=60)
    X[3, 1] = np.nan
    r, n = tk.corr_columns(X, y)
    ok = np.isfinite(X[:, 1])
    assert abs(r[1] - np.corrcoef(X[ok, 1], y[ok])[0, 1]) < 1e-12 and n[1] == 59 and n[0] == 60
    assert abs(r[2] - np.corrcoef(X[:, 2], y)[0, 1]) < 1e-12


def test_effective_n_and_pvalue():
    assert tk.effective_n(100.0, 0.0, 0.0) == 100.0
    assert tk.effective_n(100.0, 0.6, 0.6) < 60
    assert tk.corr_pvalue(0.5, 50) < tk.corr_pvalue(0.5, 10)


def test_fdr_bh():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.6, np.nan])
    sig = tk.fdr_bh(p, 0.05)
    assert sig.tolist() == [True, True, False, False, False, False, False]


def test_lagged_correlation_recovers_lag():
    T = 400
    x = np.convolve(RNG.normal(size=T + 4), np.ones(5) / 5, "same")[:T]
    y = np.full(T, np.nan); y[3:] = x[:-3] * 0.9 + RNG.normal(0, 0.1, T - 3)    # y(t) driven by x(t-3)
    X = np.column_stack([x, RNG.normal(size=T)])
    res = tk.lagged_correlation(X, y, [0, 1, 2, 3, 4, 5, 6])
    assert res["lag_of_max"][0] == 3 and res["r_max"][0] > 0.9
    assert res["p"][3][0] < 1e-6 and res["n"][0][0] > 300


def test_lagged_correlation_vs_index():
    T = 300
    idx = np.convolve(RNG.normal(size=T + 4), np.ones(5) / 5, "same")[:T]
    F = np.full((T, 2), np.nan); F[2:, 0] = idx[:-2] + RNG.normal(0, 0.1, T - 2); F[:, 1] = RNG.normal(size=T)
    res = tk.lagged_correlation_vs_index(F, idx, range(0, 7))
    assert res["lag_of_max"][0] == 2 and res["r_max"][0] > 0.9


def test_pair_statistics():
    x = RNG.normal(size=60); y = 0.8 * x + RNG.normal(0, 0.5, 60)
    s = tk.pair_statistics(x, y, n_boot=300)
    assert s["pearson_r_ci_lo"] < s["pearson_r"] < s["pearson_r_ci_hi"]
    assert s["perm_p"] < 0.01 and abs(s["bias"]) < 1e-9 and s["spearman_rho"] > 0.6


def test_mca_recovers_coupled_mode():
    n, nx, ny = 40, 300, 120
    t = RNG.normal(size=n)
    px, py = RNG.normal(size=nx), RNG.normal(size=ny)
    X = np.outer(t, px) + RNG.normal(0, 1.0, (n, nx))
    Y = np.outer(t, py) + RNG.normal(0, 1.0, (n, ny))
    m = tk.mca(X, Y, n_modes=2, n_perm=199)
    assert m["ec_corr"][0] > 0.85 and m["scf"][0] > 0.5 and m["p_scf"][0] < 0.05
    assert abs(np.corrcoef(m["U"][:, 0], t)[0, 1]) > 0.9
    assert abs(np.corrcoef(m["pattern_x"][0], px)[0, 1]) > 0.8
    # equals a direct SVD of the cross-covariance
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    s = np.linalg.svd(Xc.T @ Yc, compute_uv=False)
    assert np.allclose(m["singular_values"], s[:2], rtol=1e-8)


def test_mca_no_signal_not_significant():
    X, Y = RNG.normal(size=(40, 100)), RNG.normal(size=(40, 50))
    m = tk.mca(X, Y, n_modes=1, n_perm=199)
    assert m["p_scf"][0] > 0.05


def test_phase_and_composite():
    idx = np.array([1.0, -0.8, 0.1, 0.5, -0.5, np.nan])
    assert tk.phase_from_index(idx).tolist() == ["El Nino", "La Nina", "Neutral", "El Nino", "La Nina", "missing"]
    Y = RNG.normal(size=(60, 4)); el = np.arange(60) < 20; la = (np.arange(60) >= 20) & (np.arange(60) < 40)
    Y[el, 0] += 3
    d, p, sig = tk.composite_difference(Y, el, la)
    assert d[0] > 2 and sig[0] and p[0] < 1e-6


def test_pattern_similarity_and_analogues():
    lib = RNG.normal(size=(30, 200))
    target = lib[7] + RNG.normal(0, 0.1, 200)
    out, top = tk.rank_analogues(target, lib, np.arange(1990, 2020), top_k=3)
    assert out[0]["year"] == 1997 and out[0]["pattern_corr"] > 0.95
    out2, _ = tk.rank_analogues(target, lib, np.arange(1990, 2020), top_k=3, exclude_year=1997)
    assert out2[0]["year"] != 1997


def test_loo_skill_regression_beats_climatology():
    n = 45
    idx = RNG.normal(size=n); y = 0.8 * idx + RNG.normal(0, 0.5, n)
    s = tk.loo_skill(idx[:, None], y, method="regression")
    assert s["r"] > 0.7 and s["mse_skill_score"] > 0.3
    z = tk.loo_skill(RNG.normal(size=(n, 1)), y, method="regression")
    assert z["mse_skill_score"] < 0.1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
