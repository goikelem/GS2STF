"""Unit tests for the numerical kernels (run: python -m pytest tests -q  or  python tests/test_kernels.py)."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.drought_indices import index_kernels as k  # noqa: E402

RNG = np.random.default_rng(0)


def _monthly(nyears=50, cells=6, start_year=1981):
    t = nyears * 12
    months = np.tile(np.arange(1, 13), nyears)
    years = start_year + np.repeat(np.arange(nyears), 12)
    return t, months, years


def test_classify_boundaries():
    z = np.array([2.0, 1.99, 1.5, 1.49, 1.0, 0.99, 0.0, -0.99, -1.0, -1.49, -1.5, -1.99, -2.0, -3, np.nan])
    exp = [7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0]
    assert k.classify(z).tolist() == exp


def test_classify_no_gaps():
    z = np.linspace(-4, 4, 80001)
    c = k.classify(z)
    assert (c >= 1).all() and (c <= 7).all()


def test_accumulate():
    x = np.arange(1, 11, dtype=float)[:, None]
    a = k.accumulate(x, 3)
    assert np.isnan(a[:2]).all() and a[2, 0] == 6 and a[-1, 0] == 27
    x[4, 0] = np.nan
    a = k.accumulate(x, 3)
    assert np.isnan(a[4:7]).all() and a[7, 0] == 21


def test_spi_standard_normal_in_reference():
    t, months, years = _monthly()
    x = RNG.gamma(2.0, 40.0, size=(t, 6))
    x[RNG.random(x.shape) < 0.05] = 0.0
    ref = (years >= 1991) & (years <= 2020)
    z = k.spi_from_accumulated(x, months, ref)
    zr = z[ref]
    assert abs(np.nanmean(zr)) < 0.15 and abs(np.nanstd(zr) - 1) < 0.15


def test_spi_matches_scipy_gamma_mle():
    """Thom approximation should agree closely with a full scipy gamma MLE (no zeros)."""
    t, months, years = _monthly(60)
    x = RNG.gamma(3.0, 30.0, size=(t, 3))
    ref = np.ones(t, dtype=bool)
    z = k.spi_from_accumulated(x, months, ref)
    for c in range(3):
        sel = months == 1
        a, loc, sc = stats.gamma.fit(x[sel, c], floc=0)
        z_ref = stats.norm.ppf(stats.gamma.cdf(x[sel, c], a, loc, sc))
        assert np.nanmax(np.abs(z_ref - z[sel, c])) < 0.1


def test_spi_zero_handling_and_monotonic():
    t, months, years = _monthly()
    x = RNG.gamma(1.5, 20.0, size=(t, 1))
    x[RNG.random(x.shape) < 0.2] = 0.0
    z = k.spi_from_accumulated(x, months, np.ones(t, dtype=bool))
    m = months == 5
    order = np.argsort(x[m, 0])
    zz = z[m, 0][order]
    assert (np.diff(zz) >= -1e-12).all()


def test_spei_loglogistic_recovers_distribution():
    n = 4000
    alpha, beta, gam = 80.0, 4.0, -150.0
    u = RNG.random((n, 2))
    x = gam + alpha * (u / (1 - u)) ** (1 / beta)        # inverse CDF of log-logistic
    a, b, g = k.loglogistic_params_pwm(x)
    assert np.allclose(b, beta, rtol=0.12) and np.allclose(a, alpha, rtol=0.12)
    months = np.ones(n, dtype=int)
    z = k.spei_from_accumulated(x, months, np.ones(n, dtype=bool))
    assert abs(np.nanmean(z)) < 0.05 and abs(np.nanstd(z) - 1) < 0.08


def test_spei_nan_propagation_and_clip():
    t, months, years = _monthly()
    x = RNG.gamma(3.0, 60.0, size=(t, 2)) - 150.0     # positively skewed water balance
    x[:, 1] = np.nan
    z = k.spei_from_accumulated(x, months, np.ones(t, dtype=bool))
    assert np.isnan(z[:, 1]).all() and np.isfinite(z[:, 0]).all()
    assert np.nanmax(np.abs(z)) <= 3.09 + 1e-9


def test_spei_failure_diagnostic():
    t, months, years = _monthly()
    x = -RNG.gamma(3.0, 60.0, size=(t, 2))            # negative skew -> log-logistic cannot fit
    f = k.spei_fit_failure_fraction(x, months, np.ones(t, dtype=bool))
    assert f.shape == (2,) and (f > 0.5).all()


def test_ra_and_hargreaves_reasonable():
    lat = np.array([0.0, 10.0, -10.0, 50.0])
    ra = k.monthly_mean_ra(lat)
    assert ra.shape == (12, 4)
    assert 14 < ra[:, 0].mean() < 17                      # equator ~15.5 mm/day
    assert ra[5, 3] > ra[11, 3]                           # N-hemisphere: June > December
    pet = k.hargreaves_pet(np.full((12, 1), 15.0), np.full((12, 1), 28.0), ra[:, [1]], np.full(12, 30))
    assert 100 < pet.mean() < 220                         # ~4-6 mm/day * 30
    assert (k.hargreaves_pet(np.array([[20.0]]), np.array([[10.0]]), np.array([[10.0]]), np.array([30])) >= 0).all()


def test_cell_area_total():
    lat = np.arange(-89.5, 90, 1.0)
    lon = np.arange(0.5, 360, 1.0)
    a = k.cell_area_km2(lat, lon).sum()
    assert abs(a - 4 * np.pi * 6371.0088 ** 2) / a < 1e-4


def test_spell_statistics():
    z = np.array([0, -1.2, -1.5, 0.3, -2.0, -1.0, -1.1, 0, 0.5, -1.0], dtype=float)[:, None]
    s = k.spell_statistics(z, -1.0, below=True)
    assert s["n_events"][0] == 3
    assert s["max_duration"][0] == 3 and abs(s["mean_duration"][0] - 2.0) < 1e-12
    assert abs(s["frequency_pct"][0] - 60.0) < 1e-9
    assert abs(s["max_severity"][0] - 4.1) < 1e-9


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
