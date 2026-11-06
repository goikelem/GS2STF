"""Tests for seasonal forecasting + verification (python tests/test_seasonal.py  or  pytest tests)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.forecasting import seasonal as S  # noqa: E402
from src.verification import probabilistic_scores as V  # noqa: E402


def make_indices(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1981-01-01", "2026-12-01", freq="MS")
    def ar(phi, s):
        x = np.zeros(len(idx)); e = rng.normal(0, s, len(idx))
        for i in range(1, len(idx)): x[i] = phi * x[i - 1] + e[i]
        return x
    df = pd.DataFrame({"R34": ar(0.93, 0.35), "WIOD": ar(0.85, 0.25), "EIOD": ar(0.85, 0.25)}, index=idx)
    df["DMI"] = df["WIOD"] - df["EIOD"]
    return df


SPEC = S.ForecastSpec("JJAS", init_month=5, predictor_months=3)
YEARS = list(range(1981, 2026))


def planted(n_targets=3, signal=1.0, noise=0.6, seed=1):
    ind = make_indices(); X = S.build_predictors(ind, SPEC, YEARS + [2026])
    rng = np.random.default_rng(seed)
    base = (X["R34"] / X["R34"].std()).loc[YEARS].values
    Y = pd.DataFrame({f"S{j}": 400 + 60 * (signal * base + noise * rng.normal(size=len(YEARS))) for j in range(n_targets)}, index=YEARS)
    return ind, X, Y


def test_spec_dates_and_lead():
    d = SPEC.describe(2020)
    assert d["init_date"] == "2020-05-31" and d["lead_months"] == 1 and d["target_start"] == "2020-06-01" and d["target_end"] == "2020-09-30"
    f = S.ForecastSpec("FMAM", 11); d = f.describe(2021)
    assert d["init_date"] == "2020-11-30" and d["lead_months"] == 3 and d["target_end"] == "2021-05-31"
    o = S.ForecastSpec("ONDJ", 7); assert o.describe(2026)["target_end"] == "2027-01-31" and o.describe(2026)["lead_months"] == 3
    o2 = S.ForecastSpec("ONDJ", 7, year_label="end"); assert o2.describe(2027)["target_start"] == "2026-10-01"
    assert S.ForecastSpec("JJAS", 6).lead_months == 12            # init month == first month of season -> previous year's


def test_predictors_use_only_data_up_to_init():
    ind = make_indices()
    X = S.build_predictors(ind, SPEC, YEARS)
    for y in (1990, 2010, 2025):
        trunc = ind.loc[:SPEC.init_date(y)]                      # nothing after the initialization date exists
        assert np.allclose(S.build_predictors(trunc, SPEC, [y]).loc[y], X.loc[y])
    changed = ind.copy(); changed.loc["2010-06-01":] += 99.0     # corrupt everything after init for 2010
    assert np.allclose(S.build_predictors(changed, SPEC, [2010]).loc[2010], X.loc[2010])
    short = ind.loc[:"2026-04-01"]                               # window needs March-May 2026
    assert S.build_predictors(short, SPEC, [2026]).loc[2026].isna().all()


def test_signal_is_selected_and_skillful_random_is_not():
    ind, X, Y = planted()
    h = S.hindcast(Y, X, SPEC, transform="none")
    s = V.summarize(h)
    assert (h.selected.values == "R34").mean() > 0.6, h.selected.tail(3)
    assert s["pooled"]["rpss"] > 0.1 and s["per_target"]["correlation"].mean() > 0.5
    assert np.allclose(h.probs.sum(-1), 1, atol=1e-9) and (h.probs >= -1e-12).all()
    rng = np.random.default_rng(5)
    Yr = pd.DataFrame(rng.normal(400, 60, size=Y.shape), index=Y.index, columns=Y.columns)
    hr = S.hindcast(Yr, X, SPEC)
    sr = V.summarize(hr)
    assert sr["pooled"]["rpss"] < 0.08, sr["pooled"]["rpss"]      # no skill invented from noise
    assert (hr.selected.values == "climatology").mean() > 0.5


def test_hindcast_never_uses_future_or_test_year():
    ind, X, Y = planted(n_targets=2)
    h1 = S.hindcast(Y, X, SPEC)
    t = 2012
    Y2 = Y.copy(); Y2.loc[t] = 1e6; Y2.loc[t + 1:] = -1e6        # corrupt test year and everything after it
    h2 = S.hindcast(Y2, X, SPEC)
    for fld in ("forecast", "t1", "t2", "p_below", "p_above", "clim"):
        assert np.allclose(getattr(h1, fld).loc[t], getattr(h2, fld).loc[t]), fld
    assert h1.forecast.index.min() == 1981 + 15                  # needs min_train earlier years
    assert S._train_years(2000, [1998, 1999, 2000, 2001, 2002, 2010], "blocked", 1) == [1998, 2002, 2010]
    hb = S.hindcast(Y, X, SPEC, scheme="blocked")
    assert hb.forecast.index.min() == 1981 and V.summarize(hb)["pooled"]["rpss"] > 0.1


def test_forecast_output_format_and_missing_predictors():
    ind, X, Y = planted(n_targets=2)
    meta = pd.DataFrame({"station": ["S0", "S1"], "lat": [10.4, 11.2], "lon": [35.0, 36.1]})
    f = S.forecast(Y, X, SPEC, 2026, meta=meta, variable="precipitation")
    assert list(f.columns[:7]) == ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"]
    assert (f["Forecast_Period"] == "JJAS_2026").all() and f["init_date"].iloc[0] == "2026-05-31"
    assert np.allclose(f[["P_below", "P_normal", "P_above"]].sum(1), 1)
    assert np.allclose(f["Anomaly"], f["Forecast"] - f["Climatology"])
    Xbad = X.copy(); Xbad.loc[2026] = np.nan
    try:
        S.forecast(Y, Xbad, SPEC, 2026); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    Yfut = Y.copy(); Yfut.loc[2026] = 1e9                        # the target year's own value must be ignored
    assert np.allclose(S.forecast(Yfut, X, SPEC, 2026)["Forecast"], S.forecast(Y, X, SPEC, 2026)["Forecast"])


def test_gaps_transform_and_grid():
    ind, X, Y = planted(n_targets=3)
    Yg = Y.copy(); Yg.loc[[1990, 1995], "S1"] = np.nan; Yg["S2"] = np.nan   # gaps and an all-NaN station
    h = S.hindcast(Yg.clip(lower=1), X, SPEC, transform="sqrt")
    assert h.forecast["S1"].notna().any() and h.forecast["S2"].isna().all()
    assert (h.forecast["S0"].dropna() >= 0).all()
    field = np.stack([Y.values[:, 0], Y.values[:, 1], Y.values[:, 0] * 0.9, Y.values[:, 2]], 1).reshape(len(YEARS), 2, 2)
    field[:, 1, 1] = np.nan                                                  # masked cell
    g = S.forecast_gridded(field, YEARS, [10, 11], [35, 36], X, SPEC, 2026)
    assert g["forecast"].shape == (2, 2) and g["probabilities"].shape == (3, 2, 2)
    assert np.isnan(g["forecast"][1, 1]) and np.allclose(g["probabilities"][:, 0, 0].sum(), 1)


def test_score_sanity():
    cat = np.array([0, 1, 2, 1, 0, 2, 1, 1, 2, 0])
    perfect = np.eye(3)[cat]; clim = np.full((10, 3), 1 / 3)
    assert abs(1 - V.rps(perfect, cat).mean() / V.rps(clim, cat).mean() - 1) < 1e-12
    assert abs(V.probabilistic_scores(clim, cat)["rpss"]) < 1e-12
    assert V.heidke(cat, cat) == 1.0 and abs(V.heidke(np.array([0, 1, 2, 0, 1, 2]), np.array([0, 0, 1, 1, 2, 2]))) < 0.51
    assert abs(V.crps_gaussian(0.0, 0.0, 1.0) - (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))) < 1e-12
    assert V.roc_auc(np.array([.9, .8, .2, .1]), np.array([1, 1, 0, 0])) == 1.0
    assert V.roc_auc(np.array([.1, .2, .8, .9]), np.array([1, 1, 0, 0])) == 0.0


def test_bootstrap_ci_and_reliability():
    ind, X, Y = planted()
    h = S.hindcast(Y, X, SPEC)
    ci = V.bootstrap(h, n_boot=200, block=2, level=0.90)
    assert ci.loc["rpss", "lo_90"] < ci.loc["rpss", "mean"] < ci.loc["rpss", "hi_90"] and ci.loc["rpss", "lo_90"] > 0
    rel = V.summarize(h)["reliability"]
    assert {"mean_forecast_prob", "observed_frequency", "n", "category"} <= set(rel.columns)
    d = S.predictor_diagnostics(X[["R34", "WIOD", "EIOD", "DMI"]])
    assert d["vif"]["DMI"] > 10                                               # DMI = WIOD - EIOD is collinear


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
