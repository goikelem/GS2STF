"""Tests for bias correction and dynamical-model integration (python tests/test_dynamical.py  or  pytest tests)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_seasonal import SPEC, YEARS, make_indices, planted  # noqa: E402

from src.forecasting import dynamical as D  # noqa: E402
from src.forecasting import ensemble as E  # noqa: E402
from src.forecasting import seasonal as S  # noqa: E402
from src.preprocessing import bias_correction as B  # noqa: E402
from src.verification import probabilistic_scores as V  # noqa: E402


def tmp():
    return Path(tempfile.mkdtemp())


# ---------------------------------------------------------------- bias correction
def test_linear_scaling_variants():
    rng = np.random.default_rng(0)
    obs = rng.normal(20, 3, 400); mod = 0.7 * obs + 5 + rng.normal(0, 0.01, 400)
    assert abs(B.LinearScaling("additive").fit(obs, mod).apply(mod).mean() - obs.mean()) < 1e-9
    pr = rng.gamma(2, 50, 400); md = pr * 1.6
    assert abs(B.LinearScaling("multiplicative").fit(pr, md).apply(md).mean() - pr.mean()) < 1e-9
    v = B.LinearScaling("variance").fit(obs, mod).apply(mod)
    assert abs(v.mean() - obs.mean()) < 1e-9 and abs(v.std(ddof=1) - obs.std(ddof=1)) < 1e-9
    for bad in (lambda: B.LinearScaling("x"), lambda: B.LinearScaling().fit([1, 2], [1, 2]), lambda: B.LinearScaling().fit([1, 2, 3], [1, 2])):
        try:
            bad(); raise AssertionError
        except ValueError:
            pass


def test_quantile_mapping_distribution_tails_and_monotonicity():
    rng = np.random.default_rng(1)
    obs = rng.gamma(2, 40, 2000); mod = 2.5 * rng.gamma(2, 40, 2000) + 10      # wrong scale and offset, other sample
    qm = B.QuantileMapping(100, "multiplicative").fit(obs, mod)
    c = qm.apply(mod)
    assert np.abs(np.quantile(c, [.1, .5, .9]) - np.quantile(obs, [.1, .5, .9])).max() < 0.08 * np.quantile(obs, .9)
    xs = np.linspace(mod.min() - 100, mod.max() + 500, 300)
    out = qm.apply(xs)
    assert (np.diff(out) >= -1e-9).all() and (out >= 0).all()                  # monotone, never negative
    t = B.QuantileMapping(50, "additive").fit(obs, obs + 4.0)
    assert np.allclose(t.apply(np.array([obs.max() + 100 + 4.0])), obs.max() + 100, atol=1e-6)   # constant offset beyond the range
    assert np.isnan(t.apply(np.array([np.nan]))[0])


def test_grouped_and_rolling_correction_are_leak_free():
    rng = np.random.default_rng(2)
    month = np.tile(np.arange(12), 30); obs = 20 + 5 * np.sin(month / 12 * 6.28) + rng.normal(0, 1, 360)
    mod = obs + (month - 6) * 0.8                                              # month-dependent bias
    cor = B.fit_grouped(lambda: B.LinearScaling("additive"), obs, mod, month)
    fixed = B.apply_grouped(cor, mod, month)
    assert abs((fixed - obs).mean()) < 1e-9 and np.abs(fixed - obs).max() < 1e-9
    yrs = list(range(1990, 2020))
    O = pd.DataFrame(rng.normal(300, 50, (30, 2)), index=yrs, columns=["a", "b"]); M = 0.5 * O + 40 + rng.normal(0, 5, (30, 2))
    f = lambda: B.LinearScaling("additive")
    r1 = B.rolling_correct(M, O, f, min_train=10)
    assert r1.iloc[:10].isna().all().all() and r1.iloc[10:].notna().all().all()
    O2, M2 = O.copy(), M.copy(); O2.iloc[15:] = 1e6; M2.iloc[16:] = -1e6      # corrupt obs from row 15, model after row 15
    r2 = B.rolling_correct(M2, O2, f, min_train=10)
    assert np.allclose(r1.iloc[:16], r2.iloc[:16], equal_nan=True)


# ---------------------------------------------------------------- loaders
def test_loaders_check_init_month_and_format():
    d = tmp()
    pd.DataFrame({"season_year": [1990, 1991], "init_month": [5, 5], "R34": [0.1, 0.2], "DMI": [0, 1]}).to_csv(d / "i.csv", index=False)
    x = D.load_index_forecasts(d / "i.csv", SPEC)
    assert list(x.columns) == ["DYN_R34", "DYN_DMI"] and list(x.index) == [1990, 1991]
    pd.DataFrame({"season_year": [1990], "init_month": [3], "R34": [0.1]}).to_csv(d / "bad.csv", index=False)
    pd.DataFrame({"R34": [0.1]}).to_csv(d / "noyear.csv", index=False)
    pd.DataFrame({"season_year": [1990, 1990], "R34": [0.1, 0.3]}).to_csv(d / "dup.csv", index=False)
    for f in ("bad", "noyear", "dup"):
        try:
            D.load_index_forecasts(d / f"{f}.csv", SPEC); raise AssertionError(f)
        except ValueError:
            pass
    st = pd.DataFrame({"Station": ["A", "A", "B", "B", "A"], "season_year": [1990, 1991, 1990, 1991, 1990],
                       "Variable": ["pr", "pr", "pr", "pr", "tmax"], "init_month": 5, "value": [1., 2., 3., 4., 99.]})
    st.to_csv(d / "s.csv", index=False)
    m = D.load_station_forecasts(d / "s.csv", SPEC, variable="precipitation")
    assert m.shape == (2, 2) and m.loc[1991, "B"] == 4.0
    try:
        D.load_station_forecasts(d / "s.csv", SPEC, variable="tmin"); raise AssertionError
    except ValueError:
        pass


# ---------------------------------------------------------------- calibrated member
def _hidden_driver_data(seed=3, n_st=4, model_noise=0.5, bias=True):
    """Obs driven by a hidden factor that the ocean indices do NOT carry; the model forecast does (with bias and scale error)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=len(YEARS))
    Y = pd.DataFrame({f"S{j}": 400 + 60 * (h + 0.6 * rng.normal(size=len(YEARS))) for j in range(n_st)}, index=YEARS)
    scale, offset = (3.0, 150.0) if bias else (60.0, 400.0)                                   # model units differ from observations
    M = pd.DataFrame({f"S{j}": offset + scale * (h + model_noise * rng.normal(size=len(YEARS))) for j in range(n_st)}, index=YEARS)
    return Y, M


def test_dynamical_member_skill_noise_and_leak_control():
    ind, X, _ = planted()
    Y, M = _hidden_driver_data()
    h = D.hindcast(Y, M, SPEC)
    s = V.summarize(h)
    assert s["pooled"]["rpss"] > 0.1 and (h.selected.values == "dynamical").mean() > 0.6     # bias and scale are calibrated away
    assert np.allclose(h.probs.sum(-1), 1, atol=1e-9)
    rng = np.random.default_rng(9)
    Mn = pd.DataFrame(rng.normal(150, 3, M.shape), index=M.index, columns=M.columns)         # model with no skill
    hn = D.hindcast(Y, Mn, SPEC)
    assert V.summarize(hn)["pooled"]["rpss"] < 0.06 and (hn.selected.values == "climatology").mean() > 0.5
    t = 2008
    Y2, M2 = Y.copy(), M.copy(); Y2.loc[t:] = 1e6; M2.loc[t + 1:] = -1e6                    # test-year model forecast IS available
    h2 = D.hindcast(Y2, M2, SPEC)
    for fld in ("forecast", "t1", "t2", "p_below", "p_above", "clim"):
        assert np.allclose(getattr(h, fld).loc[t], getattr(h2, fld).loc[t]), fld


def test_dynamical_forecast_table_and_missing_stations():
    Y, M = _hidden_driver_data(n_st=3)
    M.loc[2026] = [150 + 3.0, 150 - 3.0, np.nan]                                              # no model forecast for station S2
    meta = pd.DataFrame({"station": list(Y.columns), "lat": [10, 11, 12], "lon": [35, 36, 37]})
    f = D.forecast(Y, M, SPEC, 2026, meta=meta, variable="precipitation")
    assert list(f.columns[:7]) == ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"]
    assert f.loc[0, "Forecast"] > f.loc[1, "Forecast"] and np.isnan(f.loc[2, "Forecast"]) and (f["Method"] == "dynamical").all()
    assert np.allclose(f.loc[:1, ["P_below", "P_normal", "P_above"]].sum(axis=1), 1)
    try:
        D.forecast(Y, M.drop(index=2026), SPEC, 2026); raise AssertionError
    except ValueError:
        pass


def test_ensemble_trusts_the_dynamical_member_when_it_has_the_skill():
    ind, X, _ = planted()
    Y, M = _hidden_driver_data(seed=5)
    M.loc[2026] = [150 + 3.0 * 1.2] * 4
    meta = pd.DataFrame({"station": list(Y.columns), "lat": 10.0, "lon": 35.0})
    dyn = (D.hindcast(Y, M, SPEC), D.forecast(Y, M, SPEC, 2026, meta=meta, variable="precipitation"))
    fc, hc, allh = E.run_multimodel(Y, X, SPEC, 2026, ("ridge", "pcr"), meta=meta, variable="precipitation", extra_members={"dynamical": dyn})
    assert {"climatology", "ridge", "pcr", "dynamical", "ensemble"} <= set(allh)
    cmp_ = V.compare(allh, n_boot=80).set_index("method")
    assert cmp_.loc["dynamical", "rpss"] > cmp_.loc["ridge", "rpss"] + 0.1 and cmp_.loc["ensemble", "rpss"] > cmp_.loc["ridge", "rpss"] + 0.05
    assert fc["Weight_dynamical"].min() > fc["Weight_ridge"].max()
    assert np.allclose(fc[["P_below", "P_normal", "P_above"]].sum(axis=1), 1)


def test_dynamical_index_predictors_are_used():
    ind, X, Y = planted(n_targets=3)                                                          # Y driven by the observed R34 window
    rng = np.random.default_rng(4)
    drv = (X["R34"] / X["R34"].std())
    dyn = pd.DataFrame({"DYN_R34": drv + 0.2 * rng.normal(size=len(X)), "DYN_junk": rng.normal(size=len(X))}, index=X.index)
    Xf = X[["WIOD", "EIOD"]].join(dyn)                                                        # ocean indices that do not carry the driver
    h = S.hindcast(Y, Xf, SPEC)
    assert np.mean([("DYN_R34" in x) for x in h.selected.values.ravel()]) > 0.8
    assert V.summarize(h)["pooled"]["rpss"] > 0.15


def test_cli_with_dynamical_inputs_and_init_month_guard():
    d = tmp(); ind = make_indices(); ind.reset_index(names="Date").to_csv(d / "idx.csv", index=False)
    Xo = S.build_predictors(ind, SPEC, YEARS + [2026])
    rng = np.random.default_rng(6); h = rng.normal(size=len(YEARS))
    days = pd.date_range("1981-01-01", "2025-12-31"); rows = []
    for k, stn in enumerate(["A1", "A2"]):
        yr_fac = pd.Series(np.exp(0.4 * (h + 0.5 * rng.normal(size=len(YEARS)))), index=YEARS)
        jjas = np.isin(days.month, [6, 7, 8, 9])
        rain = np.where(jjas, rng.gamma(2, 4, len(days)), rng.gamma(0.4, 1, len(days))) * np.where(jjas, yr_fac.reindex(days.year).values, 1.0)
        rows.append(pd.DataFrame({"Station": stn, "Date": days.strftime("%Y-%m-%d"), "Lat": 10 + k, "Lon": 35 + k, "Variable": "precipitation", "Value": rain.round(2)}))
    pd.concat(rows).to_csv(d / "obs.csv", index=False)
    mod = [{"Station": s, "season_year": y, "init_month": 5, "value": 100 + 20 * (h[i] if y <= 2025 else 1.0) + rng.normal(0, 4)}
           for s in ("A1", "A2") for i, y in enumerate(YEARS + [2026]) if i < len(YEARS) or y == 2026]
    pd.DataFrame(mod).to_csv(d / "model.csv", index=False)
    pd.DataFrame({"season_year": YEARS + [2026], "init_month": 5, "R34": (Xo["R34"].values + 0.3 * rng.normal(size=len(YEARS) + 1))}).to_csv(d / "dynidx.csv", index=False)
    base = [sys.executable, "main.py", "--stations", str(d / "obs.csv"), "--station-season", "JJAS", "--station-forecast", "--init-month", "5",
            "--target-year", "2026", "--sst-indices", str(d / "idx.csv"), "--predictors", "WIOD,EIOD", "--methods", "ridge",
            "--station-output", str(d / "out")]
    r = subprocess.run(base + ["--dynamical-stations", str(d / "model.csv"), "--dynamical-indices", str(d / "dynidx.csv")],
                       cwd=ROOT, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-1000:]
    f = pd.read_csv(d / "out" / "station_forecast_JJAS_2026_init05.csv")
    assert {"Weight_dynamical", "Forecast_dynamical", "Weight_ridge", "Weight_climatology"} <= set(f.columns) and len(f) == 2
    assert np.allclose(f[[c for c in f.columns if c.startswith("Weight_")]].sum(axis=1), 1, atol=1e-3)
    diag = pd.read_csv(d / "out" / "predictor_diagnostics_JJAS_init05.csv", index_col=0)
    assert "DYN_R34" in diag.index
    pd.DataFrame({"season_year": YEARS + [2026], "init_month": 3, "R34": 0.0}).to_csv(d / "wrong.csv", index=False)
    r2 = subprocess.run(base[:-2] + ["--station-output", str(d / "out2"), "--dynamical-indices", str(d / "wrong.csv")],
                        cwd=ROOT, capture_output=True, text=True, timeout=280)
    assert r2.returncode != 0 and "init_month" in r2.stderr


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
