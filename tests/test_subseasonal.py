"""Tests for subseasonal forecasting (python tests/test_subseasonal.py  or  pytest tests)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_seasonal import make_indices  # noqa: E402

from src.forecasting import seasonal as S  # noqa: E402
from src.forecasting import subseasonal as SS  # noqa: E402
from src.verification import probabilistic_scores as V  # noqa: E402

SPEC = SS.SubseasonalSpec(6, 1, (2, 4), pool_weeks=1, recent_weeks=2, predictor_months=3)
YEARS = list(range(1981, 2026))


def make_obs(signal=True, persist=0.0, seed=0, stations=("A", "B", "C"), dry=False):
    """Daily rain for 1981-2025 (+ nothing for 2026).  Rain from 1 Jun to 31 Jul is scaled by exp(-0.5 * R34 Mar-May)."""
    ind = make_indices()
    r34 = S.build_predictors(ind, S.ForecastSpec("JJAS", 5), YEARS)["R34"]
    r34 = (r34 / r34.std())
    rng = np.random.default_rng(seed)
    days = pd.date_range("1981-01-01", "2025-12-31")
    rows = []
    for k, stn in enumerate(stations):
        wet = np.isin(days.month, [6, 7, 8, 9])
        base = np.where(wet, rng.gamma(2.0, 4.0, len(days)), rng.gamma(0.3, 0.6, len(days)))
        if dry:
            base = np.where(days.month <= 4, 0.0, base)
        fac = np.where((days.month == 6) | (days.month == 7), np.exp(-0.5 * r34.reindex(days.year).values), 1.0) if signal else 1.0
        if persist:                                                         # slowly varying wetness (memory ~ a month)
            ar = np.zeros(len(days)); e = rng.normal(0, 0.25, len(days))
            for i in range(1, len(days)): ar[i] = 0.97 * ar[i - 1] + e[i]
            fac = fac * np.exp(persist * ar)
        rain = base * fac
        rows.append(pd.DataFrame({"station": stn, "date": days, "lat": 10.0 + k, "lon": 35.0 + k, "variable": "precipitation", "value": rain}))
    return ind, pd.concat(rows, ignore_index=True)


def test_spec_dates_and_windows():
    s = SPEC; d = s.describe(2026)
    assert d["init_date"] == "2026-06-01" and d["target_start"] == "2026-06-08" and d["target_end"] == "2026-06-28"
    assert d["lead_weeks"] == "2-4" and d["forecast_period"] == "W2-4_2026-06-01" and d["resolution"] == "3-week window"
    w = SS.SubseasonalSpec(6, 1, (3, 3)).describe(2026)
    assert w["target_start"] == "2026-06-15" and w["target_end"] == "2026-06-21" and w["resolution"] == "weekly"
    assert SS.SubseasonalSpec(6, 1, (3, 6)).target_window(2026)[1] == pd.Timestamp("2026-07-12")
    assert s.recent_window(2026) == (pd.Timestamp("2026-05-18"), pd.Timestamp("2026-05-31"))
    assert [m.strftime("%Y-%m") for m in s.sst_months(2026)] == ["2026-03", "2026-04", "2026-05"]
    assert [m.strftime("%Y-%m") for m in SS.SubseasonalSpec(6, 15).sst_months(2026)] == ["2026-03", "2026-04", "2026-05"]   # June incomplete
    for bad in ((0, 3), (3, 2), (1, 13)):
        try:
            SS.SubseasonalSpec(6, 1, bad); raise AssertionError
        except ValueError:
            pass


def test_window_aggregation_and_completeness():
    idx = pd.date_range("2000-01-01", "2000-01-31")
    daily = pd.DataFrame({"x": 2.0, "y": 2.0}, index=idx)
    daily.loc["2000-01-10", "y"] = np.nan                                   # 1 of 7 days missing in the window
    a, b = pd.Timestamp("2000-01-08"), pd.Timestamp("2000-01-14")
    assert np.allclose(SS.window_value(daily, a, b, "sum"), [14.0, np.nan], equal_nan=True)          # 6/7 < 0.9 -> NaN
    assert np.allclose(SS.window_value(daily, a, b, "sum", 0.8), [14.0, 14.0])                       # scaled mean x length
    assert np.allclose(SS.window_value(daily, a, b, "mean", 0.8), [2.0, 2.0])
    assert np.isnan(SS.window_value(daily, pd.Timestamp("2000-01-28"), pd.Timestamp("2000-02-03"), "sum")).all()   # outside record


def test_predictors_use_only_data_before_init():
    ind, obs = make_obs()
    daily = SS.prepare_daily(obs, "precipitation")
    sm = SS.build_samples(daily, ind, SPEC, [2010, 2026], "precipitation")
    i = int(np.where((sm.meta.year == 2010) & (sm.meta["shift"] == 0))[0][0])
    ind2 = ind.copy(); ind2.loc["2010-06-01":] += 99.0                       # corrupt everything from June on
    d2 = daily.copy(); d2.loc["2010-06-01":, :] = 1e6                        # ... and all rain from the init date on
    sm2 = SS.build_samples(d2, ind2, SPEC, [2010], "precipitation")
    j = int(np.where((sm2.meta.year == 2010) & (sm2.meta["shift"] == 0))[0][0])
    assert np.allclose(sm.X.iloc[i], sm2.X.iloc[j]) and np.allclose(sm.R[i], sm2.R[j])
    assert not np.allclose(sm.Y[i], sm2.Y[j])                                # only the target changed
    k = int(np.where((sm.meta.year == 2026) & (sm.meta["shift"] == 0))[0][0])
    assert np.isnan(sm.Y[k]).all() and np.isfinite(sm.X.iloc[k]).all() and np.isnan(sm.R[k]).all()   # synthetic record ends 2025: no 2026 rain, indices known


def test_training_embargo():
    ind, obs = make_obs()
    sm = SS.build_samples(SS.prepare_daily(obs, "precipitation"), ind, SPEC, YEARS, "precipitation")
    init = SPEC.init_date(2000)
    tr = SS._train_mask(sm, 2000, init)
    assert sm.meta.loc[tr, "year"].max() == 1999 and (sm.meta.loc[tr, "target_end"] < init).all()
    late = SS.SubseasonalSpec(12, 20, (2, 4), pool_weeks=2)                  # target windows reach into the next January
    sm2 = SS.build_samples(SS.prepare_daily(obs, "precipitation"), ind, late, YEARS, "precipitation")
    tr2 = SS._train_mask(sm2, 2000, late.init_date(2000))
    assert (sm2.meta.loc[tr2, "target_end"] < late.init_date(2000)).all()


def test_hindcast_never_sees_test_year_or_later():
    ind, obs = make_obs()
    daily = SS.prepare_daily(obs, "precipitation")
    sm = SS.build_samples(daily, ind, SPEC, YEARS, "precipitation")
    h1 = SS.hindcast(sm, SPEC, "ridge")
    t = 2012
    d2 = daily.copy(); d2.loc[f"{t}-01-01":, :] = 1e5                        # corrupt the test year and everything after
    sm2 = SS.build_samples(d2, ind, SPEC, YEARS, "precipitation")
    h2 = SS.hindcast(sm2, SPEC, "ridge")
    for f in ("forecast", "clim", "t1", "t2", "p_below", "p_above"):
        assert np.allclose(getattr(h1, f).loc[t], getattr(h2, f).loc[t]), f
    assert h1.forecast.index.min() == 1981 + 10


def test_planted_signal_gives_skill_and_noise_does_not():
    ind, obs = make_obs(signal=True)
    hs = SS.hindcast(SS.build_samples(SS.prepare_daily(obs, "precipitation"), ind, SPEC, YEARS, "precipitation"), SPEC, "ridge")
    s = V.summarize(hs)
    assert s["pooled"]["rpss"] > 0.08 and np.allclose(hs.probs.sum(-1), 1, atol=1e-9)
    assert (np.array([("R34" in x or "TNA" in x) for x in hs.selected.values.ravel()]).mean() > 0.4)
    ind2, obs2 = make_obs(signal=False, seed=3)
    hn = SS.hindcast(SS.build_samples(SS.prepare_daily(obs2, "precipitation"), ind2, SPEC, YEARS, "precipitation"), SPEC, "ridge")
    assert V.summarize(hn)["pooled"]["rpss"] < 0.06, V.summarize(hn)["pooled"]["rpss"]


def test_persistence_predictor_is_used_when_it_carries_information():
    gains, used = [], []
    for seed in (7, 11, 13):                                              # skill at weeks 2-4 is small: average over seeds
        ind, obs = make_obs(signal=False, persist=0.8, seed=seed)
        sm = SS.build_samples(SS.prepare_daily(obs, "precipitation"), ind, SPEC, YEARS, "precipitation")
        h = SS.hindcast(sm, SPEC, "ridge"); hn = SS.hindcast(sm, SPEC, "ridge", use_persistence=False)
        gains.append(V.summarize(h)["pooled"]["rpss"] - V.summarize(hn)["pooled"]["rpss"])
        used.append(np.mean([("persist" in x) for x in h.selected.values.ravel()]))
    assert np.mean(used) > 0.4 and np.mean(gains) > 0.03 and min(gains) > 0, (used, gains)


def test_weekly_totals_climatology_forecast_has_no_systematic_anomaly():
    ind, obs = make_obs(signal=False, seed=21)                                # skewed 3-week rainfall totals
    daily = SS.prepare_daily(obs, "precipitation")
    sm = SS.build_samples(daily, ind, SPEC, YEARS + [2026], "precipitation")
    f = SS.forecast(sm, SPEC, 2026, "climatology")
    assert (np.abs(f["Forecast"] / f["Climatology"] - 1) < 0.05).all(), (f["Forecast"] / f["Climatology"]).tolist()
    h = SS.hindcast(sm, SPEC, "climatology")
    scored = h.obs.notna().all(axis=1).values                                                   # 2026 has no observations in this record
    fc, cl = h.forecast.values[scored], h.clim.values[scored]
    assert scored.sum() > 20 and abs(fc.mean() / cl.mean() - 1) < 0.01                          # forecast = training mean (a median would be ~10% low)
    assert abs(fc.mean() / h.obs.values[scored].mean() - 1) < 0.08                              # and no worse than sampling noise against the outcomes


def test_forecast_table_format_and_target_year_isolation():
    ind, obs = make_obs()
    meta = pd.DataFrame({"station": ["A", "B", "C"], "lat": [10, 11, 12], "lon": [35, 36, 37]})
    daily = SS.prepare_daily(obs, "precipitation")
    sm = SS.build_samples(daily, ind, SPEC, YEARS + [2026], "precipitation")
    f = SS.forecast(sm, SPEC, 2026, "ridge", meta=meta)
    assert list(f.columns[:7]) == ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"]
    assert (f["Forecast_Period"] == "W2-4_2026-06-01").all() and (f["target_start"] == "2026-06-08").all()
    assert np.allclose(f[["P_below", "P_normal", "P_above"]].sum(axis=1), 1) and (f["Forecast"] >= 0).all()
    assert np.allclose(f["Anomaly"], f["Forecast"] - f["Climatology"])
    ind2 = ind.copy(); ind2.loc["2026-06-01":] = 1e3                          # data from the init date on must not matter
    f2 = SS.forecast(SS.build_samples(daily, ind2, SPEC, YEARS + [2026], "precipitation"), SPEC, 2026, "ridge", meta=meta)
    assert np.allclose(f["Forecast"], f2["Forecast"])
    short = ind.loc[:"2026-04-01"]                                            # May 2026 missing -> clear error
    try:
        SS.forecast(SS.build_samples(daily, short, SPEC, YEARS + [2026], "precipitation"), SPEC, 2026, "ridge"); raise AssertionError
    except ValueError as e:
        assert "incomplete" in str(e)


def test_dry_season_is_not_forecast():
    ind, obs = make_obs(dry=True)
    dry = SS.SubseasonalSpec(1, 15, (2, 4))
    sm = SS.build_samples(SS.prepare_daily(obs, "precipitation"), ind, dry, YEARS + [2026], "precipitation")
    f = SS.forecast(sm, dry, 2026, "ridge")
    assert f["Forecast"].isna().all() and f["Predictors"].str.contains("dry-season").all()
    h = SS.hindcast(sm, dry, "ridge")
    assert len(h.forecast) == 0 or h.forecast.isna().all().all()


def test_ensemble_over_methods_and_weekly_resolution():
    ind, obs = make_obs()
    meta = pd.DataFrame({"station": ["A", "B", "C"], "lat": [10, 11, 12], "lon": [35, 36, 37]})
    fc, hc, allh = SS.run_station_subseasonal(obs, meta, ind, SPEC, 2025, "precipitation", methods=("ridge", "pcr"))
    assert {"climatology", "ridge", "pcr", "ensemble"} <= set(allh)
    assert np.allclose(fc[["P_below", "P_normal", "P_above"]].sum(axis=1), 1) and (fc["Method"].str.startswith("ensemble")).all()
    cmp_ = V.compare(allh, n_boot=60)
    assert abs(cmp_.set_index("method").loc["climatology", "rpss"]) < 1e-9
    week3 = SS.SubseasonalSpec(6, 1, (3, 3), 1)
    f3, h3, _ = SS.run_station_subseasonal(obs, meta, ind, week3, 2025, "precipitation", methods=("ridge",))
    assert (f3["lead_weeks"] == "3").all() and (f3["target_end"] == "2025-06-21").all()


def test_cli_end_to_end():
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    d = Path(tempfile.mkdtemp())
    ind, obs = make_obs(signal=True)
    ind.reset_index(names="Date").to_csv(d / "idx.csv", index=False)
    long = obs.rename(columns={"station": "Station", "date": "Date", "lat": "Lat", "lon": "Lon", "variable": "Variable", "value": "Value"})
    long["Date"] = long["Date"].dt.strftime("%Y-%m-%d"); long.round(2).to_csv(d / "obs.csv", index=False)
    out = d / "out"
    r = subprocess.run([sys.executable, "main.py", "--stations", str(d / "obs.csv"), "--subseasonal", "--init-date", "2025-06-01",
                        "--lead-weeks", "2", "4", "--predictors", "R34,DMI", "--sst-indices", str(d / "idx.csv"),
                        "--methods", "ridge", "--station-output", str(out)], cwd=root, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-1200:]
    f = pd.read_csv(out / "subseasonal_forecast_2025-06-01_W2-4.csv")
    assert set(f["lead_weeks"].astype(str)) == {"2", "3", "4", "2-4"} and len(f) == 4 * 3
    assert list(f.columns[:7]) == ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"]
    assert f.loc[f["lead_weeks"].astype(str) == "2-4", "target_end"].iloc[0] == "2025-06-28"
    assert f.loc[f["lead_weeks"].astype(str) == "3", "target_start"].iloc[0] == "2025-06-15"
    assert np.allclose(f[["P_below", "P_normal", "P_above"]].sum(axis=1), 1, atol=5e-4)            # CSV values are rounded to 4 decimals
    sk = pd.read_csv(out / "subseasonal_skill_summary_2025-06-01_W2-4.csv")
    assert len(sk) == 4 and {"rpss", "rpss_lo_90", "rpss_hi_90", "n_hindcast_years"} <= set(sk.columns)
    meta = json.loads((out / "subseasonal_run_metadata.json").read_text())
    assert meta["lead_weeks"] == [2, 4] and (out / "subseasonal_forecast_2025-06-01_W2-4.xlsx").exists()


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
