"""Tests for extra ocean predictors, predictor selection, SST as a forecast target (python tests/test_predictors.py)."""
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_seasonal import SPEC, YEARS, make_indices  # noqa: E402
from test_grid import LAT, LON  # noqa: E402

from src.forecasting import seasonal as S  # noqa: E402
from src.preprocessing import grid_processing as G  # noqa: E402
from src.teleconnections import sst_indices as SI  # noqa: E402
from src.verification import probabilistic_scores as V  # noqa: E402


def extras_abs(seed=2, start="1981-01-01", end="2026-12-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="MS"); m = idx.month.values
    def ar(phi, s):
        x = np.zeros(len(idx)); e = rng.normal(0, s, len(idx))
        for i in range(1, len(idx)): x[i] = phi * x[i - 1] + e[i]
        return x
    return pd.DataFrame({"NINO3": 26.5 + 0.9 * np.cos(2 * np.pi * (m - 12) / 12) + ar(0.93, 0.3),
                         "TNA": 27 + 1.5 * np.cos(2 * np.pi * (m - 8) / 12) + ar(0.88, 0.25),
                         "TSA": 26 + 1.2 * np.cos(2 * np.pi * (m - 3) / 12) + ar(0.88, 0.25)}, index=idx)


def test_extra_predictors_anomalies_and_gradient():
    ex = extras_abs(); core = make_indices()
    t = SI.extra_predictors(ex, core)
    assert {"R34", "DMI", "NINO3", "TNA", "TSA", "ATL_GRAD"} <= set(t.columns)
    assert np.allclose(t["ATL_GRAD"], t["TNA"] - t["TSA"], equal_nan=True)
    base = t.loc["1991":"2020"]
    assert np.allclose(base["TNA"].groupby(base.index.month).mean(), 0, atol=1e-9)          # monthly climatology removed
    only = SI.extra_predictors(ex)
    assert "R34" not in only.columns and only.shape[0] == len(ex)


def test_load_predictor_table_and_column_selection():
    d = Path(tempfile.mkdtemp())
    t = SI.extra_predictors(extras_abs(), make_indices()).round(4)
    out = pd.DataFrame({"Date": t.index.strftime("%Y-%m-%d"), "Year": t.index.year, "Month": t.index.month}).join(t.reset_index(drop=True))
    out.to_csv(d / "p.csv", index=False)
    full = SI.load_predictor_table(d / "p.csv")
    assert "Year" not in full.columns and "TNA" in full.columns and isinstance(full.index, pd.DatetimeIndex)
    assert list(SI.load_predictor_table(d / "p.csv", ["R34", "TNA"]).columns) == ["R34", "TNA"]
    for bad in (lambda: SI.load_predictor_table(d / "p.csv", ["NOPE"]), lambda: SI.load_predictor_table(d / "none.csv")):
        try:
            bad(); raise AssertionError
        except SystemExit:
            pass


def _table():
    return SI.extra_predictors(extras_abs(), make_indices())


def test_atlantic_signal_found_among_many_predictors_and_restriction_hurts():
    tab = _table()
    Xall = S.build_predictors(tab, SPEC, YEARS)
    rng = np.random.default_rng(8)
    drv = (Xall["TNA"] / Xall["TNA"].std()).loc[YEARS].values
    Y = pd.DataFrame({f"S{j}": 400 + 60 * (drv + 0.6 * rng.normal(size=len(YEARS))) for j in range(4)}, index=YEARS)
    h_all = S.hindcast(Y, Xall, SPEC)
    sel = h_all.selected.values.ravel()
    assert np.mean([("TNA" in x.split("+")) or ("ATL_GRAD" in x.split("+")) for x in sel]) > 0.95   # the planted driver is always used
    core = S.build_predictors(tab[["R34", "WIOD", "EIOD", "DMI"]], SPEC, YEARS)
    r_all = V.summarize(h_all)["pooled"]["rpss"]; r_core = V.summarize(S.hindcast(Y, core, SPEC))["pooled"]["rpss"]
    assert r_all > 0.1 and r_all > r_core + 0.08, (r_all, r_core)
    d = S.predictor_diagnostics(Xall)
    assert d["vif"]["DMI"] > 10 and "ATL_GRAD" in d["vif"].index                       # collinear predictors are flagged


def test_min_gain_limits_chance_predictors_but_keeps_real_signal():
    tab = _table(); X = S.build_predictors(tab, SPEC, YEARS)
    share = {0.0: [], S.MIN_GAIN: []}
    for seed in range(4):
        Y = pd.DataFrame(np.random.default_rng(100 + seed).normal(400, 60, (len(YEARS), 6)), index=YEARS, columns=list("abcdef"))
        for g in share:
            share[g].append((S.hindcast(Y, X, SPEC, min_gain=g).selected.values != "climatology").mean())
    assert np.mean(share[S.MIN_GAIN]) < 0.6 * np.mean(share[0.0])                       # far fewer chance fits on pure noise
    drv = (X["TNA"] / X["TNA"].std()).loc[YEARS].values
    Y = pd.DataFrame({f"S{j}": 400 + 60 * (drv + 0.6 * np.random.default_rng(j).normal(size=len(YEARS))) for j in range(4)}, index=YEARS)
    a, b = (V.summarize(S.hindcast(Y, X, SPEC, min_gain=g))["pooled"]["rpss"] for g in (0.0, S.MIN_GAIN))
    assert b > 0.9 * a                                                                   # a strong real signal is not given up


def test_pipeline_writes_extra_predictors_without_touching_core_outputs():
    d = Path(tempfile.mkdtemp())
    core = make_indices(); ex = extras_abs()
    absolute = pd.DataFrame({"R34": 27 + core["R34"], "WIOD": 28.5 + core["WIOD"], "EIOD": 28.2 + core["EIOD"]}).join(ex)
    absolute.index.name = "Date"; absolute.to_csv(d / "regional.csv")
    r = subprocess.run([sys.executable, "main.py", "--from-regional-csv", str(d / "regional.csv"), "--output", str(d / "out"),
                        "--skip-forecast"], cwd=ROOT, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-800:]
    ind = d / "out" / "indices"
    comb = pd.read_csv(ind / "all_climate_indices_monthly.csv"); pred = pd.read_csv(ind / "all_predictors_monthly.csv")
    assert list(comb.columns) == ["Date", "Year", "Month", "R34", "WIOD", "EIOD", "DMI"]       # core outputs unchanged
    assert {"NINO3", "TNA", "TSA", "ATL_GRAD", "R34", "DMI"} <= set(pred.columns)
    assert np.allclose(pred["R34"], comb["R34"])
    assert "NINO3" not in pd.read_csv(ind / "regional_monthly_sst_degC.csv").columns


def test_station_cli_end_to_end_with_predictor_restriction():
    d = Path(tempfile.mkdtemp()); tab = _table(); tab.reset_index(names="Date").to_csv(d / "idx.csv", index=False)
    X = S.build_predictors(tab, SPEC, YEARS); rng = np.random.default_rng(6)
    days = pd.date_range("1981-01-01", "2025-12-31"); rows = []
    sig = (X["TNA"] / X["TNA"].std()).reindex(days.year).values
    for k, stn in enumerate(["A1", "A2"]):
        jjas = np.isin(days.month, [6, 7, 8, 9])
        rain = np.where(jjas, rng.gamma(2, 4, len(days)), rng.gamma(0.4, 1, len(days))) * np.where(jjas, np.exp(0.4 * sig), 1.0)
        rows.append(pd.DataFrame({"Station": stn, "Date": days.strftime("%Y-%m-%d"), "Lat": 10 + k, "Lon": 35 + k, "Variable": "precipitation", "Value": rain.round(2)}))
    pd.concat(rows).to_csv(d / "obs.csv", index=False)
    out = d / "out"
    r = subprocess.run([sys.executable, "main.py", "--stations", str(d / "obs.csv"), "--station-season", "JJAS", "--station-forecast",
                        "--init-month", "5", "--target-year", "2026", "--sst-indices", str(d / "idx.csv"), "--predictors", "R34,TNA,TSA",
                        "--methods", "ridge", "--station-output", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-800:]
    f = pd.read_csv(out / "station_forecast_JJAS_2026_init05.csv")
    assert list(f.columns[:7]) == ["Station", "lat", "lon", "Forecast_Period", "Variable", "Forecast", "Anomaly"] and len(f) == 2
    assert f["Predictors"].str.contains("TNA").all() and set(sum([p.split("+") for p in f["Predictors"]], [])) <= {"R34", "TNA", "TSA"}
    diag = pd.read_csv(out / "predictor_diagnostics_JJAS_init05.csv", index_col=0)
    assert list(diag.index) == ["R34", "TNA", "TSA"] and (out / "station_forecast_JJAS_2026_init05.xlsx").exists()


def test_sst_as_forecast_target_and_diagnostics_file():
    tab = _table(); X = S.build_predictors(tab, SPEC, YEARS)
    rng = np.random.default_rng(4)
    t = pd.date_range("1981-01-01", "2025-12-01", freq="MS")
    r34 = X["R34"].reindex(t.year).values
    data = np.empty((len(t), len(LAT), len(LON)))
    for i, ts in enumerate(t):
        season = 28 + np.cos(2 * np.pi * (ts.month - 4) / 12)
        data[i] = season + (0.6 * r34[i] if ts.month in (6, 7, 8, 9) else 0.0) + rng.normal(0, 0.15, (len(LAT), len(LON)))
    grid = G.MonthlyGrid(data, t, LAT, LON, "sst", "degC")
    d = Path(tempfile.mkdtemp()); (tab.reset_index(names="Date")).to_csv(d / "idx.csv", index=False)
    args = Namespace(grid_variable="sst", target_year=2026, init_month=5, grid_nc=None, sst_indices=d / "idx.csv", predictors="R34,TNA,DMI",
                     grid_output=d / "out", season="JJAS", predictor_months=3, grid_region=None, bbox=None, clip_file=None,
                     grid_coarsen=1, methods="ridge", hindcast_scheme="rolling", ensemble_mode="skill", max_predictors=2)
    res = G.run(args, grid=grid)
    assert np.nanmean(res["skill"]["rpss"]) > 0.15 and np.nanmean(res["skill"]["correlation"]) > 0.5
    assert abs(np.nanmean(res["climatology"]) - 27.8) < 0.3                                   # JJAS mean SST in degC (averaged, not summed)
    files = sorted(p.name for p in (d / "out").iterdir())
    assert any(f.startswith("sst_JJAS_2026") and f.endswith(".nc") for f in files)
    diag = pd.read_csv(next((d / "out").glob("predictor_diagnostics_*.csv")), index_col=0)
    assert list(diag.index) == ["R34", "TNA", "DMI"] and "VIF" in diag.columns               # --predictors respected
    from scipy.io import netcdf_file
    f = netcdf_file(str(next((d / "out").glob("*.nc"))), "r", mmap=False)
    assert f.variable.decode() == "sst"; f.close()


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
