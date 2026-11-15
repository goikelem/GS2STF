"""End-to-end leakage audit: forecasts must not change when everything after the initialization date is replaced by garbage.

Each workflow is run twice through the real command-line code: (a) data that stop at the initialization date, (b) the same data
plus a 'future' (observations of the target period, ocean indices after the init month) filled with absurd values.
The forecast tables must be identical.  (python tests/test_leakage_e2e.py)
"""
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from examples.make_demo_data import main as make_demo  # noqa: E402

FORECAST_COLS = ["Forecast", "Anomaly", "Climatology", "P_below", "P_normal", "P_above", "Tercile_lower", "Tercile_upper"]


def _demo():
    d = Path(tempfile.mkdtemp()); make_demo(d / "demo")
    obs = pd.read_csv(d / "demo" / "stations_long.csv"); ind = pd.read_csv(d / "demo" / "predictors.csv")
    return d, obs, ind


def _future_obs(obs, start="2026-06-01", end="2026-12-31"):
    days = pd.date_range(start, end); rows = []
    rng = np.random.default_rng(0)
    for (stn, lat, lon), g in obs.groupby(["Station", "Lat", "Lon"]):
        for var, val in (("precipitation", 500.0), ("tmax", 60.0)):
            rows.append(pd.DataFrame({"Station": stn, "Date": days.strftime("%Y-%m-%d"), "Lat": lat, "Lon": lon, "Variable": var,
                                      "Value": val + rng.normal(0, 1, len(days))}))
    return pd.concat([obs] + rows, ignore_index=True)


def _run(args, cwd=ROOT):
    r = subprocess.run([sys.executable, "main.py", *args], cwd=cwd, capture_output=True, text=True, timeout=290)
    assert r.returncode == 0, r.stderr[-1000:]


def test_seasonal_station_forecast_ignores_the_future():
    d, obs, ind = _demo()
    ind_bad = ind.copy(); ind_bad.loc[pd.to_datetime(ind_bad["Date"]) >= "2026-06-01", ["R34", "WIOD", "EIOD", "TNA", "TSA", "DMI"]] = 99.0
    obs.to_csv(d / "obs_clean.csv", index=False); _future_obs(obs).to_csv(d / "obs_future.csv", index=False)
    ind.to_csv(d / "ind_clean.csv", index=False); ind_bad.to_csv(d / "ind_bad.csv", index=False)
    common = ["--station-season", "JJAS", "--station-forecast", "--init-month", "5", "--target-year", "2026", "--methods", "ridge,pcr",
              "--wet-day-threshold", "1.0"]
    _run(["--stations", str(d / "obs_clean.csv"), "--sst-indices", str(d / "ind_clean.csv"), *common, "--station-output", str(d / "a")])
    _run(["--stations", str(d / "obs_future.csv"), "--sst-indices", str(d / "ind_bad.csv"), *common, "--station-output", str(d / "b")])
    a = pd.read_csv(d / "a" / "station_forecast_JJAS_2026_init05.csv"); b = pd.read_csv(d / "b" / "station_forecast_JJAS_2026_init05.csv")
    assert len(a) == len(b) == 18 and list(a["Station"]) == list(b["Station"])
    for c in FORECAST_COLS:
        assert np.allclose(a[c], b[c], equal_nan=True, rtol=1e-9, atol=1e-9), c
    wc = [c for c in a.columns if c.startswith("Weight_")]
    assert np.allclose(a[wc], b[wc], equal_nan=True)                                    # ensemble weights use no future outcome either


def test_subseasonal_forecast_ignores_the_future():
    d, obs, ind = _demo()
    ind_bad = ind.copy(); ind_bad.loc[pd.to_datetime(ind_bad["Date"]) >= "2026-06-01", ["R34", "WIOD", "EIOD", "TNA", "TSA", "DMI"]] = -99.0
    obs.to_csv(d / "obs_clean.csv", index=False); _future_obs(obs).to_csv(d / "obs_future.csv", index=False)
    ind.to_csv(d / "ind_clean.csv", index=False); ind_bad.to_csv(d / "ind_bad.csv", index=False)
    common = ["--subseasonal", "--init-date", "2026-06-01", "--lead-weeks", "2", "4", "--methods", "ridge"]
    _run(["--stations", str(d / "obs_clean.csv"), "--sst-indices", str(d / "ind_clean.csv"), *common, "--station-output", str(d / "a")])
    _run(["--stations", str(d / "obs_future.csv"), "--sst-indices", str(d / "ind_bad.csv"), *common, "--station-output", str(d / "b")])
    a = pd.read_csv(d / "a" / "subseasonal_forecast_2026-06-01_W2-4.csv"); b = pd.read_csv(d / "b" / "subseasonal_forecast_2026-06-01_W2-4.csv")
    assert len(a) == len(b) == 4 * 12
    for c in FORECAST_COLS:
        assert np.allclose(a[c], b[c], equal_nan=True, rtol=1e-9, atol=1e-9), c


def test_grid_forecast_ignores_the_future():
    from src.preprocessing import grid_processing as G
    from test_grid import LAT, LON, make_grid
    g, ind = make_grid()
    d = Path(tempfile.mkdtemp()); ind.reset_index(names="Date").to_csv(d / "idx.csv", index=False)
    ind_bad = ind.copy(); ind_bad.loc[ind_bad.index >= "2026-06-01"] = 99.0
    ind_bad.reset_index(names="Date").to_csv(d / "idx_bad.csv", index=False)
    t_future = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    fut = np.random.default_rng(1).uniform(0, 1e4, (len(t_future),) + g.data.shape[1:])
    g_future = G.MonthlyGrid(np.concatenate([g.data, fut]), g.time.append(t_future), g.lat, g.lon, g.kind, g.units)
    def args(idx, out):
        return Namespace(grid_variable="precipitation", target_year=2026, init_month=5, grid_nc=None, sst_indices=d / idx, grid_output=d / out,
                         season="JJAS", predictor_months=3, grid_region=None, bbox=None, clip_file=None, grid_coarsen=1, methods="ridge,pcr",
                         hindcast_scheme="rolling", ensemble_mode="skill", max_predictors=2)
    ra = G.run(args("idx.csv", "ga"), grid=g)
    rb = G.run(args("idx_bad.csv", "gb"), grid=g_future)
    for k in ("forecast", "anomaly", "climatology", "probabilities"):
        assert np.allclose(ra[k], rb[k], equal_nan=True, rtol=1e-9, atol=1e-9), k
    # negative control: corrupting an index month INSIDE the allowed window (April 2026 for a May init) must change the forecast,
    # otherwise the comparison above would be blind
    ind_in = ind.copy(); ind_in.loc["2026-04-01"] = 99.0
    ind_in.reset_index(names="Date").to_csv(d / "idx_in.csv", index=False)
    rc = G.run(args("idx_in.csv", "gc"), grid=g)
    assert not np.allclose(ra["forecast"], rc["forecast"], equal_nan=True, rtol=1e-6)


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
