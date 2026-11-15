"""The demo data generator and the demo commands run end to end (python tests/test_demo.py)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_end_to_end():
    d = Path(tempfile.mkdtemp())
    r = subprocess.run([sys.executable, "examples/make_demo_data.py", str(d / "demo")], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
    obs = pd.read_csv(d / "demo" / "stations_long.csv")
    assert set(obs["Variable"]) == {"precipitation", "tmax"} and obs["Station"].nunique() == 6 and obs["Date"].max() == "2026-05-31"
    common = ["--stations", str(d / "demo" / "stations_long.csv"), "--sst-indices", str(d / "demo" / "predictors.csv")]
    r = subprocess.run([sys.executable, "main.py", *common, "--station-season", "JJAS", "--station-forecast", "--init-month", "5",
                        "--target-year", "2026", "--methods", "ridge", "--wet-day-threshold", "1.0", "--station-output", str(d / "seasonal")],
                       cwd=ROOT, capture_output=True, text=True, timeout=290)
    assert r.returncode == 0, r.stderr[-800:]
    f = pd.read_csv(d / "seasonal" / "station_forecast_JJAS_2026_init05.csv")
    assert set(f["Variable"]) == {"precipitation", "tmax", "wet_day_fraction"} and len(f) == 18
    assert np.allclose(f[["P_below", "P_normal", "P_above"]].sum(axis=1), 1, atol=5e-4)
    r = subprocess.run([sys.executable, "main.py", *common, "--subseasonal", "--init-date", "2026-06-01", "--lead-weeks", "2", "4",
                        "--methods", "ridge", "--station-output", str(d / "sub")], cwd=ROOT, capture_output=True, text=True, timeout=290)
    assert r.returncode == 0, r.stderr[-800:]
    g = pd.read_csv(d / "sub" / "subseasonal_forecast_2026-06-01_W2-4.csv")
    assert set(g["lead_weeks"].astype(str)) == {"2", "3", "4", "2-4"}
    assert (ROOT / "examples" / "run_demo.sh").exists()


if __name__ == "__main__":
    try:
        test_demo_runs_end_to_end(); print("PASS test_demo_runs_end_to_end")
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAIL test_demo_runs_end_to_end", repr(e)); sys.exit(1)
