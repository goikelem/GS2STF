"""Environment robustness: input-file matching, open-ended date range, Arrow-backed strings, friendly errors."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import settings  # noqa: E402
from src.data_ingestion import netcdf_loader as NL  # noqa: E402
from src.forecasting import seasonal as S  # noqa: E402


def test_resolve_input_file_exact_pattern_latest_and_missing():
    d = Path(tempfile.mkdtemp())
    (d / "prcp-chirps-ethio-198101_202609.nc").write_text("x")
    assert NL.resolve_input_file(d, "prcp-chirps-ethio-198101_202609.nc").name == "prcp-chirps-ethio-198101_202609.nc"      # exact
    assert NL.resolve_input_file(d, "prcp-chirps-ethio-198101_202607.nc").name == "prcp-chirps-ethio-198101_202609.nc"      # date block differs
    (d / "prcp-chirps-ethio-198101_202610.nc").write_text("x")
    assert NL.resolve_input_file(d, "prcp-chirps-ethio-198101_202607.nc").name == "prcp-chirps-ethio-198101_202610.nc"      # latest of several
    (d / "sst_era5_198101_202610.nc").write_text("x")
    assert NL.resolve_input_file(d, "sst_era5_198001_202607.nc").name == "sst_era5_198101_202610.nc"                       # start date differs too
    try:
        NL.resolve_input_file(d, "tmin-era5-chirts-ethio-198101_202607.nc"); raise AssertionError
    except FileNotFoundError as e:
        assert "tmin-era5-chirts-ethio-*.nc" in str(e) and "prcp-chirps-ethio-198101_202609.nc" in str(e)               # lists what IS there
    try:
        NL.resolve_input_file(d, "nodatehere.nc"); raise AssertionError
    except FileNotFoundError:
        pass


def test_default_config_has_no_end_date_and_current_file_names():
    assert settings.ANALYSIS_END is None
    assert settings.PRCP_FILE.endswith("202609.nc") and settings.SST_FILE == "sst_era5_198101_202610.nc"


def test_string_columns_reshape_for_every_string_backend():
    shape = (2, 2)
    for dtype in (object, "string", "str"):
        s = pd.Series(["a", "b", "climatology", "R34+TNA"], dtype=dtype)
        assert S._grid(s, shape).shape == shape and S._grid(s, shape)[1, 1] == "R34+TNA"
    try:                                                                       # Arrow-backed strings (pandas 3 + pyarrow), as on many user machines
        import pyarrow  # noqa: F401
    except ImportError:
        return
    s = pd.Series(["a", "b", "c", "d"], dtype="string[pyarrow]")
    assert S._grid(s, shape).shape == shape


def test_missing_input_gives_a_short_message_not_a_traceback():
    out = Path(tempfile.mkdtemp())
    r = subprocess.run([sys.executable, "main.py", "--from-regional-csv", str(out / "nope.csv"), "--output", str(out / "o")],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    text = r.stdout + r.stderr
    assert r.returncode == 2 and "nope.csv" in text and "Traceback" not in text, text[-600:]


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
