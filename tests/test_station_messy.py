"""The station loader copes with messy real-world files, and never guesses silently (python tests/test_station_messy.py)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_ingestion import station_csv_loader as L  # noqa: E402
from src.preprocessing import station_processing as P  # noqa: E402


def write(text, name="f.csv", encoding="utf-8"):
    p = Path(tempfile.mkdtemp()) / name
    p.write_bytes(text.encode(encoding)); return p


def test_bom_semicolon_decimal_comma_trace_and_missing_tokens():
    p = write("\ufeffStation;Date;Lat;Lon;Variable;Value\nA1;2000-01-01;10,4;35,0;rainfall;12,5\nA1;2000-01-02;10,4;35,0;rainfall;T\n"
              "A1;2000-01-03;10,4;35,0;rainfall;NA\nA1;2000-01-04;10,4;35,0;rainfall;3,0\nA1;2000-01-05;10,4;35,0;rainfall;-\n"
              "A1;2000-01-06;10,4;35,0;rainfall;abc\n")
    d = L.load_station_csv(p)
    o = d.observations
    assert list(o.value.fillna(-1)) == [12.5, 0.0, -1, 3.0, -1, -1] and (o.lat == 10.4).all() and (o.lon == 35.0).all()
    r = d.report
    assert r["delimiter"] == "';'" and r["trace_rain_set_to_zero"] == 1 and r["unparseable_values_set_to_nan"] == 1
    assert r["unparseable_value_examples"] == ["abc"] and (o.variable == "precipitation").all()


def test_tolerant_headers_and_value_column_named_after_variable():
    p = write("Station ID\tDate\tLatitude\tLongitude\tRain\n A1 \t2000-01-01\t10.4\t35.0\t1.0\n A1 \t2000-01-02\t10.4\t35.0\t2.0\n")
    d = L.load_station_csv(p)
    assert list(d.observations.station.unique()) == ["A1"] and (d.observations.variable == "precipitation").all()
    assert any("taken from the name of the value column" in a for a in d.report["assumptions"])
    t = write("station,date,lat,lon,Tmax\nB,2000-01-01,1,2,30.5\n")
    assert L.load_station_csv(t).observations.variable.iloc[0] == "tmax"
    q = write("Station,Date,lat,long,value\nB,2000-01-01,1,2,30.5\n")
    assert L.load_station_csv(q, variable="tmin").observations.variable.iloc[0] == "tmin"


def test_latin1_encoding_fallback():
    p = write("Station,Date,lat,long,value\nStation Wolkité,2000-01-01,8.3,37.8,1.0\n", encoding="latin-1")
    d = L.load_station_csv(p, variable="precipitation")
    assert d.report["encoding"] == "latin-1" and d.observations.station.iloc[0].startswith("Station Wolkit")


def test_dates_are_never_guessed_silently():
    unamb = write("Station,Date,lat,long,value\nA,25/01/2000,1,2,1.0\nA,26/01/2000,1,2,2.0\n")
    d = L.load_station_csv(unamb, variable="precipitation")
    assert [x.strftime("%Y-%m-%d") for x in d.observations.date] == ["2000-01-25", "2000-01-26"] and "detected" in d.report["date_format"]
    mdy = write("Station,Date,lat,long,value\nA,01/25/2000,1,2,1.0\n")
    assert L.load_station_csv(mdy, variable="precipitation").observations.date.iloc[0] == pd.Timestamp("2000-01-25")
    amb = write("Station,Date,lat,long,value\nA,01/02/2000,1,2,1.0\nA,02/02/2000,1,2,2.0\n")
    try:
        L.load_station_csv(amb, variable="precipitation"); raise AssertionError("ambiguous dates must raise")
    except ValueError as e:
        assert "ambiguous" in str(e) and "--station-dayfirst" in str(e)
    assert L.load_station_csv(amb, variable="precipitation", dayfirst=True).observations.date.iloc[0] == pd.Timestamp("2000-02-01")
    assert L.load_station_csv(amb, variable="precipitation", dayfirst=False).observations.date.iloc[0] == pd.Timestamp("2000-01-02")
    mixed = write("Station,Date,lat,long,value\nA,25/01/2000,1,2,1.0\nA,01/25/2000,1,2,2.0\n")
    twod = write("Station,Date,lat,long,value\nA,25/01/00,1,2,1.0\n")
    for bad in (mixed, twod):
        try:
            L.load_station_csv(bad, variable="precipitation"); raise AssertionError
        except ValueError:
            pass
    dots = write("Station,Date,lat,long,value\nA,31.01.2000,1,2,1.0\n")
    assert L.load_station_csv(dots, variable="precipitation", date_format="%d.%m.%Y").observations.date.iloc[0] == pd.Timestamp("2000-01-31")
    assert L.load_station_csv(dots, variable="precipitation").observations.date.iloc[0] == pd.Timestamp("2000-01-31")   # 31 > 12: detected
    iso_time = write("Station,Date,lat,long,value\nA,2000-01-31 09:00:00,1,2,1.0\nA,2000-02-01T09:00:00,1,2,1.0\n")
    assert len(L.load_station_csv(iso_time, variable="precipitation").observations) == 2


def test_unit_problems_are_reported_and_impossible_values_removed():
    rows = ["Station,Date,lat,long,value"] + [f"A,{d.strftime('%Y-%m-%d')},1,2,{285 + (i % 5)}" for i, d in enumerate(pd.date_range("2000-01-01", periods=60))]
    d = L.load_station_csv(write("\n".join(rows)), variable="tmax")
    assert d.report["physically_impossible_values_set_to_nan"] == 60 and any("tenths of degrees" in w for w in d.report["warnings"])
    wet = ["Station,Date,lat,long,value"] + [f"A,{x.strftime('%Y-%m-%d')},1,2,{600 + i}" for i, x in enumerate(pd.date_range("2000-01-01", periods=60))]
    r = L.load_station_csv(write("\n".join(wet)), variable="precipitation").report
    assert any("tenths of mm" in w for w in r["warnings"]) and r["physically_impossible_values_set_to_nan"] == 0
    good = [f"A,{x.strftime('%Y-%m-%d')},1,2,1.0" for x in pd.date_range("2000-01-01", periods=300)]
    bad_dates = write("\n".join(["Station,Date,lat,long,value"] + good + ["A,not a date,1,2,1.0", "A,,1,2,1.0"]))
    r = L.load_station_csv(bad_dates, variable="precipitation")
    assert r.report["dropped_unparseable_dates"] == 2 and len(r.observations) == 300           # a few bad rows are dropped and counted
    half = write("Station,Date,lat,long,value\nA,2000-01-01,1,2,1.0\nA,not a date,1,2,1.0\nA,,1,2,1.0\n")
    try:
        L.load_station_csv(half, variable="precipitation"); raise AssertionError("a mostly-garbage date column must raise")
    except ValueError:
        pass


def test_coverage_report():
    d = pd.date_range("2000-01-01", "2002-12-31")
    obs = pd.DataFrame({"station": "A", "date": d, "lat": 1.0, "lon": 2.0, "variable": "precipitation", "value": 1.0})
    obs.loc[(obs.date >= "2001-03-01") & (obs.date <= "2001-04-09"), "value"] = np.nan             # a 40-day gap
    obs.loc[obs.date < "2000-02-01", "value"] = np.nan                                              # leading gap is outside the record
    c = P.coverage_report(obs).iloc[0]
    assert c["longest_gap_days"] == 40 and str(c["first"]) == "2000-02-01" and str(c["last"]) == "2002-12-31"
    assert c["days_in_record"] == (pd.Timestamp("2002-12-31") - pd.Timestamp("2000-02-01")).days + 1
    assert abs(c["pct_missing"] - 100 * 40 / c["days_in_record"]) < 0.01
    assert c["years_with_90pct_days"] == 2                                  # 2000: 335/366 = 91.5% ok, 2001: 325/365 = 89.0% fails, 2002 ok


def test_folder_of_files_is_combined():
    d = Path(tempfile.mkdtemp())
    days = pd.date_range("2000-01-01", periods=10)
    for k, stn in enumerate(["S1", "S2"]):                                   # one file per station, different separators
        pd.DataFrame({"Station": stn, "Date": days.strftime("%Y-%m-%d"), "lat": 10.0 + k, "long": 35.0, "value": float(k)}).to_csv(
            d / f"{stn}.csv", index=False, sep=";" if k else ",")
    pd.DataFrame({"Station": "S1", "Date": days[:3].strftime("%Y-%m-%d"), "lat": 10.5, "long": 35.0, "value": 9.0}).to_csv(d / "S1_extra.csv", index=False)
    r = L.load_station_csv(d, variable="precipitation")
    assert r.report["n_files"] == 3 and r.report["duplicate_records_across_files_dropped"] == 3
    assert r.report["stations_with_conflicting_coordinates"] == ["S1"] and len(r.observations) == 20
    assert set(r.stations.station) == {"S1", "S2"} and abs(r.stations.set_index("station").loc["S1", "lat"] - 10.0) < 1e-9     # median of 10 x 10.0 and 3 x 10.5
    (d / "bad.csv").write_text("Station,Date,lat,long,value\nA,01/02/2000,1,2,1\n")
    try:
        L.load_station_csv(d, variable="precipitation"); raise AssertionError
    except ValueError as e:
        assert "bad.csv" in str(e)                                           # the offending file is named
    try:
        L.load_station_csv(Path(tempfile.mkdtemp()), variable="precipitation"); raise AssertionError
    except FileNotFoundError:
        pass


def test_cli_messages_flags_and_coverage_file():
    d = Path(tempfile.mkdtemp())
    days = pd.date_range("1995-01-01", "2005-12-31")
    amb = pd.DataFrame({"Station": "A1", "Date": days.strftime("%d/%m/%Y"), "lat": 10.0, "long": 35.0, "value": 1.0})
    amb = amb[amb.Date.str[:2].astype(int) <= 12]                                    # keep only days <= 12: format is ambiguous
    amb["Date"] = pd.to_datetime(amb["Date"], format="%d/%m/%Y").dt.strftime("%d/%m/%Y")
    amb = amb[amb.Date.str[3:5].astype(int) <= 12]
    amb.to_csv(d / "amb.csv", index=False)
    base = [sys.executable, "main.py", "--stations", str(d / "amb.csv"), "--station-season", "JJAS", "--station-output", str(d / "out")]
    r = subprocess.run(base, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "ambiguous" in r.stderr and "Traceback" not in r.stderr, r.stderr[-500:]
    r = subprocess.run(base + ["--station-dayfirst"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
    cov = pd.read_csv(d / "out" / "station_coverage.csv")
    assert list(cov["station"]) == ["A1"] and cov.loc[0, "first"].startswith("1995-01-01")
    assert (d / "out" / "station_data_quality_report.json").exists() or (d / "out" / "station_metadata.csv").exists()


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
