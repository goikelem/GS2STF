"""Unit tests for the station pathway (python tests/test_station.py  or  pytest tests)."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_ingestion import station_csv_loader as L  # noqa: E402
from src.preprocessing import station_processing as P  # noqa: E402


def _write(df, name="x.csv"):
    p = os.path.join(tempfile.mkdtemp(), name); df.to_csv(p, index=False); return p


def _daily(stn="S1", lat=10.4, lon=35.0, start="1991-01-01", end="2020-12-31", var="precipitation", val=1.0):
    d = pd.date_range(start, end)
    return pd.DataFrame({"Station": stn, "Date": d.strftime("%Y-%m-%d"), "Lat": lat, "Lon": lon,
                         "Variable": var, "Value": val})


def test_long_and_simple_format_and_cleaning():
    long = pd.concat([_daily(end="1991-01-31"), _daily(var="tmax", val=29.5, end="1991-01-31")], ignore_index=True)
    long.loc[0, "Value"] = -999            # sentinel
    long.loc[1, "Value"] = -2              # negative rain
    long = pd.concat([long, long.iloc[[5]]], ignore_index=True)  # duplicate record
    d = L.load_station_csv(_write(long))
    assert set(d.observations.variable) == {"precipitation", "tmax"}
    assert d.report["sentinel_values_set_to_nan"] == 1 and d.report["negative_precipitation_set_to_nan"] == 1
    assert d.report["duplicate_records_dropped"] == 1
    simple = _daily(end="1991-01-10").rename(columns={"Lat": "lat", "Lon": "long", "Value": "value"}).drop(columns="Variable")
    d2 = L.load_station_csv(_write(simple))
    assert d2.observations.variable.unique().tolist() == ["precipitation"] and d2.report["assumptions"]
    d3 = L.load_station_csv(_write(simple), variable="rain")          # alias
    assert d3.observations.variable.iloc[0] == "precipitation"


def test_column_map_and_missing_columns():
    df = _daily(end="1991-01-05").rename(columns={"Station": "ID", "Date": "DAY", "Lat": "Y", "Lon": "X", "Value": "RR"})
    d = L.load_station_csv(_write(df.drop(columns="Variable")), column_map={"station": "ID", "date": "DAY", "lat": "Y", "lon": "X", "value": "RR"})
    assert len(d.observations) == 5
    try:
        L.load_station_csv(_write(df.drop(columns=["Variable", "RR"])))
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_separate_files_and_coordinate_conflict():
    r = _daily(end="1991-01-10"); t = _daily(var="tmax", val=30.0, end="1991-01-10")
    t.loc[3, "Lat"] = 12.0
    d = L.load_station_files({"precipitation": _write(r.drop(columns="Variable")), "tmax": _write(t.drop(columns="Variable"))})
    assert d.report["variables"] == ["precipitation", "tmax"] and d.stations.shape[0] == 1


def test_monthly_completeness_and_sum_vs_mean():
    d = L.load_station_csv(_write(pd.concat([_daily(end="1991-03-31", val=2.0), _daily(var="tmax", end="1991-03-31", val=30.0)])))
    o = d.observations
    o = o[~((o.variable == "precipitation") & (o.date == "1991-02-10"))]       # one missing rain day in February
    m = P.monthly_aggregate(o).set_index(["variable", "month"])["value"]
    assert m[("precipitation", pd.Timestamp("1991-01-01"))] == 62.0              # 31 days x 2 mm
    assert np.isnan(m[("precipitation", pd.Timestamp("1991-02-01"))])             # incomplete -> NaN (precip needs 100%)
    assert abs(m[("tmax", pd.Timestamp("1991-02-01"))] - 30.0) < 1e-9


def test_cross_year_season_labels():
    d = L.load_station_csv(_write(_daily(start="1991-01-01", end="1993-12-31", val=1.0)))
    mo = P.monthly_aggregate(d.observations)
    s_start = P.seasonal_aggregate(mo, "ONDJ", "start").set_index("season_year")["value"]
    assert list(s_start.index) == [1991, 1992] and abs(s_start[1991] - (31 + 30 + 31 + 31)) < 1e-9
    s_end = P.seasonal_aggregate(mo, "ONDJ", "end").set_index("season_year")["value"]
    assert list(s_end.index) == [1992, 1993]
    jjas = P.seasonal_aggregate(mo, "JJAS")
    assert list(jjas.season_year) == [1991, 1992, 1993] and (jjas.value == 30 + 31 + 31 + 30).all()
    assert len(P.seasonal_aggregate(mo, [6, 7, 8, 9, 10], "start")) == 3           # user-defined month list


def test_anomalies_reference_period():
    rng = np.random.default_rng(0)
    yrs = np.arange(1981, 2021)
    s = pd.DataFrame({"station": "S1", "lat": 1.0, "lon": 2.0, "variable": "precipitation", "season": "JJAS",
                      "season_year": yrs, "value": 400 + 50 * rng.normal(size=len(yrs))})
    a = P.seasonal_anomalies(s, 1991, 2020)
    ref = s[s.season_year >= 1991].value.mean()
    assert np.allclose(a.climatology, ref)
    assert np.allclose(a.anomaly, s.value - ref)
    assert np.allclose(a.percent_of_normal, 100 * s.value / ref)
    short = P.seasonal_anomalies(s[s.season_year >= 2010], 1991, 2020)           # only 11 reference years < 15
    assert short.anomaly.isna().all()


def test_grid_extraction_nearest_bilinear_and_outside():
    lat = np.array([14.0, 13.0, 12.0]); lon = np.array([36.0, 37.0, 38.0])     # descending lat
    t = pd.date_range("2000-01-01", periods=2, freq="MS")
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    f = np.stack([LAT * 10 + LON, LAT * 10 + LON + 1])
    st = pd.DataFrame({"station": ["A", "B", "C"], "lat": [13.0, 12.5, 30.0], "lon": [37.0, 37.5, 37.0]})
    v, c = P.extract_grid_predictors(st, f, lat, lon, t, "nearest")
    assert v.loc[t[0], "A"] == 13 * 10 + 37 and v.loc[t[1], "A"] == 13 * 10 + 38
    assert np.isnan(v["C"]).all() and not c.set_index("station").loc["C", "inside_grid"]
    vb, _ = P.extract_grid_predictors(st, f, lat, lon, t, "bilinear")
    assert abs(vb.loc[t[0], "B"] - (12.5 * 10 + 37.5)) < 1e-9                      # linear field is reproduced exactly
    lon360 = np.array([36.0, 37.0, 38.0]); st2 = st.copy(); st2.loc[0, "lon"] = 37.0
    assert P.extract_grid_predictors(st2, f, lat, lon360 - 360 + 360, t)[0].loc[t[0], "A"] == 167


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
