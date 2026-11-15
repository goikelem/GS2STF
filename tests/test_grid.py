"""Tests for the gridded pathway: aggregation, clipping, grid forecast, NetCDF/GeoTIFF export, maps, CLI workflow."""
import json
import os
import struct
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_seasonal import SPEC, YEARS, make_indices  # noqa: E402

from src.export import geotiff_export, netcdf_export  # noqa: E402
from src.forecasting import seasonal as S  # noqa: E402
from src.preprocessing import grid_processing as G  # noqa: E402
from src.spatial import forecast_maps, shapefile_clip as C, station_maps  # noqa: E402

LAT = np.array([12.0, 11.0, 10.0, 9.0, 8.0, 7.0])          # descending, as many products are
LON = np.array([34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0])


def tmp():
    return Path(tempfile.mkdtemp())


def make_grid(signal=True, seed=3):
    ind = make_indices(); X = S.build_predictors(ind, SPEC, YEARS)
    rng = np.random.default_rng(seed)
    t = pd.date_range("1981-01-01", "2025-12-01", freq="MS")
    r34 = (X["R34"] / X["R34"].std()).reindex(t.year).values
    base = 50 + 5 * np.arange(len(LAT))[:, None] + 2 * np.arange(len(LON))[None, :]
    data = np.empty((len(t), len(LAT), len(LON)))
    for i, ts in enumerate(t):
        jjas = ts.month in (6, 7, 8, 9)
        amp = (-0.35 * r34[i] if signal and jjas else 0.0)
        data[i] = base * (1 + amp) + rng.normal(0, 4, base.shape)
    return G.MonthlyGrid(np.clip(data, 0, None), t, LAT, LON, "prcp", "mm/month"), ind


def test_seasonal_field_labels_and_completeness():
    t = pd.date_range("2000-01-01", "2003-06-01", freq="MS")
    data = np.ones((len(t), 2, 2)); data[:, 0, 0] = np.arange(len(t))
    g = G.MonthlyGrid(data.copy(), t, np.array([1., 2.]), np.array([3., 4.]), "prcp")
    yrs, f = G.seasonal_field(g, "JJAS")
    assert yrs == [2000, 2001, 2002] and np.allclose(f[:, 1, 1], 4.0)            # 2003 stops in June -> incomplete; sum of 4 months of 1.0
    yrs, f = G.seasonal_field(g, "ONDJ", "start")
    assert yrs == [2000, 2001, 2002] and np.allclose(f[0, 1, 1], 4.0)            # Oct 2000 - Jan 2001
    yrs, _ = G.seasonal_field(g, "ONDJ", "end")
    assert yrs == [2001, 2002, 2003]
    g.data[5, 1, 1] = np.nan                                                    # June 2000 missing in one cell
    yrs, f = G.seasonal_field(g, "JJAS")
    assert np.isnan(f[0, 1, 1]) and not np.isnan(f[0, 0, 0]) and not np.isnan(f[1, 1, 1])
    gt = G.MonthlyGrid(np.full((len(t), 1, 1), 20.0), t, np.array([1.]), np.array([3.]), "tmax")
    assert abs(G.seasonal_field(gt, "JJAS")[1][0, 0, 0] - 20.0) < 1e-9          # temperature is averaged


def test_bbox_and_polygon_masks():
    m = C.bbox_mask(LAT, LON, (35.0, 37.0, 8.0, 10.0))
    assert m.sum() == 3 * 3 and m[LAT == 9.0][0][LON == 36.0][0]
    lon360 = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
    assert C.bbox_mask(np.array([0.0]), lon360, (-8.0, 6.0, -1.0, 1.0)).sum() == 3     # wraps the date line / 0 meridian
    sq = [[np.array([[34.5, 7.5], [38.5, 7.5], [38.5, 11.5], [34.5, 11.5], [34.5, 7.5]]),
           np.array([[36.5, 9.5], [37.5, 9.5], [37.5, 10.5], [36.5, 10.5], [36.5, 9.5]])]]   # exterior + hole
    pm = C.polygon_mask(LAT, LON, sq)
    inside = {(la, lo) for la in LAT for lo in LON if 7.5 < la < 11.5 and 34.5 < lo < 38.5 and not (9.5 < la < 10.5 and 36.5 < lo < 37.5)}
    got = {(la, lo) for i, la in enumerate(LAT) for j, lo in enumerate(LON) if pm[i, j]}
    assert got == inside and (10.0, 37.0) not in got
    cr, la2, lo2, m2 = C.crop_to_mask({"a": np.arange(42.0).reshape(6, 7)}, LAT, LON, pm)
    assert cr["a"].shape == m2.shape == (len(la2), len(lo2)) and np.isnan(cr["a"][~m2]).all()
    try:
        C.crop_to_mask({"a": np.zeros((6, 7))}, LAT, LON, C.bbox_mask(LAT, LON, (100.0, 101.0, 0.0, 1.0))); raise AssertionError
    except ValueError:
        pass
    assert C.region_bbox("Tigray") == (36.0, 41.0, 12.5, 15.0)


def _write_shp(path, rings):
    pts = np.vstack(rings); parts = [0]
    for r in rings[:-1]:
        parts.append(parts[-1] + len(r))
    content = struct.pack("<i", 5) + struct.pack("<4d", pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    content += struct.pack("<ii", len(rings), len(pts)) + struct.pack(f"<{len(parts)}i", *parts) + pts.astype("<f8").tobytes()
    rec = struct.pack(">ii", 1, len(content) // 2) + content
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(rec)) // 2) + struct.pack("<2i", 1000, 5) + struct.pack("<8d", *([0.0] * 8))
    Path(path).write_bytes(header + rec)


def test_geojson_and_shapefile_readers():
    d = tmp()
    outer = [[34.5, 7.5], [38.5, 7.5], [38.5, 11.5], [34.5, 11.5], [34.5, 7.5]]
    hole = [[36.5, 9.5], [36.5, 10.5], [37.5, 10.5], [37.5, 9.5], [36.5, 9.5]]
    (d / "r.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [outer, hole]}}]}))
    pg = C.polygon_mask(LAT, LON, C.read_polygons(d / "r.geojson"))
    _write_shp(d / "r.shp", [np.array(outer), np.array(hole)])
    ps = C.polygon_mask(LAT, LON, C.read_polygons(d / "r.shp"))
    assert (pg == ps).all() and pg.sum() == 4 * 4 - 1                                # 16 cells in the outer ring minus the hole cell
    assert not pg[LAT == 10.0][0][LON == 37.0][0]                                  # the hole is excluded
    (d / "r.prj").write_text('PROJCS["UTM 37N"]')
    try:
        C.read_polygons(d / "r.shp"); raise AssertionError
    except ValueError as e:
        assert "projected" in str(e)
    try:
        C.read_polygons(d / "x.txt"); raise AssertionError
    except (ValueError, FileNotFoundError):
        pass


def test_grid_forecast_skill_and_mask():
    g, ind = make_grid()
    yrs, field = G.seasonal_field(g, "JJAS")
    mask = np.ones(field.shape[1:], bool); mask[0, 0] = False; field[:, 5, 6] = np.nan      # masked + all-NaN cell
    r = S.run_grid_forecast(field, yrs, LAT, LON, ind, SPEC, 2026, methods=("ridge", "pcr", "cca"), transform="sqrt",
                            mask=mask)
    assert r["forecast"].shape == (6, 7) and r["probabilities"].shape == (3, 6, 7)
    assert np.isnan(r["forecast"][0, 0]) and np.isnan(r["forecast"][5, 6]) and r["n_cells"] == 40
    ok = ~np.isnan(r["probabilities"][0])
    assert np.allclose(r["probabilities"].sum(axis=0)[ok], 1) and (r["dominant_category"][~ok] == -1).all()
    assert np.nanmean(r["skill"]["rpss"]) > 0.1 and np.nanmean(r["skill"]["correlation"]) > 0.4
    assert r["meta"]["forecast_period"] == "JJAS_2026" and r["meta"]["lead_months"] == 1
    gn, _ = make_grid(signal=False)
    rn = S.run_grid_forecast(G.seasonal_field(gn, "JJAS")[1], yrs, LAT, LON, ind, SPEC, 2026, methods=("ridge",), transform="sqrt")
    assert np.nanmean(rn["skill"]["rpss"]) < 0.05                                   # no skill invented without a signal


def test_netcdf_and_geotiff_roundtrip():
    from scipy.io import netcdf_file
    import tifffile
    g, ind = make_grid()
    yrs, field = G.seasonal_field(g, "JJAS"); field[:, 2, 3] = np.nan
    r = S.run_grid_forecast(field, yrs, LAT, LON, ind, SPEC, 2026, methods=("ridge",), transform="sqrt")
    d = tmp()
    nc = netcdf_export.write_forecast_netcdf(d / "f.nc", r, "precipitation", "mm", ["ridge"])
    f = netcdf_file(str(nc), "r", mmap=False)
    assert f.dimensions["lat"] == 6 and f.dimensions["lon"] == 7 and f.init_date.decode() == "2026-05-31"
    assert f.forecast_period.decode() == "JJAS_2026" and int(f.lead_months) == 1 and f.target_end.decode() == "2026-09-30"
    pb = f.variables["prob_below"][:].copy()
    assert pb.shape == (6, 7) and pb[2, 3] == -9999.0 and abs(pb[0, 0] - r["probabilities"][0][0, 0]) < 1e-5
    assert np.allclose(f.variables["lat"][:], LAT) and "hindcast_rpss" in f.variables
    f.close()
    bands = {"forecast": r["forecast"], "prob_above": r["probabilities"][2]}
    tif = geotiff_export.write_forecast_geotiff(d / "f.tif", bands, np.sort(LAT), LON)        # ascending lat input
    with tifffile.TiffFile(str(tif)) as t:
        a = t.asarray(); p = t.pages[0]
        scale = p.tags[33550].value; tie = p.tags[33922].value; keys = p.tags[34735].value
        assert a.shape == (2, 6, 7) and abs(scale[0] - 1.0) < 1e-9 and abs(scale[1] - 1.0) < 1e-9
        assert abs(tie[3] - 33.5) < 1e-9 and abs(tie[4] - 12.5) < 1e-9 and 4326 in keys
        assert "forecast" in p.tags[42112].value and "prob_above" in p.tags[42112].value
        assert float(p.tags[42113].value) == -9999.0
    flipped = geotiff_export.write_forecast_geotiff(d / "g.tif", {"x": np.arange(42.0).reshape(6, 7)}, LAT, LON)  # descending already
    with tifffile.TiffFile(str(flipped)) as t:
        assert t.asarray()[0, 0, 0] == 0.0                                          # first row = northernmost
    asc = geotiff_export.write_forecast_geotiff(d / "h.tif", {"x": np.arange(42.0).reshape(6, 7)}, LAT[::-1], LON)
    with tifffile.TiffFile(str(asc)) as t:
        assert t.asarray()[0, 0, 0] == 35.0                                         # ascending input is flipped north-first


def test_maps_png():
    g, ind = make_grid()
    yrs, field = G.seasonal_field(g, "JJAS")
    r = S.run_grid_forecast(field, yrs, LAT, LON, ind, SPEC, 2026, methods=("ridge",), transform="sqrt")
    d = tmp()
    paths = forecast_maps.plot_forecast_maps(r, d, "precipitation", "mm", polygons=[[np.array([[34.5, 7.5], [38.5, 7.5], [38.5, 11.5], [34.5, 7.5]])]])
    assert len(paths) == 3 and all(p.stat().st_size > 5000 for p in paths)
    ft = r["forecast_table"].copy(); ft["Station"] = ["S%d" % i for i in range(len(ft))]
    ft["lat"] = np.linspace(7, 12, len(ft)); ft["lon"] = np.linspace(34, 40, len(ft)); ft["Variable"] = "precipitation"
    sp = station_maps.plot_station_forecasts(ft, d)
    assert len(sp) == 2 and all(p.stat().st_size > 5000 for p in sp)


def test_coarsen_array():
    from src.preprocessing import regridding as R
    a = np.arange(24.0).reshape(1, 4, 6); a[0, 0, 0] = np.nan
    c, la, lo = R.coarsen_array(a, np.array([4., 3., 2., 1.]), np.arange(6.0), 2)
    assert c.shape == (1, 2, 3) and np.allclose(la, [3.5, 1.5]) and np.allclose(lo, [0.5, 2.5, 4.5])
    assert abs(c[0, 0, 0] - np.nanmean([1, 6, 7])) < 1e-12                    # NaN skipped; 3 of 4 cells valid
    b = a.copy(); b[0, :2, :2] = np.nan; b[0, 0, 0] = np.nan; b[0, 1, 1] = np.nan; b[0, 1, 0] = np.nan
    assert np.isnan(R.coarsen_array(b, np.arange(4.), np.arange(6.), 2)[0][0, 0, 0])   # < 50% valid -> NaN
    assert R.coarsen_array(a, np.arange(4.), np.arange(6.), 4)[0].shape == (1, 1, 1)
    try:
        R.coarsen_array(a, np.arange(4.), np.arange(6.), 9); raise AssertionError
    except ValueError:
        pass


def test_cli_workflow_end_to_end():
    g, ind = make_grid()
    d = tmp(); ind.reset_index(names="Date").to_csv(d / "idx.csv", index=False)
    args = Namespace(grid_variable="precipitation", target_year=2026, init_month=5, grid_nc=None, sst_indices=d / "idx.csv",
                     grid_output=d / "out", season="JJAS", predictor_months=3, grid_region=None, bbox=[34.5, 39.5, 7.5, 11.5],
                     clip_file=None, grid_coarsen=1, methods="ridge,pcr", hindcast_scheme="rolling", ensemble_mode="skill", max_predictors=2)
    res = G.run(args, grid=g)
    names = sorted(p.name for p in (d / "out").iterdir())
    assert any(n.endswith(".nc") for n in names) and any(n.endswith(".tif") for n in names)
    assert sum(n.endswith(".png") for n in names) == 3 and any(n.startswith("method_comparison") for n in names)
    meta = json.loads(next((d / "out").glob("*_metadata.json")).read_text())
    assert meta["forecast_period"] == "JJAS_2026" and meta["region"] == [34.5, 39.5, 7.5, 11.5] and meta["n_cells"] == res["n_cells"]
    assert res["forecast"].shape == (4, 5)                                           # lat 8-11, lon 35-39 after cropping to the box
    args.grid_coarsen, args.bbox, args.grid_output, args.methods = 2, None, d / "out2", "ridge"
    assert G.run(args, grid=g)["forecast"].shape == (3, 3)                           # 6 x 7 grid -> 3 x 3 blocks (edge trimmed)


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
