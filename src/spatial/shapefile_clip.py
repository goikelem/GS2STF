"""Clipping of forecast grids to a bounding box, named region, or polygon (GeoJSON or ESRI Shapefile).

Polygons are lists of rings (lon, lat); the first ring is the exterior and the others are holes (even-odd rule is used,
so holes and multi-part shapes work).  Shapefiles are read by a small built-in reader for Polygon / PolygonZ / PolygonM
shapes; it needs WGS84 longitude/latitude coordinates (a projected .prj is refused rather than guessed).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from src import settings


# ---------------------------------------------------------------- bounding boxes
def region_bbox(name_or_box) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) from a region name in config/regions.yaml or from a 4-sequence."""
    if isinstance(name_or_box, str):
        if name_or_box not in settings.STUDY_REGIONS:
            raise KeyError(f"Unknown region '{name_or_box}'. Configured: {list(settings.STUDY_REGIONS)}")
        r = settings.STUDY_REGIONS[name_or_box]
        return (*r["lon"], *r["lat"])
    lon0, lon1, lat0, lat1 = map(float, name_or_box)
    if lon0 >= lon1 or lat0 >= lat1:
        raise ValueError("bbox must be (lon_min, lon_max, lat_min, lat_max) with min < max")
    return lon0, lon1, lat0, lat1


def _lon_like(lon_pts, lon_axis):
    """Express longitudes in the grid's convention (0..360 or -180..180)."""
    lon_pts = np.asarray(lon_pts, dtype=float)
    return np.mod(lon_pts, 360.0) if np.nanmax(lon_axis) > 180 else np.where(lon_pts > 180, lon_pts - 360.0, lon_pts)


def bbox_mask(lat, lon, bbox) -> np.ndarray:
    """Boolean (lat, lon) mask of cell centres inside the box (inclusive)."""
    lon0, lon1, lat0, lat1 = region_bbox(bbox)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    la = (lat >= lat0) & (lat <= lat1)
    lo_b = _lon_like([lon0, lon1], lon)
    lo = (lon >= lo_b[0]) & (lon <= lo_b[1]) if lo_b[0] <= lo_b[1] else (lon >= lo_b[0]) | (lon <= lo_b[1])
    return la[:, None] & lo[None, :]


# ---------------------------------------------------------------- polygons
def read_polygons(path) -> list[list[np.ndarray]]:
    """Read polygons from .geojson/.json or .shp.  Returns a list of polygons, each a list of (n, 2) lon/lat rings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext in (".geojson", ".json"):
        return _read_geojson(path)
    if ext == ".shp":
        return _read_shp(path)
    raise ValueError(f"Unsupported polygon file '{path.name}' (use .geojson, .json or .shp)")


def _geom_polys(g):
    t = g["type"]
    if t == "Polygon":
        return [[np.asarray(r, float)[:, :2] for r in g["coordinates"]]]
    if t == "MultiPolygon":
        return [[np.asarray(r, float)[:, :2] for r in poly] for poly in g["coordinates"]]
    if t == "GeometryCollection":
        return [p for gg in g["geometries"] for p in _geom_polys(gg)]
    return []


def _read_geojson(path):
    d = json.loads(path.read_text())
    crs = (d.get("crs") or {}).get("properties", {}).get("name", "")
    if crs and not any(k in crs for k in ("4326", "CRS84")):
        raise ValueError(f"GeoJSON CRS '{crs}' is not WGS84 lon/lat; reproject first")
    geoms = ([f["geometry"] for f in d["features"]] if d["type"] == "FeatureCollection"
             else [d["geometry"]] if d["type"] == "Feature" else [d])
    polys = [p for g in geoms if g for p in _geom_polys(g)]
    if not polys:
        raise ValueError(f"No polygon geometry found in {path.name}")
    return polys


def _read_shp(path):
    prj = path.with_suffix(".prj")
    if prj.exists() and "PROJCS" in prj.read_text(errors="ignore").upper():
        raise ValueError(f"{prj.name} describes a projected CRS; reproject the shapefile to WGS84 lon/lat first")
    b = path.read_bytes()
    if struct.unpack(">i", b[:4])[0] != 9994:
        raise ValueError(f"{path.name} is not a shapefile")
    pos, polys = 100, []
    while pos + 8 <= len(b):
        _, words = struct.unpack(">ii", b[pos:pos + 8])
        rec = b[pos + 8: pos + 8 + 2 * words]; pos += 8 + 2 * words
        stype = struct.unpack("<i", rec[:4])[0]
        if stype == 0:
            continue
        if stype not in (5, 15, 25):
            raise ValueError(f"Shape type {stype} not supported; need polygons")
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npts]
        off = 44 + 4 * nparts
        xy = np.frombuffer(rec[off: off + 16 * npts], dtype="<f8").reshape(npts, 2)
        polys.append([xy[parts[i]:parts[i + 1]].copy() for i in range(nparts)])      # even-odd handles holes
    if not polys:
        raise ValueError(f"No polygons in {path.name}")
    return polys


def polygon_mask(lat, lon, polygons) -> np.ndarray:
    """Boolean (lat, lon) mask of cell centres inside any polygon (even-odd rule; holes excluded)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    LON, LAT = np.meshgrid(lon, lat)
    out = np.zeros(LAT.shape, dtype=bool)
    for rings in polygons:
        allpts = np.vstack(rings)
        lo = _lon_like(allpts[:, 0], lon)
        xmin, xmax, ymin, ymax = lo.min(), lo.max(), allpts[:, 1].min(), allpts[:, 1].max()
        cand = (LAT >= ymin) & (LAT <= ymax) & (LON >= xmin) & (LON <= xmax)
        if not cand.any():
            continue
        px, py = LON[cand], LAT[cand]
        inside = np.zeros(px.shape, dtype=bool)
        for ring in rings:
            x = _lon_like(ring[:, 0], lon); y = ring[:, 1]
            x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
            for xa, ya, xb, yb in zip(x1, y1, x2, y2):
                if ya == yb:
                    continue
                crosses = ((ya > py) != (yb > py)) & (px < (xb - xa) * (py - ya) / (yb - ya) + xa)
                inside ^= crosses
            if not np.allclose(ring[0], ring[-1]):                      # close an open ring
                xa, ya, xb, yb = x[-1], y[-1], x[0], y[0]
                if ya != yb:
                    inside ^= ((ya > py) != (yb > py)) & (px < (xb - xa) * (py - ya) / (yb - ya) + xa)
        out[cand] |= inside
    return out


def region_mask(lat, lon, bbox=None, polygons=None) -> np.ndarray:
    """Combine a bounding box and/or polygons (both given -> intersection); all-True if neither is given."""
    m = np.ones((len(lat), len(lon)), dtype=bool)
    if bbox is not None:
        m &= bbox_mask(lat, lon, bbox)
    if polygons is not None:
        m &= polygon_mask(lat, lon, polygons)
    return m


def crop_to_mask(arrays: dict, lat, lon, mask: np.ndarray):
    """Crop (..., lat, lon) arrays to the mask's extent and set cells outside the mask to NaN.

    Returns (cropped arrays dict, lat, lon, cropped mask).  Raises if the mask selects no cell.
    """
    if not mask.any():
        raise ValueError("The region selects no grid cell (check the bounding box / polygon and the grid's longitude convention)")
    ri, ci = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    r0, r1, c0, c1 = ri.min(), ri.max() + 1, ci.min(), ci.max() + 1
    m = mask[r0:r1, c0:c1]
    out = {}
    for k, a in arrays.items():
        a = np.asarray(a)
        sub = a[..., r0:r1, c0:c1]
        out[k] = np.where(m, sub, np.nan) if np.issubdtype(sub.dtype, np.floating) else sub
    return out, np.asarray(lat)[r0:r1], np.asarray(lon)[c0:c1], m
