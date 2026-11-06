"""GeoTIFF writer (requires rasterio).

Origin: climate_maps.py."""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np


log = logging.getLogger("clim.maps")


def write_geotiff(field2d, lat, lon, outfile, nodata=-9999.0):
    """Single-band GeoTIFF (EPSG:4326, north-up).  Skipped with a warning if rasterio is missing."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception:
        log.warning("rasterio not installed: GeoTIFF %s not written", Path(outfile).name)
        return None
    lat = np.asarray(lat); lon = np.asarray(lon)
    arr = np.asarray(field2d, dtype="float32")
    if lat.size > 1 and lat[0] < lat[-1]:           # rasters are north-up
        arr, lat = arr[::-1], lat[::-1]
    dlat = abs(float(np.median(np.diff(lat)))) if lat.size > 1 else 1.0
    dlon = abs(float(np.median(np.diff(lon)))) if lon.size > 1 else 1.0
    tr = from_origin(float(lon.min()) - dlon / 2, float(lat.max()) + dlat / 2, dlon, dlat)
    arr = np.where(np.isfinite(arr), arr, nodata)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(outfile, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr, nodata=nodata) as dst:
        dst.write(arr, 1)
    return Path(outfile)
