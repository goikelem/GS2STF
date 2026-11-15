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


def write_forecast_geotiff(path, bands: dict, lat, lon, nodata: float = -9999.0):
    """Write several 2-D fields as one multi-band GeoTIFF (EPSG:4326) with band names; needs only ``tifffile``.

    ``bands`` maps band name -> (lat, lon) array.  Rows are written north-to-south regardless of the input latitude order.
    Band names go into the GDAL metadata tag, so QGIS / GDAL show them.  Cells are assumed regular and cell-centred.
    """
    import numpy as np
    from pathlib import Path
    import tifffile
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("need at least 2 latitudes and 2 longitudes")
    dy, dx = abs(float(np.median(np.diff(lat)))), abs(float(np.median(np.diff(lon))))
    names = list(bands)
    arr = np.stack([np.asarray(bands[k], dtype="f4") for k in names])
    if lat[0] < lat[-1]:                                       # GeoTIFF rows run north -> south
        arr = arr[:, ::-1, :]
    if lon[0] > lon[-1]:
        arr = arr[:, :, ::-1]
    arr = np.where(np.isfinite(arr), arr, np.float32(nodata)).astype("f4")
    west, north = float(lon.min()) - dx / 2, float(lat.max()) + dy / 2
    keys = (1, 1, 0, 3,  1024, 0, 1, 2,  1025, 0, 1, 1,  2048, 0, 1, 4326)      # geographic, PixelIsArea, WGS84
    gdal_md = "<GDALMetadata>" + "".join(
        f'<Item name="DESCRIPTION" sample="{i}" role="description">{n}</Item>' for i, n in enumerate(names)) + "</GDALMetadata>"
    extratags = [(33550, "d", 3, (dx, dy, 0.0), True), (33922, "d", 6, (0.0, 0.0, 0.0, west, north, 0.0), True),
                 (34735, "H", len(keys), keys, True), (42112, "s", 0, gdal_md, True), (42113, "s", 0, str(nodata), True)]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr, photometric="minisblack", planarconfig="separate", extratags=extratags)
    return path
