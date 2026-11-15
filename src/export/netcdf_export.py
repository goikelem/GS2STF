"""NetCDF writers for gridded fields.

Origin: climate_workflow.py, teleconnection.py."""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np


log = logging.getLogger("clim.workflow")


def make_dataset(arr, name, like, attrs, dtype="float32"):
    import xarray as xr
    ds = xr.Dataset({name: (("time", "lat", "lon"), arr.astype(dtype))},
                    coords={"time": like.indexes["time"], "lat": like["lat"].values, "lon": like["lon"].values})
    ds[name].attrs.update(attrs)
    ds["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    ds["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    return ds


def make_map_dataset(fields: dict, like, attrs=None):
    import xarray as xr
    ds = xr.Dataset({n: (("lat", "lon"), v.astype("float32")) for n, v in fields.items()},
                    coords={"lat": like["lat"].values, "lon": like["lon"].values})
    ds.attrs.update(attrs or {})
    return ds


def save_netcdf(ds, path: Path, int_var: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": 4}
        if int_var and v == int_var:
            enc[v]["_FillValue"] = 0
    ds.to_netcdf(path, encoding=enc)
    log.info("Saved %s", path)


def write_nc(fields: dict, coords: dict, path: Path, attrs=None):
    import xarray as xr
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {n: (dims, np.asarray(v, dtype="float32")) for n, (dims, v) in fields.items()}
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(attrs or {})
    ds.to_netcdf(path, encoding={n: {"zlib": True, "complevel": 4} for n in data})
    log.info("Saved %s", path)


def write_forecast_netcdf(path, result: dict, variable: str, units: str = "", methods=(), fill: float = -9999.0):
    """Write a gridded forecast (output of forecasting.seasonal.run_grid_forecast) as CF-style NetCDF-3 classic.

    Uses scipy only (no xarray / netCDF4 needed); the file opens in xarray, QGIS, CDO, NCL and Panoply.
    Fields: forecast, anomaly, climatology, prob_below / prob_normal / prob_above, dominant_category (0 below,
    1 normal, 2 above, -1 none) and the hindcast skill maps (rpss, correlation, rmse_skill).  Initialization date,
    lead time and target period are stored as global attributes.
    """
    from pathlib import Path
    from scipy.io import netcdf_file
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lat, lon = np.asarray(result["lat"], float), np.asarray(result["lon"], float)
    f = netcdf_file(str(path), "w", version=2)
    try:
        f.createDimension("lat", len(lat)); f.createDimension("lon", len(lon))
        v = f.createVariable("lat", "d", ("lat",)); v[:] = lat
        v.units = "degrees_north"; v.standard_name = "latitude"; v.axis = "Y"
        v = f.createVariable("lon", "d", ("lon",)); v[:] = lon
        v.units = "degrees_east"; v.standard_name = "longitude"; v.axis = "X"

        def put(name, arr, long_name, vunits="1", dtype="f"):
            v = f.createVariable(name, dtype, ("lat", "lon"))
            a = np.asarray(arr, dtype=float)
            v[:] = np.where(np.isfinite(a), a, fill).astype("f4" if dtype == "f" else "i4")
            v.long_name = long_name; v.units = vunits
            v._FillValue = np.float32(fill) if dtype == "f" else np.int32(fill)

        put("forecast", result["forecast"], f"forecast {variable}", units)
        put("anomaly", result["anomaly"], f"forecast anomaly of {variable} relative to training-year mean", units)
        put("climatology", result["climatology"], f"training-year mean of {variable}", units)
        for k, nm in enumerate(("below", "normal", "above")):
            put(f"prob_{nm}", result["probabilities"][k], f"probability of {nm}-normal {variable} (tercile category)")
        put("dominant_category", np.where(result["dominant_category"] >= 0, result["dominant_category"], np.nan),
            "most probable tercile category: 0 below, 1 normal, 2 above", dtype="i")
        for k, lab in (("rpss", "ranked probability skill score vs climatology (hindcast)"),
                       ("correlation", "hindcast correlation of forecast and observation"),
                       ("rmse_skill", "hindcast RMSE skill vs climatology")):
            if k in result.get("skill", {}):
                put(f"hindcast_{k}", result["skill"][k], lab)
        meta = result["meta"]
        f.Conventions = "CF-1.8"
        f.title = f"Seasonal {variable} forecast with tercile probabilities"
        f.forecast_period = meta["forecast_period"]; f.init_date = meta["init_date"]
        f.lead_months = int(meta["lead_months"]); f.target_start = meta["target_start"]; f.target_end = meta["target_end"]
        f.variable = variable; f.methods = ",".join(methods) if methods else "unspecified"
        f.n_training_years = int(result.get("n_train_years", 0))
        f.note = "Terciles and models use training years only; probabilities from a Gaussian predictive distribution."
    finally:
        f.close()
    return path
