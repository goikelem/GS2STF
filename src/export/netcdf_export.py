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
