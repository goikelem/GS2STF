"""NetCDF loading: dataset inspection / coordinate detection, gridded precipitation + temperature loading, SST loading.

Origin: data_processing.py (inspection), climate_data.py (rain/temperature), teleconnection.py (SST)."""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from src import settings
from src.data_ingestion.data_validation import full_monthly, same_grid
from src.preprocessing.regridding import block_factor


log = logging.getLogger("sst.data")


SST_NAME_HINTS = ["analysed_sst", "sst", "tos", "sea_surface_temperature", "thetao", "temperature", "to"]


TIME_HINTS = ["time", "valid_time", "t", "date"]


LAT_HINTS = ["latitude", "lat", "y", "nav_lat"]


LON_HINTS = ["longitude", "lon", "x", "nav_lon"]


def _find_coord(ds, hints, units_hint=None, standard_names=(), axis=None):
    names = list(ds.coords) + list(ds.dims)
    for n in names:
        a = ds[n].attrs
        if a.get("standard_name", "") in standard_names:
            return n
        if axis and a.get("axis", "").upper() == axis:
            return n
        if units_hint and str(a.get("units", "")).lower() in units_hint:
            return n
    for h in hints:
        for n in names:
            if n.lower() == h:
                return n
    raise KeyError(f"Could not identify coordinate (hints={hints}); available: {names}")


def _find_sst_var(ds, lat, lon, time):
    cands = [v for v in ds.data_vars if {lat, lon}.issubset(ds[v].dims)]
    if not cands:
        raise KeyError(f"No data variable spans lat/lon. Variables: {list(ds.data_vars)}")
    for hint in SST_NAME_HINTS:
        for v in cands:
            if v.lower() == hint:
                return v
    for v in cands:
        sn = ds[v].attrs.get("standard_name", "") + ds[v].attrs.get("long_name", "")
        if "temperature" in sn.lower() or "sst" in sn.lower():
            return v
    return cands[0]


def open_dataset(path: Path):
    import xarray as xr
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input NetCDF not found: {path}\nEdit sst_workflow.input_nc in config/datasets.yaml or pass --input to main.py.")
    try:
        ds = xr.open_dataset(path, decode_times=True, use_cftime=False, mask_and_scale=True)
    except Exception:  # non-standard calendars
        log.warning("Standard time decoding failed; retrying with cftime.")
        ds = xr.open_dataset(path, decode_times=True, use_cftime=True, mask_and_scale=True)
    return ds


def inspect_dataset(ds) -> dict:
    """Identify variable/coordinates and summarise the file; returns a metadata dict."""
    time = _find_coord(ds, TIME_HINTS, standard_names=("time",), axis="T")
    lat = _find_coord(ds, LAT_HINTS, units_hint=("degrees_north", "degree_north", "degree_n", "degrees_n"),
                      standard_names=("latitude",), axis="Y")
    lon = _find_coord(ds, LON_HINTS, units_hint=("degrees_east", "degree_east", "degree_e", "degrees_e"),
                      standard_names=("longitude",), axis="X")
    var = _find_sst_var(ds, lat, lon, time)
    da = ds[var]
    units = str(da.attrs.get("units", "")).strip()

    tvals = pd.DatetimeIndex(pd.to_datetime(
        [str(t) if not hasattr(t, "isoformat") else t.isoformat() for t in ds[time].values])
    ) if ds[time].dtype == object else pd.DatetimeIndex(ds[time].values)
    dt = pd.Series(tvals).diff().dropna()
    step_days = float(dt.median() / pd.Timedelta(days=1)) if len(dt) else np.nan

    latv, lonv = ds[lat].values, ds[lon].values
    meta = {
        "variable": var, "time": time, "lat": lat, "lon": lon, "units": units,
        "dims": {k: int(v) for k, v in da.sizes.items()},
        "time_start": str(tvals.min()), "time_end": str(tvals.max()), "n_times": int(len(tvals)),
        "median_step_days": step_days,
        "is_daily": bool(step_days <= 1.5),
        "calendar": str(ds[time].encoding.get("calendar", "standard")),
        "lat_range": [float(latv.min()), float(latv.max())],
        "lat_descending": bool(latv[0] > latv[-1]),
        "lon_range": [float(lonv.min()), float(lonv.max())],
        "lon_convention": "0..360" if lonv.max() > 180 else "-180..180",
        "lat_res": float(abs(np.median(np.diff(latv)))) if latv.size > 1 else np.nan,
        "lon_res": float(abs(np.median(np.diff(lonv)))) if lonv.size > 1 else np.nan,
        "duplicate_timestamps": int(tvals.duplicated().sum()),
    }
    # Kelvin detection from units first, then from a small sample
    kelvin = units.lower() in ("k", "kelvin", "degk", "degrees_k", "degree_k")
    if not kelvin and not units:
        sample = da.isel({time: 0}).values
        kelvin = bool(np.nanmean(sample) > 150)
    meta["kelvin"] = kelvin
    log.info("Dataset: var=%s dims=%s units=%s (%s)", var, meta["dims"], units or "?", "K->C" if kelvin else "degC")
    log.info("Time: %s -> %s | n=%d | step=%.2f d | calendar=%s", meta["time_start"], meta["time_end"],
             meta["n_times"], step_days, meta["calendar"])
    log.info("Lat %s (descending=%s) res=%.3f | Lon %s [%s] res=%.3f", meta["lat_range"], meta["lat_descending"],
             meta["lat_res"], meta["lon_range"], meta["lon_convention"], meta["lon_res"])
    if meta["duplicate_timestamps"]:
        log.warning("%d duplicate timestamps will be dropped (first kept).", meta["duplicate_timestamps"])
    return meta


def _to_pandas_time(ds, time_name):
    vals = ds[time_name].values
    if vals.dtype == object:  # cftime
        return pd.DatetimeIndex([pd.Timestamp(v.year, v.month, v.day) for v in vals])
    return pd.DatetimeIndex(vals)


PRCP_HINTS = ["precip", "prcp", "pr", "rain", "rainfall", "tp", "total_precipitation"]


TMIN_HINTS = ["tmin", "mn2t", "tasmin", "t2m_min", "minimum", "temperature_min"]


TMAX_HINTS = ["tmax", "mx2t", "tasmax", "t2m_max", "maximum", "temperature_max"]


HINTS = {"prcp": PRCP_HINTS, "tmin": TMIN_HINTS, "tmax": TMAX_HINTS}


def _pick_variable(ds, kind: str, lat: str, lon: str, forced: str | None):
    if forced:
        if forced not in ds.data_vars:
            raise KeyError(f"Variable '{forced}' not in file. Available: {list(ds.data_vars)}")
        return forced
    cands = [v for v in ds.data_vars if {lat, lon}.issubset(ds[v].dims)]
    if not cands:
        raise KeyError(f"No gridded variable (lat/lon dims) found. Variables: {list(ds.data_vars)}")
    if len(cands) == 1:
        return cands[0]
    for h in HINTS[kind]:
        for v in cands:
            name = (v + " " + str(ds[v].attrs.get("long_name", "")) + " "
                    + str(ds[v].attrs.get("standard_name", ""))).lower()
            if v.lower() == h or h in name:
                return v
    raise KeyError(f"Cannot decide which variable is '{kind}' among {cands}; set it explicitly "
                   f"(--{kind}-var).")


def load_variable(path: Path, kind: str, forced: str | None = None):
    """Open a NetCDF file and return (DataArray with dims time/lat/lon, QA dict).

    The returned array is monthly (daily or finer input is aggregated: sum for
    precipitation, mean for temperature), in mm/month (prcp) or degC (temperature), with
    invalid values masked as NaN.  Original lat/lon order and values are preserved.
    """
    import xarray as xr
    path = Path(path)
    ds = open_dataset(path)
    time = _find_coord(ds, TIME_HINTS, standard_names=("time",), axis="T")
    lat = _find_coord(ds, LAT_HINTS, units_hint=("degrees_north", "degree_north", "degree_n", "degrees_n"),
                         standard_names=("latitude",), axis="Y")
    lon = _find_coord(ds, LON_HINTS, units_hint=("degrees_east", "degree_east", "degree_e", "degrees_e"),
                         standard_names=("longitude",), axis="X")
    var = _pick_variable(ds, kind, lat, lon, forced)
    ren = {a: b for a, b in ((time, "time"), (lat, "lat"), (lon, "lon")) if a != b}
    da = ds[var].rename(ren).transpose("time", "lat", "lon", ...)
    extra = [d for d in da.dims if d not in ("time", "lat", "lon")]
    if extra:                                      # e.g. a singleton level/band dimension
        for d in extra:
            if da.sizes[d] != 1:
                raise ValueError(f"{path.name}: extra dimension '{d}' of size {da.sizes[d]}; expected 1")
        da = da.squeeze(extra, drop=True)

    # time axis -> pandas DatetimeIndex, drop duplicates
    tidx = da.indexes["time"]
    if not isinstance(tidx, pd.DatetimeIndex):
        tidx = pd.DatetimeIndex(pd.to_datetime([str(t) for t in tidx.values]))
        da = da.assign_coords(time=tidx)
    keep = ~pd.Index(tidx).duplicated()
    n_dup = int((~keep).sum())
    if n_dup:
        log.warning("%s: %d duplicate timestamps dropped (first kept)", path.name, n_dup)
        da = da.isel(time=np.where(keep)[0])
    da = da.sortby("time")

    units = str(da.attrs.get("units", "")).strip()
    step = pd.Series(da.indexes["time"]).diff().dropna()
    step_days = float(step.median() / pd.Timedelta(days=1)) if len(step) else np.nan
    qa = {"file": path.name, "variable": var, "units_in": units, "n_times_in": int(da.sizes["time"]),
          "time_start": str(da.indexes["time"][0]), "time_end": str(da.indexes["time"][-1]),
          "step_days": step_days, "lat_range": [float(da.lat.min()), float(da.lat.max())],
          "lon_range": [float(da.lon.min()), float(da.lon.max())],
          "lat_descending": bool(da.lat.size > 1 and float(da.lat[0]) > float(da.lat[-1])),
          "lat_res": float(abs(np.median(np.diff(da.lat.values)))) if da.lat.size > 1 else None,
          "lon_res": float(abs(np.median(np.diff(da.lon.values)))) if da.lon.size > 1 else None,
          "duplicate_timestamps": n_dup}

    # --- units / physical validity ---------------------------------------------------------
    da = da.astype("float32")
    if kind == "prcp":
        da = da.where(da >= 0)                      # negative values (e.g. -9999 fill) are invalid
        per_day = (settings.PRCP_UNITS == "mm/day") or ("day" in units.lower() and "month" not in units.lower())
        qa["prcp_units_assumed"] = "mm/day" if per_day else "mm/month"
        if per_day and step_days > 25:             # monthly file holding a daily-rate field
            days = da["time"].dt.days_in_month
            da = da * days
            log.info("%s: converted mm/day -> mm/month using days in month", path.name)
        elif per_day:
            pass                                    # daily data: summed below
    else:
        mean_val = float(da.isel(time=0).mean(skipna=True)) if da.sizes["time"] else np.nan
        kelvin = units.lower() in ("k", "kelvin", "degk", "degrees_k") or mean_val > 150
        if kelvin:
            da = da - 273.15
            log.info("%s: Kelvin detected -> converted to degC", path.name)
        qa["kelvin_converted"] = bool(kelvin)
        da = da.where((da > -90) & (da < 70))       # physically implausible air temperatures masked

    # --- aggregation to monthly -------------------------------------------------------------
    if step_days <= 1.5:
        log.info("%s: daily data detected; aggregating to monthly (%s)", path.name,
                 "sum" if kind == "prcp" else "mean")
        da = da.resample(time="MS").sum(skipna=False) if kind == "prcp" else da.resample(time="MS").mean()
    da = da.assign_coords(time=da.indexes["time"].to_period("M").to_timestamp())
    da.attrs.update({"units": "mm/month" if kind == "prcp" else "degC", "source_file": path.name})
    da.name = kind
    qa["units_out"] = da.attrs["units"]
    return da, qa


def load_inputs(input_dir: Path, need_temperature: bool, prcp_var=None, tmin_var=None, tmax_var=None,
                start=None, end=None):
    """Load precipitation (+ tmin/tmax when needed) on a common grid and monthly time axis.

    Temperature fields are bilinearly interpolated onto the precipitation grid when their
    grids differ (the precipitation grid is kept as the analysis grid).  Returns
    (dict of DataArrays, QA dict).
    """
    input_dir = Path(input_dir)
    start = start or settings.ANALYSIS_START
    end = end or settings.ANALYSIS_END
    files = {"prcp": input_dir / settings.PRCP_FILE}
    forced = {"prcp": prcp_var or settings.PRCP_VAR, "tmin": tmin_var or settings.TMIN_VAR,
              "tmax": tmax_var or settings.TMAX_VAR}
    if need_temperature:
        files.update({"tmin": input_dir / settings.TMIN_FILE, "tmax": input_dir / settings.TMAX_FILE})
    for k, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"{k} file not found: {f}")

    data, qa = {}, {}
    for k, f in files.items():
        data[k], qa[k] = load_variable(f, k, forced[k])

    # common period (intersection, limited to the requested window)
    t0 = max([pd.Timestamp(start)] + [d.indexes["time"][0] for d in data.values()])
    t1 = min([pd.Timestamp(end)] + [d.indexes["time"][-1] for d in data.values()])
    if t0 > t1:
        raise ValueError("No overlapping period between the input datasets and the analysis window.")
    for k in data:
        data[k], qa[k]["missing_months"] = full_monthly(data[k], t0, t1)

    # spatial alignment on the precipitation grid
    ref = data["prcp"]
    for k in ("tmin", "tmax"):
        if k in data and not same_grid(ref, data[k]):
            log.warning("%s grid differs from precipitation (%dx%d vs %dx%d): bilinear interpolation onto the "
                        "precipitation grid", k, data[k].lat.size, data[k].lon.size, ref.lat.size, ref.lon.size)
            data[k] = data[k].interp(lat=ref.lat, lon=ref.lon, method="linear")
            qa[k]["regridded_to_prcp"] = True
    if "tmin" in data:
        if bool((data["tmin"].mean("time") > data["tmax"].mean("time")).mean() > 0.01):
            log.warning("tmin exceeds tmax at >1%% of cells: check that the files are not swapped")

    # missing-value summary (reported, not hidden)
    for k, d in data.items():
        qa[k]["missing_fraction"] = float(np.isnan(d.values[:: max(1, d.sizes['time'] // 24)]).mean())
        qa[k]["analysis_period"] = [str(d.indexes["time"][0].date()), str(d.indexes["time"][-1].date())]
        qa[k]["shape"] = [int(d.sizes["time"]), int(d.sizes["lat"]), int(d.sizes["lon"])]
        log.info("%s: %s | %s -> %s | grid %dx%d | ~%.1f%% missing", k, qa[k]["variable"],
                 qa[k]["analysis_period"][0], qa[k]["analysis_period"][1], d.sizes["lat"], d.sizes["lon"],
                 100 * qa[k]["missing_fraction"])
    return data, qa


def load_sst_monthly(path: Path, target_res: float, start: str, end: str):
    """Monthly SST (degC) block-averaged to ~target_res degrees: (values[T,nlat,nlon], time, lat, lon).

    Processed one calendar year at a time so large daily global files stay memory-safe."""
    path = Path(path)
    ds = open_dataset(path)
    meta = inspect_dataset(ds)
    da = ds[meta["variable"]].rename({meta["time"]: "time", meta["lat"]: "lat", meta["lon"]: "lon"})
    da = da.transpose("time", "lat", "lon", ...)
    for d in [d for d in da.dims if d not in ("time", "lat", "lon")]:
        da = da.isel({d: 0}, drop=True)
    f = block_factor(meta["lat_res"], target_res)
    years = sorted(set(pd.DatetimeIndex(da.indexes["time"]).year))
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    chunks, times = [], []
    for y in years:
        if y < t0.year or y > t1.year:
            continue
        c = da.sel(time=slice(f"{y}-01-01", f"{y}-12-31")).astype("float32").load()
        if meta["kelvin"]:
            c = c - 273.15
        c = c.where((c > -2.5) & (c < 40))
        if meta["is_daily"] or meta["median_step_days"] < 25:
            c = c.resample(time="MS").mean()
        else:
            c = c.assign_coords(time=c.indexes["time"].to_period("M").to_timestamp())
        if f > 1:
            c = c.coarsen(lat=f, lon=f, boundary="trim").mean()
        chunks.append(c.values)
        times.extend(list(c.indexes["time"]))
        lat, lon = c["lat"].values, c["lon"].values
    ds.close()
    vals = np.concatenate(chunks, axis=0)
    ti = pd.DatetimeIndex(times)
    keep = (ti >= t0) & (ti <= t1)
    log.info("SST: %d months %s -> %s, grid %dx%d (block factor %d)", keep.sum(), ti[keep][0].strftime("%Y-%m"),
             ti[keep][-1].strftime("%Y-%m"), vals.shape[1], vals.shape[2], f)
    return vals[keep], ti[keep], lat, lon
