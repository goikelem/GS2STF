"""Regional SST extraction and index construction (R34, WIOD, EIOD, DMI), with ENSO/IOD event bookkeeping.

Origin: data_processing.py, indices_calculation.py, teleconnection.py."""
from __future__ import annotations

import calendar
import json
import logging

import numpy as np
import pandas as pd

from src import settings
from src.data_ingestion.netcdf_loader import _to_pandas_time
from src.teleconnections.enso import enso_events
from src.teleconnections.events import event_summary
from src.teleconnections.iod import iod_events


log = logging.getLogger("sst.data")


def normalise_lon(lon):
    """Map any longitude(s) to -180..180."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def region_masks(latv: np.ndarray, lonv: np.ndarray, lat_bounds, lon_bounds):
    """Boolean index arrays for lat and lon (handles 0-360 / -180-180, any lat order, dateline crossing)."""
    lat_lo, lat_hi = sorted(lat_bounds)
    lat_m = (latv >= lat_lo) & (latv <= lat_hi)
    lon_n = normalise_lon(lonv)
    lo, hi = normalise_lon(lon_bounds[0]), normalise_lon(lon_bounds[1])
    if lo <= hi:
        lon_m = (lon_n >= lo) & (lon_n <= hi)
    else:  # crosses the dateline
        lon_m = (lon_n >= lo) | (lon_n <= hi)
    return lat_m, lon_m


def cos_lat_weighted_mean(values: np.ndarray, lat_vals: np.ndarray) -> np.ndarray:
    """values: (time, lat, lon) -> (time,) mean weighted by cos(lat), ignoring NaNs per time step."""
    w = np.cos(np.deg2rad(lat_vals))[None, :, None] * np.ones_like(values)
    valid = np.isfinite(values)
    num = np.nansum(np.where(valid, values, 0.0) * w, axis=(1, 2))
    den = np.sum(np.where(valid, w, 0.0), axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _cache_key(meta, regions) -> str:
    return json.dumps({"regions": regions, "end": meta["time_end"], "start": meta["time_start"],
                       "n": meta["n_times"], "var": meta["variable"]}, sort_keys=True)


def regional_monthly_sst(ds, meta: dict, regions: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (monthly SST degC per region, quality report) indexed by month start."""
    cache_dir = settings.out("cache")
    csv, key_f = cache_dir / "regional_monthly_sst.csv", cache_dir / "regional_monthly_sst.key"
    rep_f = cache_dir / "quality_report.csv"
    key = _cache_key(meta, regions)
    if (not settings.FORCE_RECOMPUTE and csv.exists() and key_f.exists() and rep_f.exists()
            and key_f.read_text() == key):
        log.info("Using cached regional monthly SST (%s). Set FORCE_RECOMPUTE=True to rebuild.", csv)
        return (pd.read_csv(csv, index_col=0, parse_dates=True),
                pd.read_csv(rep_f, index_col=0, parse_dates=True))

    tname, latn, lonn, var = meta["time"], meta["lat"], meta["lon"], meta["variable"]
    tindex = _to_pandas_time(ds, tname)
    keep = ~tindex.duplicated(keep="first")
    pos_all = np.flatnonzero(keep.values)
    tindex = tindex[keep]

    latv, lonv = ds[latn].values, ds[lonn].values
    idx = {}
    for name, rg in regions.items():
        lm, om = region_masks(latv, lonv, rg["lat"], rg["lon"])
        if lm.sum() == 0 or om.sum() == 0:
            raise ValueError(f"Region {name} {rg} contains no grid points (lat pts={lm.sum()}, lon pts={om.sum()}).")
        idx[name] = (np.flatnonzero(lm), np.flatnonzero(om))
        log.info("Region %-4s: %d lat x %d lon points (lat %.2f..%.2f, lon %.2f..%.2f)", name, lm.sum(), om.sum(),
                 latv[lm].min(), latv[lm].max(), normalise_lon(lonv[om]).min(), normalise_lon(lonv[om]).max())

    da = ds[var]
    dim_order = da.dims
    series = {n: [] for n in regions}
    nvalid_frac = {n: [] for n in regions}
    times_out = []
    years = sorted(set(tindex.year))
    for yr in years:
        sel_pos = pos_all[(tindex.year == yr)]
        if sel_pos.size == 0:
            continue
        for name in regions:
            li, oi = idx[name]
            sub = da.isel({tname: sel_pos, latn: li, lonn: oi}).transpose(tname, latn, lonn)
            arr = np.asarray(sub.values, dtype="float64")
            if meta["kelvin"]:
                arr = arr - 273.15
            arr[(arr < -2.5) | (arr > 40)] = np.nan   # land/ice/invalid guard (degC)
            series[name].append(cos_lat_weighted_mean(arr, latv[li]))
            nvalid_frac[name].append(np.isfinite(arr).mean(axis=(1, 2)))
        times_out.append(tindex[tindex.year == yr])
        log.info("Processed %d", yr)

    t_all = times_out[0].append(times_out[1:]) if len(times_out) > 1 else times_out[0]
    daily = pd.DataFrame({n: np.concatenate(series[n]) for n in regions}, index=t_all)
    frac = pd.DataFrame({n: np.concatenate(nvalid_frac[n]) for n in regions}, index=t_all)

    # Monthly aggregation + completeness
    if meta["is_daily"]:
        monthly = daily.resample("MS").mean()
        nobs = daily.iloc[:, 0].resample("MS").count()
        dim = pd.Series([calendar.monthrange(d.year, d.month)[1] for d in monthly.index], index=monthly.index)
        cover = nobs / dim
        last_obs = daily.index.max()
        month_end = last_obs.replace(day=calendar.monthrange(last_obs.year, last_obs.month)[1])
        complete = cover >= settings.MIN_DAYS_FRACTION
        if last_obs.normalize() < month_end.normalize():  # latest month still in progress
            complete.iloc[-1] = False
    else:
        monthly = daily.resample("MS").mean()
        cover = pd.Series(1.0, index=monthly.index)
        complete = monthly.notna().all(axis=1)

    report = pd.DataFrame({"n_obs": daily.iloc[:, 0].resample("MS").count(),
                           "coverage": cover, "complete": complete,
                           **{f"valid_frac_{n}": frac[n].resample("MS").mean() for n in regions}})
    n_bad = int((~complete).sum())
    if n_bad:
        log.warning("%d incomplete month(s) excluded; latest complete month = %s", n_bad,
                    monthly.index[complete.values].max().strftime("%Y-%m"))
    monthly = monthly[complete.values]
    nan_ct = int(monthly.isna().sum().sum())
    if nan_ct:
        log.warning("%d NaN monthly values remain (all-land/missing regions?); they are left as NaN.", nan_ct)
    log.info("Latest complete month: %s", monthly.index.max().strftime("%Y-%m"))

    monthly.to_csv(csv)
    report.to_csv(rep_f)
    key_f.write_text(key)
    return monthly, report


def box_mean(arr, lat, lon, box) -> np.ndarray:
    """cos(lat)-weighted mean over a lat/lon box (NaN cells excluded per time step)."""
    lm, om = region_masks(lat, lon, box["lat"], box["lon"])
    sub = arr[:, lm][:, :, om]
    w = np.cos(np.deg2rad(lat[lm]))[None, :, None] * np.ones_like(sub)
    w = np.where(np.isfinite(sub), w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w.sum((1, 2)) > 0, np.nansum(np.where(np.isfinite(sub), sub, 0) * w, (1, 2)) / w.sum((1, 2)),
                        np.nan)


def build_indices(anom: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {'R34','WIOD','EIOD','DMI'} monthly anomaly series (degC)."""
    missing = [k for k in ("R34", "WIOD", "EIOD") if k not in anom]
    if missing:
        raise KeyError(f"Missing regional anomaly series: {missing}")
    idx = {k: anom[k].copy() for k in ("R34", "WIOD", "EIOD")}
    idx["DMI"] = (anom["WIOD"] - anom["EIOD"]).rename("DMI")   # DMI = WIOD - EIOD
    for k, s in idx.items():
        s.name = k
    return idx


def indices_frame(idx: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame({k: idx[k] for k in settings.INDICES})
    df.index.name = "Date"
    return df


def save_indices(idx: dict[str, pd.Series], monthly_sst: pd.DataFrame, clim: pd.DataFrame,
                 baseline: tuple[int, int]) -> pd.DataFrame:
    d = settings.out("indices")
    df = indices_frame(idx)
    for k in settings.INDICES:
        o = pd.DataFrame({"Date": df.index.strftime("%Y-%m-%d"), "Year": df.index.year,
                          "Month": df.index.month, k: df[k].round(4).values})
        o.to_csv(d / f"{k}_monthly_index.csv", index=False)
    comb = pd.DataFrame({"Date": df.index.strftime("%Y-%m-%d"), "Year": df.index.year, "Month": df.index.month})
    for k in settings.INDICES:
        comb[k] = df[k].round(4).values
    comb.to_csv(d / "all_climate_indices_monthly.csv", index=False)
    monthly_sst.round(4).to_csv(d / "regional_monthly_sst_degC.csv", index_label="Date")
    clim.round(4).to_csv(d / f"regional_monthly_climatology_{baseline[0]}-{baseline[1]}.csv")
    # anomalies per region (R34/WIOD/EIOD already equal the indices)
    log.info("Saved index CSVs to %s", d)
    return comb


def save_events(idx: dict[str, pd.Series]) -> dict:
    d = settings.out("analysis")
    r34, dmi = idx["R34"].dropna(), idx["DMI"].dropna()
    en_ev, en_ph = enso_events(r34)
    io_ev, io_ph = iod_events(dmi)
    en_ev.to_csv(d / "ENSO_events.csv", index=False)
    io_ev.to_csv(d / "IOD_events.csv", index=False)
    en_ph.rename("ENSO_phase").to_csv(d / "ENSO_monthly_phase.csv", index_label="Date")
    io_ph.rename("IOD_phase").to_csv(d / "IOD_monthly_phase.csv", index_label="Date")
    ny = len(r34) / 12
    summ = pd.concat([event_summary(en_ev, ny, "ENSO (Nino3.4)"), event_summary(io_ev, len(dmi) / 12, "IOD (DMI)")])
    summ.to_csv(d / "event_duration_frequency_summary.csv", index=False)
    log.info("ENSO events: %d | IOD events: %d", len(en_ev), len(io_ev))
    return {"enso_events": en_ev, "iod_events": io_ev, "enso_phase": en_ph, "iod_phase": io_ph, "summary": summ}
