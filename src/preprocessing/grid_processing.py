"""Gridded-field preprocessing: monthly grids -> seasonal fields (years x lat x lon), mirroring the station pathway.

Seasonal totals (precipitation) / means (temperature) require every month of the season in the record; a cell with any
missing month stays NaN for that year.  Cross-year seasons use the same start/end year labels as the station pathway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import settings
from src.forecasting.seasonal import MIN_GAIN as S_MIN_GAIN
from src.preprocessing.station_processing import assign_season_year, season_months

log = logging.getLogger("s2s.grid")
SUM_KINDS = {"prcp", "precipitation"}          # everything else (temperature, SST) is averaged


@dataclass
class MonthlyGrid:
    data: np.ndarray            # (time, lat, lon), monthly, NaN = missing
    time: pd.DatetimeIndex      # month starts
    lat: np.ndarray
    lon: np.ndarray
    kind: str                   # 'prcp' | 'tmax' | 'tmin' | 'sst' ...
    units: str = ""


def load_monthly_grid(path, kind: str) -> MonthlyGrid:
    """Read a NetCDF file with the project loader (needs xarray); daily input is aggregated to months there."""
    from src.data_ingestion import netcdf_loader
    if kind == "sst":                              # SST: block-averaged to settings.SST_RES_DEG, degC, monthly
        vals, time, lat, lon = netcdf_loader.load_sst_monthly(path, settings.SST_RES_DEG, settings.ANALYSIS_START, settings.ANALYSIS_END)
        return MonthlyGrid(np.asarray(vals, dtype=float), pd.DatetimeIndex(time).to_period("M").to_timestamp(), lat, lon, "sst", "degC")
    da, _qa = netcdf_loader.load_variable(path, kind)
    t = pd.DatetimeIndex(da.indexes["time"]).to_period("M").to_timestamp()
    return MonthlyGrid(np.asarray(da.values, dtype=float), t, da["lat"].values, da["lon"].values, kind,
                       str(da.attrs.get("units", "")))


def seasonal_field(grid: MonthlyGrid, season, year_label: str | None = None, how: str | None = None):
    """Aggregate to a season.  Returns (years list, field (n_years, lat, lon))."""
    year_label = year_label or settings.STATION_SEASON_YEAR_LABEL
    how = how or ("sum" if grid.kind in SUM_KINDS else "mean")
    ms = season_months(season)
    t = pd.DatetimeIndex(grid.time)
    sel = np.isin(t.month, ms)
    sy = assign_season_year(t.month[sel], t.year[sel], ms, year_label)
    idx = np.where(sel)[0]
    years, fields = [], []
    for y in sorted(set(sy)):
        rows = idx[sy == y]
        if len(rows) != len(ms) or len(set(t.month[rows])) != len(ms):      # incomplete season in the record
            continue
        block = grid.data[rows]
        agg = block.sum(0) if how == "sum" else block.mean(0)
        agg = np.where(np.isnan(block).any(0), np.nan, agg)
        years.append(int(y)); fields.append(agg)
    if not fields:
        raise ValueError("No complete season found in the record")
    return years, np.stack(fields)


# ---------------------------------------------------------------- command-line workflow
KINDS = {"precipitation": "prcp", "tmax": "tmax", "tmin": "tmin", "sst": "sst"}
FILES = {"precipitation": "PRCP_FILE", "tmax": "TMAX_FILE", "tmin": "TMIN_FILE", "sst": "SST_FILE"}


def run(args, grid: MonthlyGrid | None = None) -> dict:
    """``python main.py --grid-forecast``: gridded seasonal forecast, hindcast skill, maps and exports.

    ``grid`` can be passed directly (tests, notebooks); otherwise the NetCDF file is read with the project loader.
    """
    import json
    from pathlib import Path
    from src.export import geotiff_export, netcdf_export
    from src.forecasting import seasonal as sf
    from src.spatial import forecast_maps, shapefile_clip
    from src.verification import probabilistic_scores as vs

    var = args.grid_variable
    if args.target_year is None or args.init_month is None:
        raise SystemExit("--grid-forecast needs --init-month (1-12) and --target-year")
    if grid is None:
        from src.data_ingestion import netcdf_loader
        path = Path(args.grid_nc) if args.grid_nc else netcdf_loader.resolve_input_file(settings.CLIM_INPUT_DIR, getattr(settings, FILES[var]))
        grid = load_monthly_grid(path, KINDS[var])
    from src.teleconnections import sst_indices
    cols = [c.strip() for c in args.predictors.split(",")] if getattr(args, "predictors", None) else None
    ind = sst_indices.load_predictor_table(args.sst_indices, cols)
    out = Path(args.grid_output or (settings.PROJECT_ROOT / "outputs" / "spatial_forecasts")); out.mkdir(parents=True, exist_ok=True)

    spec = sf.ForecastSpec(args.season, args.init_month, args.predictor_months, settings.STATION_SEASON_YEAR_LABEL)
    years, field = seasonal_field(grid, args.season)
    lat, lon = grid.lat, grid.lon
    if getattr(args, "grid_coarsen", 1) and args.grid_coarsen > 1:       # block-average the seasonal field (faster forecasts)
        from src.preprocessing import regridding
        field, lat, lon = regridding.coarsen_array(field, lat, lon, args.grid_coarsen)
        log.info("Coarsened by %dx to %d x %d cells", args.grid_coarsen, len(lat), len(lon))

    bbox = args.bbox if args.bbox else (args.grid_region if args.grid_region else None)
    polys = shapefile_clip.read_polygons(args.clip_file) if args.clip_file else None
    mask = shapefile_clip.region_mask(lat, lon, bbox, polys)
    arrs, lat, lon, mask = shapefile_clip.crop_to_mask({"field": field}, lat, lon, mask)
    field = arrs["field"]
    diag = sf.predictor_diagnostics(sf.build_predictors(ind, spec, range(min(years), args.target_year + 1)))
    diag["correlation"].round(3).assign(VIF=diag["vif"].round(2)).to_csv(
        out / f"predictor_diagnostics_{spec.label(args.target_year)}_init{args.init_month:02d}.csv", index_label="predictor")
    methods = [m.strip() for m in (args.methods or "ridge,pcr,cca").split(",") if m.strip()]
    log.info("%s %s: %d seasons, %d x %d grid (%d cells in region), methods %s", var, spec.label(args.target_year),
             len(years), len(lat), len(lon), int(mask.sum()), methods)
    from src.forecasting import dynamical
    dyn_idx = dynamical.load_index_forecasts(args.dynamical_indices, spec) if getattr(args, "dynamical_indices", None) else None
    res = sf.run_grid_forecast(field, years, lat, lon, ind, spec, args.target_year, methods=methods,
                               transform="sqrt" if var == "precipitation" else "none", scheme=args.hindcast_scheme,
                               mask=mask, ensemble_mode=args.ensemble_mode, max_predictors=args.max_predictors,
                               min_gain=getattr(args, "min_gain", S_MIN_GAIN), extra_predictors=dyn_idx)
    stem = f"{var}_{spec.label(args.target_year)}_init{args.init_month:02d}"
    units = grid.units or ("mm/season" if var == "precipitation" else "degC")
    nc = netcdf_export.write_forecast_netcdf(out / f"{stem}.nc", res, var, units, methods)
    bands = {"forecast": res["forecast"], "anomaly": res["anomaly"], "prob_below": res["probabilities"][0],
             "prob_normal": res["probabilities"][1], "prob_above": res["probabilities"][2],
             "dominant_category": np.where(res["dominant_category"] >= 0, res["dominant_category"], np.nan)}
    if "rpss" in res["skill"]:
        bands["hindcast_rpss"] = res["skill"]["rpss"]
    tif = geotiff_export.write_forecast_geotiff(out / f"{stem}.tif", bands, lat, lon)
    pngs = forecast_maps.plot_forecast_maps(res, out, var, units, polys)
    if len(res["members"]) > 1:
        vs.compare(res["members"], n_boot=300, block=2).round(4).to_csv(out / f"method_comparison_{stem}.csv", index=False)
    pooled = res["pooled"].round(4).to_dict()
    (out / f"{stem}_metadata.json").write_text(json.dumps(
        {**res["meta"], "variable": var, "units": units, "methods": methods, "region": args.grid_region or args.bbox or
         ("polygon:" + str(args.clip_file) if args.clip_file else "full grid"), "n_cells": res["n_cells"],
         "n_training_years": res["n_train_years"], "hindcast_scheme": args.hindcast_scheme, "pooled_hindcast": pooled}, indent=2, default=str))
    log.info("Grid forecast written to %s (%s, %s, %d maps); pooled hindcast RPSS %.3f", out, nc.name, tif.name, len(pngs),
             pooled.get("rpss", float("nan")))
    return res
