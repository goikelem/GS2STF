"""Station preprocessing: monthly and seasonal aggregation, anomalies, grid-to-station predictor extraction.

All functions take / return tidy pandas tables (columns station, lat, lon, variable, ...) so station
coordinates and metadata are never lost.  Seasons are month lists from ``settings.STATION_SEASONS``
(JJAS, ONDJ, FMAM, ... or any user-defined list); cross-year seasons such as ONDJ and DJF are handled.
"""
from __future__ import annotations

import calendar
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings
from src.data_ingestion import station_csv_loader as loader

log = logging.getLogger("s2s.stations")

SUM_VARIABLES = {"precipitation", "wet_days"}          # totals / counts; every other variable is averaged
KEYS = ["station", "lat", "lon", "variable"]


def _min_fraction(variable: str, override=None) -> float:
    if override is not None:
        return float(override)
    d = settings.STATION_MIN_DAY_FRACTION
    return float(d.get(variable, d["default"]))


# ---------------------------------------------------------------- aggregation
def monthly_aggregate(obs: pd.DataFrame, min_day_fraction: float | None = None) -> pd.DataFrame:
    """Daily -> monthly (sum for precipitation, mean otherwise).

    A month is kept only if the share of valid days >= the required fraction (precipitation 1.0, temperature
    0.8 by default; see config/datasets.yaml).  Series whose median time step is >= 27 days are treated as
    already monthly and passed through.  Returns station, lat, lon, variable, month, value, n_days.
    """
    rows = []
    for (stn, var), g in obs.groupby(["station", "variable"], sort=False):
        g = g.dropna(subset=["value"]).sort_values("date")
        if g.empty:
            continue
        lat, lon = g["lat"].iloc[0], g["lon"].iloc[0]
        month = g["date"].dt.to_period("M").dt.to_timestamp()
        step = g["date"].diff().dt.days.median()
        if pd.notna(step) and step >= 27:                     # already monthly
            m = pd.DataFrame({"month": month.values, "value": g["value"].values, "n_days": np.nan})
        else:
            frac_req = _min_fraction(var, min_day_fraction)
            agg = g.groupby(month.values)["value"].agg(["sum" if var in SUM_VARIABLES else "mean", "size"])
            agg.columns = ["value", "n_days"]
            dim = np.array([calendar.monthrange(t.year, t.month)[1] for t in agg.index])
            agg.loc[(agg["n_days"] / dim) < frac_req - 1e-12, "value"] = np.nan
            m = agg.rename_axis("month").reset_index()
        m.insert(0, "station", stn); m.insert(1, "lat", lat); m.insert(2, "lon", lon); m.insert(3, "variable", var)
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=KEYS + ["month", "value", "n_days"])
    return pd.concat(rows, ignore_index=True)


def season_months(season) -> list[int]:
    """Month list for a season name from settings.STATION_SEASONS, or pass a list of month numbers."""
    if isinstance(season, str):
        if season not in settings.STATION_SEASONS:
            raise KeyError(f"Unknown season '{season}'. Configured: {list(settings.STATION_SEASONS)} "
                           f"(add more under seasons.station in config/forecast_settings.yaml)")
        return list(settings.STATION_SEASONS[season])
    return [int(m) for m in season]


def assign_season_year(month_numbers, years, ms, year_label: str = "start") -> np.ndarray:
    """season_year label for months belonging to season ``ms``: cross-year seasons (ONDJ, DJF) are labelled by the year
    they start in (``'start'``) or end in (``'end'``)."""
    month_numbers = np.asarray(month_numbers); years = np.asarray(years)
    wrap = next((i for i in range(1, len(ms)) if ms[i] < ms[i - 1]), None)    # index where the year rolls over
    if wrap is None:
        return years.copy()
    pos = np.array([ms.index(m) for m in month_numbers])
    return np.where(pos >= wrap, years - 1, years) if year_label == "start" else np.where(pos < wrap, years + 1, years)


def seasonal_aggregate(monthly: pd.DataFrame, season, year_label: str | None = None) -> pd.DataFrame:
    """Monthly -> seasonal totals (precipitation) or means (others); a season needs ALL its months.

    Cross-year seasons (ONDJ, DJF) are labelled by the year they start in (``year_label='start'``,
    e.g. ONDJ 2026 = Oct 2026-Jan 2027) or end in (``'end'``, DJF 2027 = Dec 2026-Feb 2027).
    """
    year_label = year_label or settings.STATION_SEASON_YEAR_LABEL
    if year_label not in ("start", "end"):
        raise ValueError("year_label must be 'start' or 'end'")
    ms = season_months(season)
    name = season if isinstance(season, str) else "".join(calendar.month_abbr[m][0] for m in ms)
    d = monthly[monthly["month"].dt.month.isin(ms)].copy()
    d["season_year"] = assign_season_year(d["month"].dt.month.values, d["month"].dt.year.values, ms, year_label)
    d = d.dropna(subset=["value"])
    out = []
    for var, g in d.groupby("variable"):
        how = "sum" if var in SUM_VARIABLES else "mean"
        a = g.groupby(KEYS[:3] + ["season_year"]).agg(value=("value", how), n_months=("value", "size")).reset_index()
        a = a[a["n_months"] == len(ms)].drop(columns="n_months")
        a.insert(3, "variable", var)
        out.append(a)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=KEYS + ["season_year", "value"])
    res.insert(4, "season", name)
    return res[KEYS + ["season", "season_year", "value"]].sort_values(["variable", "station", "season_year"]).reset_index(drop=True)


# ---------------------------------------------------------------- rainfall occurrence (wet days)
def season_days(season, season_year: int, year_label: str | None = None) -> int:
    """Number of calendar days in the season labelled ``season_year`` (cross-year seasons handled)."""
    year_label = year_label or settings.STATION_SEASON_YEAR_LABEL
    ms = season_months(season)
    wrap = next((i for i in range(1, len(ms)) if ms[i] < ms[i - 1]), None)
    start = season_year - 1 if (wrap is not None and year_label == "end") else season_year
    return sum(calendar.monthrange(start + (1 if wrap is not None and i >= wrap else 0), m)[1] for i, m in enumerate(ms))


def seasonal_wet_day_fraction(obs: pd.DataFrame, season, threshold: float = 1.0, year_label: str | None = None):
    """Fraction of days with precipitation >= ``threshold`` mm per station and season (complete seasons only).

    Returns (table with variable 'wet_day_fraction', typical season length in days).  Days with missing data make
    their month incomplete (as for rainfall totals), so a fraction is never based on a partly observed season.
    """
    pr = obs[obs["variable"] == "precipitation"].copy()
    if pr.empty:
        raise ValueError("No precipitation observations")
    pr["value"] = np.where(pr["value"].isna(), np.nan, (pr["value"] >= threshold).astype(float))
    pr["variable"] = "wet_days"
    tab = seasonal_aggregate(monthly_aggregate(pr, min_day_fraction=1.0), season, year_label)
    tab["value"] = tab["value"].values / np.array([season_days(season, int(y), year_label) for y in tab["season_year"]])
    tab["variable"] = "wet_day_fraction"
    n = float(np.mean([season_days(season, int(y), year_label) for y in tab["season_year"].unique()]))
    return tab, n


# ---------------------------------------------------------------- anomalies
def seasonal_anomalies(seasonal: pd.DataFrame, ref_start: int | None = None, ref_end: int | None = None,
                       min_ref_years: int | None = None) -> pd.DataFrame:
    """Add climatology, anomaly and (precipitation) percent-of-normal against a reference period.

    Stations / variables with fewer than ``min_ref_years`` valid reference seasons get NaN anomalies
    (counted in the log) instead of a silently different baseline.
    """
    ref_start = settings.BASELINE_START if ref_start is None else ref_start
    ref_end = settings.BASELINE_END if ref_end is None else ref_end
    min_ref_years = settings.STATION_MIN_REF_YEARS if min_ref_years is None else min_ref_years
    s = seasonal.copy()
    ref = s[(s["season_year"] >= ref_start) & (s["season_year"] <= ref_end)]
    gk = ["station", "variable", "season"]
    clim = ref.groupby(gk)["value"].agg(climatology="mean", n_ref="size").reset_index()
    s = s.merge(clim, on=gk, how="left")
    ok = s["n_ref"] >= min_ref_years
    s.loc[~ok, "climatology"] = np.nan
    s["anomaly"] = s["value"] - s["climatology"]
    pr = s["variable"] == "precipitation"
    s["percent_of_normal"] = np.where(pr & (s["climatology"] > 0), 100 * s["value"] / s["climatology"], np.nan)
    n_bad = int((~clim["n_ref"].ge(min_ref_years)).sum())
    if n_bad:
        log.warning("%d station/variable/season series have < %d reference years in %d-%d: anomalies set to NaN",
                    n_bad, min_ref_years, ref_start, ref_end)
    s["ref_period"] = f"{ref_start}-{ref_end}"
    return s.drop(columns="n_ref")


# ---------------------------------------------------------------- grid -> station extraction
def _norm_lon(lon, like):
    lon = np.asarray(lon, dtype=float)
    if np.nanmax(like) > 180:
        return np.mod(lon, 360.0)
    return np.where(lon > 180, lon - 360.0, lon)


def _nearest_index(axis: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(axis - x)))


def extract_grid_predictors(stations: pd.DataFrame, field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                            time_index: pd.DatetimeIndex, method: str = "nearest",
                            max_distance_deg: float | None = None):
    """Sample a gridded field (time, lat, lon) at station coordinates.

    method 'nearest' or 'bilinear'.  Stations farther than ``max_distance_deg`` from the grid (default: 1.5 x the
    grid spacing) get NaN.  Returns (values: DataFrame[time x station], cells: DataFrame with the grid cell used,
    its coordinates and the offset in degrees) so station metadata and the matched cell stay together.
    """
    field = np.asarray(field, dtype=float)
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    if field.shape != (len(time_index), len(lat), len(lon)):
        raise ValueError(f"field shape {field.shape} does not match (time, lat, lon) = "
                         f"{(len(time_index), len(lat), len(lon))}")
    if method not in ("nearest", "bilinear"):
        raise ValueError("method must be 'nearest' or 'bilinear'")
    dlat = np.abs(np.median(np.diff(lat))) if len(lat) > 1 else 1.0
    dlon = np.abs(np.median(np.diff(lon))) if len(lon) > 1 else 1.0
    tol = max_distance_deg if max_distance_deg is not None else 1.5 * max(dlat, dlon)
    cols, cells = {}, []
    for r in stations.itertuples(index=False):
        slon = float(_norm_lon(r.lon, lon))
        i, j = _nearest_index(lat, r.lat), _nearest_index(lon, slon)
        off = float(np.hypot(lat[i] - r.lat, lon[j] - slon))
        inside = (abs(lat[i] - r.lat) <= tol) and (abs(lon[j] - slon) <= tol)
        if not inside:
            series = np.full(len(time_index), np.nan)
        elif method == "nearest":
            series = field[:, i, j]
        else:
            series = _bilinear(field, lat, lon, r.lat, slon)
        cols[r.station] = series
        cells.append({"station": r.station, "lat": r.lat, "lon": r.lon, "grid_lat": lat[i], "grid_lon": lon[j],
                      "offset_deg": off, "inside_grid": bool(inside)})
    values = pd.DataFrame(cols, index=pd.DatetimeIndex(time_index))
    cells = pd.DataFrame(cells)
    if (~cells["inside_grid"]).any():
        log.warning("%d station(s) fall outside the grid tolerance: %s", int((~cells["inside_grid"]).sum()),
                    cells.loc[~cells["inside_grid"], "station"].tolist()[:5])
    return values, cells


def _bilinear(field, lat, lon, y, x):
    """Bilinear sample; NaN if a surrounding cell is NaN or the point is outside the grid axes."""
    if not (lat.min() <= y <= lat.max() and lon.min() <= x <= lon.max()):
        return np.full(field.shape[0], np.nan)
    ia = np.argsort(lat); ja = np.argsort(lon)
    la, lo = lat[ia], lon[ja]
    i1 = int(np.clip(np.searchsorted(la, y), 1, len(la) - 1)); j1 = int(np.clip(np.searchsorted(lo, x), 1, len(lo) - 1))
    i0, j0 = i1 - 1, j1 - 1
    wy = (y - la[i0]) / (la[i1] - la[i0]); wx = (x - lo[j0]) / (lo[j1] - lo[j0])
    f = field[:, ia][:, :, ja]
    return ((1 - wy) * (1 - wx) * f[:, i0, j0] + (1 - wy) * wx * f[:, i0, j1]
            + wy * (1 - wx) * f[:, i1, j0] + wy * wx * f[:, i1, j1])


def extract_from_dataarray(stations: pd.DataFrame, da, method: str = "nearest", max_distance_deg=None):
    """Convenience wrapper for the DataArrays returned by data_ingestion.netcdf_loader (dims time, lat, lon)."""
    return extract_grid_predictors(stations, da.values, da["lat"].values, da["lon"].values,
                                   da.indexes["time"], method, max_distance_deg)


def coverage_report(obs: pd.DataFrame) -> pd.DataFrame:
    """Record length and gaps per station and variable: first / last valid date, % missing inside the record, the longest
    gap in days and the number of calendar years with at least 90% of days observed."""
    rows = []
    for (stn, var), g in obs.groupby(["station", "variable"], sort=True):
        v = g.dropna(subset=["value"]).sort_values("date")
        row = {"station": stn, "variable": var, "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0], "n_valid": int(len(v))}
        if v.empty:
            rows.append({**row, "first": pd.NaT, "last": pd.NaT, "days_in_record": 0, "pct_missing": 100.0,
                         "longest_gap_days": 0, "years_with_90pct_days": 0}); continue
        s = v.set_index("date")["value"].reindex(pd.date_range(v["date"].min(), v["date"].max()))
        miss = s.isna().values.astype(int)
        edges = np.diff(np.concatenate([[0], miss, [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        longest = int((ends - starts).max()) if len(starts) else 0
        per_year = s.notna().groupby(s.index.year).sum()
        full = np.array([366 if calendar.isleap(y) else 365 for y in per_year.index])
        rows.append({**row, "first": s.index[0].date(), "last": s.index[-1].date(), "days_in_record": int(len(s)),
                     "pct_missing": round(100.0 * float(miss.mean()), 2), "longest_gap_days": longest,
                     "years_with_90pct_days": int((per_year.values / full >= 0.9).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- command-line workflow
def run(args) -> None:
    """``python main.py --stations FILE``: load, QC, aggregate, compute anomalies, write CSVs."""
    out = Path(args.station_output or (settings.PROJECT_ROOT / "outputs" / "station_forecasts"))
    out.mkdir(parents=True, exist_ok=True)
    cmap = dict(kv.split("=", 1) for kv in args.station_columns.split(",")) if args.station_columns else None
    dayfirst = True if getattr(args, "station_dayfirst", False) else (False if getattr(args, "station_monthfirst", False) else None)
    try:
        data = loader.load_station_csv(args.stations, column_map=cmap, variable=args.station_variable,
                                       date_format=getattr(args, "station_date_format", None), dayfirst=dayfirst,
                                       sep=getattr(args, "station_sep", None))
    except (ValueError, KeyError) as e:                       # input problems: a clear message, not a traceback
        raise SystemExit(f"Cannot read {args.stations}: {e}")
    coverage_report(data.observations).to_csv(out / "station_coverage.csv", index=False)
    if getattr(args, "subseasonal", False):                 # weekly / multi-week forecasts need only the daily data
        data.stations.to_csv(out / "station_metadata.csv", index=False)
        (out / "station_data_quality_report.json").write_text(json.dumps(data.report, indent=2, default=str))
        run_subseasonal(args, data, out)
        return
    monthly = monthly_aggregate(data.observations)
    seasonal = seasonal_anomalies(seasonal_aggregate(monthly, args.station_season), args.ref_start, args.ref_end)
    data.stations.to_csv(out / "station_metadata.csv", index=False)
    monthly.to_csv(out / "station_monthly.csv", index=False)
    seasonal.round(4).to_csv(out / f"station_{args.station_season}_seasonal_anomalies.csv", index=False)
    (out / "station_data_quality_report.json").write_text(json.dumps(data.report, indent=2, default=str))
    log.info("Station summary written to %s (%d stations, %d seasonal values)", out, len(data.stations), len(seasonal))
    if getattr(args, "station_forecast", False):
        run_forecast(args, data, monthly, out)


def run_forecast(args, data, monthly, out: Path) -> None:
    """Seasonal station forecasts + out-of-sample hindcast verification from SST indices (``--station-forecast``)."""
    from src.export import csv_export
    from src.forecasting import precipitation, seasonal as sf, temperature
    from src.verification import probabilistic_scores as vs

    if args.target_year is None or args.init_month is None:
        raise SystemExit("--station-forecast needs --init-month (1-12) and --target-year (the season_year to forecast)")
    from src.teleconnections import sst_indices
    cols = [c.strip() for c in args.predictors.split(",")] if args.predictors else None
    ind = sst_indices.load_predictor_table(args.sst_indices, cols)
    spec = sf.ForecastSpec(args.station_season, args.init_month, args.predictor_months, settings.STATION_SEASON_YEAR_LABEL)
    from src.forecasting import dynamical
    dyn_idx = dynamical.load_index_forecasts(args.dynamical_indices, spec) if getattr(args, "dynamical_indices", None) else None
    seasonal_tbl = seasonal_aggregate(monthly, args.station_season)
    Xdiag = sf.build_predictors(ind, spec, range(int(seasonal_tbl["season_year"].min()), args.target_year + 1))
    if dyn_idx is not None:
        Xdiag = Xdiag.join(dyn_idx.reindex(Xdiag.index))
    diag = sf.predictor_diagnostics(Xdiag)
    diag["correlation"].round(3).assign(VIF=diag["vif"].round(2)).to_csv(
        out / f"predictor_diagnostics_{spec.name}_init{args.init_month:02d}.csv", index_label="predictor")
    methods = [m.strip() for m in (args.methods or "ridge,pcr,cca,rf").split(",") if m.strip()]
    kw = {"max_predictors": args.max_predictors, "scheme": args.hindcast_scheme, "methods": methods, "min_gain": args.min_gain,
          "ensemble_mode": args.ensemble_mode}
    forecasts, pooled = [], {}
    for var in sorted(seasonal_tbl["variable"].unique()):
        fn = precipitation.forecast_stations if var == "precipitation" else (
            lambda *a, **k: temperature.forecast_stations(*a, variable=var, **k))
        mf = None
        if getattr(args, "dynamical_stations", None) and var == args.dynamical_variable:
            mf = dynamical.load_station_forecasts(args.dynamical_stations, spec, variable=var)
        try:
            fc, hc, members = fn(seasonal_tbl, data.stations, ind, spec, args.target_year, monthly=monthly,
                                 extra_predictors=dyn_idx, model_forecasts=mf, **kw)
        except ValueError as e:
            log.warning("Skipping %s: %s", var, e); continue
        forecasts.append(fc)
        if len(members) > 1:
            cmp_ = vs.compare(members, ref="climatology", n_boot=500, block=2)
            cmp_.round(4).to_csv(out / f"method_comparison_{var}_{spec.name}_lead{spec.lead_months}.csv", index=False)
            log.info("%s method comparison (RPSS): %s", var, ", ".join(f"{r.method} {r.rpss:.2f}" for r in cmp_.itertuples()))
        summ = vs.summarize(hc)
        summ["per_target"].round(4).to_csv(out / f"hindcast_skill_{var}_{spec.name}_lead{spec.lead_months}.csv", index=False)
        summ["reliability"].round(4).to_csv(out / f"hindcast_reliability_{var}_{spec.name}_lead{spec.lead_months}.csv", index=False)
        ci = vs.bootstrap(hc, n_boot=500, block=2)
        pooled[var] = {**summ["pooled"].to_dict(), **{f"{k}_{c}": v for k, row in ci.iterrows() for c, v in row.items() if c != "n_boot"}}
        log.info("%s %s: pooled RPSS %.3f [%.3f, %.3f] over %d hindcast seasons", var, spec.label(args.target_year),
                 summ["pooled"].get("rpss", float("nan")), ci.loc["rpss", "lo_90"], ci.loc["rpss", "hi_90"], len(hc.obs))
    if getattr(args, "wet_day_threshold", None) is not None:        # rainfall occurrence: binomial GLM + the other methods
        try:
            wd_tbl, ndays = seasonal_wet_day_fraction(data.observations, args.station_season, args.wet_day_threshold)
            wd_methods = list(dict.fromkeys(methods + ["glm"]))
            wkw = {k: v for k, v in kw.items() if k not in ("methods",)}
            fc, hc, members = sf.run_station_forecast(wd_tbl, data.stations, ind, spec, args.target_year, "wet_day_fraction",
                                                      transform="none", methods=wd_methods, extra_predictors=dyn_idx,
                                                      trials=ndays, **wkw)
            fc["Forecast_wet_days"] = fc["Forecast"] * ndays; fc["Climatology_wet_days"] = fc["Climatology"] * ndays
            fc["wet_day_threshold_mm"] = args.wet_day_threshold
            forecasts.append(fc)
            summ = vs.summarize(hc)
            summ["per_target"].round(4).to_csv(out / f"hindcast_skill_wet_day_fraction_{spec.name}_lead{spec.lead_months}.csv", index=False)
            vs.compare(members, n_boot=300, block=2).round(4).to_csv(
                out / f"method_comparison_wet_day_fraction_{spec.name}_lead{spec.lead_months}.csv", index=False)
            vs.event_scores(hc).round(4).to_csv(out / f"hindcast_event_scores_wet_day_fraction_{spec.name}_lead{spec.lead_months}.csv", index=False)
            ci = vs.bootstrap(hc, n_boot=500, block=2)
            pooled["wet_day_fraction"] = {**summ["pooled"].to_dict(), **{f"{k}_{c}": v for k, row in ci.iterrows() for c, v in row.items() if c != "n_boot"}}
            log.info("wet_day_fraction %s: pooled RPSS %.3f [%.3f, %.3f]", spec.label(args.target_year), summ["pooled"].get("rpss", float("nan")),
                     ci.loc["rpss", "lo_90"], ci.loc["rpss", "hi_90"])
        except ValueError as e:
            log.warning("Skipping wet-day fraction: %s", e)
    if not forecasts:
        raise SystemExit("No forecast could be produced (see warnings above).")
    allf = pd.concat(forecasts, ignore_index=True)
    csv, xlsx = csv_export.save_station_forecasts(allf, out, f"station_forecast_{spec.label(args.target_year)}_init{args.init_month:02d}")
    pd.DataFrame(pooled).T.round(4).to_csv(out / f"hindcast_pooled_{spec.name}_lead{spec.lead_months}.csv", index_label="variable")
    (out / "forecast_run_metadata.json").write_text(json.dumps(
        {**spec.describe(args.target_year), "predictor_months": spec.predictor_months, "predictors": list(ind.columns) + (list(dyn_idx.columns) if dyn_idx is not None else []),
         "dynamical_stations_file": str(args.dynamical_stations) if getattr(args, "dynamical_stations", None) else None,
         "hindcast_scheme": args.hindcast_scheme, "max_predictors": args.max_predictors, "min_gain": args.min_gain, "methods": methods,
         "ensemble_mode": args.ensemble_mode if len(methods) > 1 else None,
         "notes": "Gaussian tercile probabilities; each method falls back to climatology where it does not beat it out of sample"}, indent=2))
    log.info("Forecasts written: %s, %s", csv.name, xlsx.name)


def run_subseasonal(args, data, out: Path) -> None:
    """``--subseasonal --init-date YYYY-MM-DD --lead-weeks LO HI``: weekly + multi-week station forecasts and skill."""
    import dataclasses
    from src.export import csv_export
    from src.forecasting import subseasonal as ss
    from src.teleconnections import sst_indices
    from src.verification import probabilistic_scores as vs

    if not args.init_date:
        raise SystemExit("--subseasonal needs --init-date YYYY-MM-DD (the day the forecast is issued; data before it are used)")
    init = pd.Timestamp(args.init_date)
    lo, hi = args.lead_weeks
    spec = ss.SubseasonalSpec(init.month, init.day, (lo, hi), args.pool_weeks, args.recent_weeks, args.predictor_months)
    cols = [c.strip() for c in args.predictors.split(",")] if args.predictors else None
    ind = sst_indices.load_predictor_table(args.sst_indices, cols)
    methods = [m.strip() for m in (args.methods or "ridge,pcr").split(",") if m.strip()]
    leads = ([(k, k) for k in range(lo, hi + 1)] if lo != hi and not args.no_weekly_breakdown else []) + [(lo, hi)]
    kw = {"max_predictors": args.max_predictors, "min_gain": args.min_gain, "min_train_years": args.min_train_years}
    forecasts, rows = [], []
    for var in sorted(data.observations["variable"].unique()):
        for lead in leads:
            sp = dataclasses.replace(spec, lead_weeks=lead)
            try:
                fc, hc, members = ss.run_station_subseasonal(data.observations, data.stations, ind, sp, init.year, var,
                                                             methods=methods, ensemble_mode=args.ensemble_mode, **kw)
            except ValueError as e:
                log.warning("Skipping %s weeks %s: %s", var, sp.lead_label, e); continue
            forecasts.append(fc)
            if len(hc.obs) == 0:
                log.warning("%s weeks %s: no hindcast year could be scored", var, sp.lead_label); continue
            summ = vs.summarize(hc); ci = vs.bootstrap(hc, n_boot=500, block=2)
            row = {"variable": var, "lead_weeks": sp.lead_label, "n_hindcast_years": len(hc.obs),
                   **{k: summ["pooled"].get(k) for k in ("rpss", "hss", "mean_target_correlation", "mean_target_rmse_skill")},
                   "rpss_lo_90": ci.loc["rpss", "lo_90"], "rpss_hi_90": ci.loc["rpss", "hi_90"]}
            rows.append(row)
            if len(members) > 1:
                vs.compare(members, n_boot=300, block=2).round(4).to_csv(
                    out / f"subseasonal_method_comparison_{var}_W{sp.lead_label}_{init.date()}.csv", index=False)
            summ["per_target"].round(4).to_csv(out / f"subseasonal_hindcast_skill_{var}_W{sp.lead_label}_{init.date()}.csv", index=False)
            log.info("%s weeks %s: pooled RPSS %.3f [%.3f, %.3f] over %d hindcast years", var, sp.lead_label, row["rpss"],
                     row["rpss_lo_90"], row["rpss_hi_90"], row["n_hindcast_years"])
    if not forecasts:
        raise SystemExit("No subseasonal forecast could be produced (see warnings above).")
    allf = pd.concat(forecasts, ignore_index=True)
    csv, xlsx = csv_export.save_station_forecasts(allf, out, f"subseasonal_forecast_{init.date()}_W{lo}-{hi}")
    pd.DataFrame(rows).round(4).to_csv(out / f"subseasonal_skill_summary_{init.date()}_W{lo}-{hi}.csv", index=False)
    (out / "subseasonal_run_metadata.json").write_text(json.dumps(
        {"init_date": str(init.date()), "lead_weeks": [lo, hi], "pool_weeks": args.pool_weeks, "recent_weeks": args.recent_weeks,
         "predictor_months": args.predictor_months, "methods": methods, "predictors": list(ind.columns),
         "min_train_years": args.min_train_years, "min_gain": args.min_gain,
         "notes": "data before the initialization date only; rolling-origin hindcasts; skill at weeks 2-6 is expected to be low"}, indent=2))
    log.info("Subseasonal forecasts written: %s, %s", csv.name, xlsx.name)
