#!/usr/bin/env python
"""SST climate-indices analysis and 12-month forecasting system.

Usage:
    python main.py                      # full pipeline with settings from config/*.yaml
    python main.py --fast               # quick run (fewer trees/CV folds, refit every 24 months)
    python main.py --force              # ignore cached intermediates
    python main.py --input X.nc --output DIR
    python main.py --from-regional-csv monthly_regional_sst.csv   # skip NetCDF step (cols: R34,WIOD,EIOD)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings


def setup_logging() -> Path:
    logdir = settings.out("logs")
    logfile = logdir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)):
        h.setFormatter(fmt)
        root.addHandler(h)
    return logfile


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="input NetCDF (default: settings.INPUT_NC)")
    ap.add_argument("--output", type=Path, default=None, help="output directory (default: settings.OUTPUT_DIR)")
    ap.add_argument("--fast", action="store_true", help="quick settings for smoke tests")
    ap.add_argument("--force", action="store_true", help="recompute everything (ignore cache)")
    ap.add_argument("--skip-forecast", action="store_true", help="analysis only")
    ap.add_argument("--no-extra-predictors", action="store_true",
                    help="skip the optional extra ocean predictors (Nino 1+2/3/4, Indian Ocean basin, tropical Atlantic)")
    ap.add_argument("--from-regional-csv", type=Path, default=None,
                    help="monthly regional SST CSV (columns Date,R34,WIOD,EIOD) instead of reading the NetCDF")

    st = ap.add_argument_group("Station observations (runs instead of the other workflows when --stations is given)")
    st.add_argument("--stations", type=Path, default=None, help="station CSV (long or simple format)")
    st.add_argument("--station-variable", default=None,
                    help="variable for a simple-format file: precipitation, tmax, tmin, tmean (default: precipitation)")
    st.add_argument("--station-season", default="JJAS", help="season for seasonal totals / anomalies, e.g. JJAS, ONDJ, FMAM")
    st.add_argument("--station-columns", default=None,
                    help="explicit column mapping, e.g. station=ID,date=DATE,lat=LAT,lon=LON,value=RR")
    st.add_argument("--station-forecast", action="store_true",
                    help="also produce seasonal station forecasts (tercile probabilities) + hindcast verification")
    st.add_argument("--sst-indices", type=Path, default=None,
                    help="monthly index table (default: <output>/indices/all_climate_indices_monthly.csv)")
    st.add_argument("--init-month", type=int, default=None, help="initialization month 1-12 (last month of predictor data)")
    st.add_argument("--target-year", type=int, default=None, help="season_year to forecast (start-year label for cross-year seasons)")
    st.add_argument("--max-predictors", type=int, default=2, help="max predictors per station model (default 2)")
    st.add_argument("--hindcast-scheme", choices=["rolling", "blocked"], default="rolling", help="validation scheme (default rolling-origin)")
    st.add_argument("--predictors", default=None,
                    help="restrict the ocean predictors, e.g. R34,DMI,TNA (default: every column of the predictor table)")
    st.add_argument("--min-gain", type=float, default=0.08,
                    help="a model must reduce climatology's leave-one-out error by this fraction (default 0.08) or the "
                         "station/cell keeps climatology; guards against chance predictors when many compete")
    st.add_argument("--dynamical-indices", type=Path, default=None, metavar="CSV",
                    help="dynamical-model forecasts of ocean indices for the target season (season_year,init_month,R34,DMI,...); "
                         "added as predictors DYN_*; see src/forecasting/dynamical.py")
    st.add_argument("--dynamical-stations", type=Path, default=None, metavar="CSV",
                    help="dynamical-model forecasts of the station variable (Station,season_year,init_month,value): "
                         "added as a calibrated ensemble member 'dynamical' (stations workflow)")
    st.add_argument("--dynamical-variable", default="precipitation",
                    help="variable the --dynamical-stations file forecasts, if it has no Variable column (default precipitation)")
    st.add_argument("--wet-day-threshold", type=float, default=None, metavar="MM",
                    help="also forecast rainfall OCCURRENCE: fraction of days with rain >= MM (e.g. 1.0) in the season, with a binomial GLM "
                         "among the methods; extra columns Forecast_wet_days / Climatology_wet_days, plus accuracy/POD/FAR/CSI tables")
    st.add_argument("--methods", default=None,
                    help="comma list from: ridge, pcr, cca, rf (several -> multi-model ensemble; one -> that method only). "
                         "Default: all four for seasonal stations, ridge,pcr,cca for grids (random forest is slow on large grids), ridge,pcr for subseasonal")
    st.add_argument("--ensemble-mode", choices=["skill", "equal", "select"], default="skill",
                    help="how members are combined: skill-weighted (default), equal, or best past member")
    gf = ap.add_argument_group("Gridded seasonal forecast (uses --season, --init-month, --target-year, --sst-indices, --methods, ...)")
    gf.add_argument("--grid-forecast", action="store_true", help="gridded seasonal forecast with tercile probabilities, maps, NetCDF + GeoTIFF")
    gf.add_argument("--grid-variable", choices=["precipitation", "tmax", "tmin", "sst"], default="precipitation",
                    help="field to forecast; sst = seasonal SST anomalies (use --bbox / --grid-region: the SST file is global)")
    gf.add_argument("--grid-nc", type=Path, default=None, help="NetCDF file (default: the configured file in config/datasets.yaml)")
    gf.add_argument("--grid-region", default=None, help="clip to a named region: " + ", ".join(settings.STUDY_REGIONS))
    gf.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"), default=None,
                    help="clip to a lon/lat box (overrides --grid-region)")
    gf.add_argument("--clip-file", type=Path, default=None, help="clip to polygons in a .geojson or .shp (WGS84 lon/lat)")
    gf.add_argument("--grid-coarsen", type=int, default=1, metavar="N",
                    help="average N x N blocks of cells before forecasting (e.g. 5 turns 0.05 deg into 0.25 deg); "
                         "~1,600 cells take about 40 s with three methods")
    gf.add_argument("--grid-output", type=Path, default=None, help="output folder (default: outputs/spatial_forecasts)")
    sb = ap.add_argument_group("Subseasonal (weekly) station forecasts (use with --stations; issued on --init-date)")
    sb.add_argument("--subseasonal", action="store_true", help="weekly / multi-week station forecasts with tercile probabilities")
    sb.add_argument("--init-date", default=None, metavar="YYYY-MM-DD", help="day the forecast is issued (data before it are used)")
    sb.add_argument("--lead-weeks", type=int, nargs=2, default=[2, 4], metavar=("FIRST", "LAST"),
                    help="target weeks after the init date, e.g. 2 4 or 3 6 (default 2 4)")
    sb.add_argument("--no-weekly-breakdown", action="store_true", help="only the aggregated window, not each single week")
    sb.add_argument("--pool-weeks", type=int, default=1, help="also train on init dates shifted by +/- this many weeks (default 1)")
    sb.add_argument("--recent-weeks", type=int, default=2, help="weeks before the init date used as the persistence predictor (default 2)")
    sb.add_argument("--min-train-years", type=int, default=10, help="earlier years needed before a subseasonal forecast is made (default 10)")
    st.add_argument("--station-dayfirst", action="store_true", help="dates like 01/02/2000 are day/month/year")
    st.add_argument("--station-monthfirst", action="store_true", help="dates like 01/02/2000 are month/day/year")
    st.add_argument("--station-date-format", default=None, metavar="FMT", help="explicit strptime format, e.g. '%%d.%%m.%%Y'")
    st.add_argument("--station-sep", default=None, help="column separator (default: detected from the header line)")
    st.add_argument("--station-output", type=Path, default=None, help="output folder (default: outputs/station_forecasts)")

    g = ap.add_argument_group("SPI / SPEI drought and wetness workflow (runs instead of the SST workflow "
                              "when --index is given)")
    g.add_argument("--index", choices=["spi", "spei", "both"], default=None, help="compute SPI, SPEI or both")
    g.add_argument("--input-dir", type=Path, default=None, help="folder with CHIRPS / tmin / tmax NetCDF files")
    g.add_argument("--output-dir", type=Path, default=None, help="output folder (default: settings.CLIM_OUTPUT_DIR)")
    g.add_argument("--scales", default=",".join(map(str, settings.SCALES)), help="accumulation periods, e.g. 1,3,6,12")
    g.add_argument("--pet-method", default=settings.PET_METHOD, help="PET method (only 'hargreaves')")
    g.add_argument("--prcp-var", default=None); g.add_argument("--tmin-var", default=None)
    g.add_argument("--tmax-var", default=None)
    g.add_argument("--start", default=None, help="analysis start, e.g. 1981-01")
    g.add_argument("--end", default=None, help="analysis end, e.g. 2026-07")
    g.add_argument("--ref-start", type=int, default=None, help="reference period start year")
    g.add_argument("--ref-end", type=int, default=None, help="reference period end year")
    g.add_argument("--region", default=settings.DEFAULT_REGION, help="region for statistics: " + ", ".join(settings.STUDY_REGIONS))
    g.add_argument("--season", default="JJAS", help="season for target-year maps and trends: " + ", ".join(settings.CLIM_SEASONS))
    g.add_argument("--target-years", default="", help="comma list, e.g. 1997,2015")
    g.add_argument("--classification", action="store_true", help="wetness/drought classes and area statistics")
    g.add_argument("--spatial-analysis", action="store_true", help="drought frequency/duration/severity, persistence, trends")
    g.add_argument("--make-maps", action="store_true", help="write maps")
    g.add_argument("--formats", default="png", help="map formats, e.g. png,pdf")
    g.add_argument("--geotiff", action="store_true", help="also write GeoTIFFs (needs rasterio)")
    g.add_argument("--teleconnection", action="store_true",
                   help="SST-rainfall lagged correlation, MCA, ENSO rainfall response, analogues/outlook "
                        "(can be run without --index)")
    g.add_argument("--sst-file", default=None, help="SST NetCDF (default: <input-dir>/" + settings.SST_FILE + ")")
    g.add_argument("--lags", default=",".join(map(str, settings.LAGS)), help="monthly lags, e.g. 0,1,2,3,4,5,6")
    g.add_argument("--lead", type=int, default=settings.OUTLOOK_LEAD, help="months between SST predictor window and season")
    g.add_argument("--predictor-months", type=int, default=settings.PREDICTOR_MONTHS)
    g.add_argument("--top-k", type=int, default=settings.ANALOGUE_TOP_K, help="number of analogue years")
    g.add_argument("--sst-res", type=float, default=settings.SST_RES_DEG, help="SST block-average resolution (deg)")
    g.add_argument("--rain-res", type=float, default=settings.RAIN_RES_DEG, help="rainfall block-average resolution (deg)")
    g.add_argument("--n-perm", type=int, default=settings.N_PERM, help="permutations / bootstrap resamples")
    g.add_argument("--fdr", type=float, default=settings.FDR_ALPHA, help="false-discovery-rate level for significance maps")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.grid_forecast:                     # gridded seasonal forecast
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
                            datefmt="%H:%M:%S")
        from src.preprocessing import grid_processing
        grid_processing.run(args)
        return
    if args.stations:                          # station CSV workflow
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
                            datefmt="%H:%M:%S")
        from src.preprocessing import station_processing
        station_processing.run(args)
        return
    if args.index or args.teleconnection:      # gridded SPI / SPEI and/or teleconnection workflow
        from src.drought_indices import workflow
        workflow.run(args)
        return
    if args.input:
        settings.INPUT_NC = args.input
    if args.output:
        settings.OUTPUT_DIR = args.output
    if args.force:
        settings.FORCE_RECOMPUTE = True
    if args.fast:
        settings.FAST_MODE, settings.RF_TREES, settings.CV_FOLDS, settings.REFIT_EVERY = True, 100, 2, 24
    for s in settings.SUBDIRS:
        settings.out(s)
    logfile = setup_logging()
    log = logging.getLogger("sst.main")
    t0 = time.time()
    log.info("Output directory: %s | log: %s", settings.OUTPUT_DIR, logfile)
    np.random.seed(settings.RANDOM_SEED)

    # heavy imports after logging is configured
    from src.data_ingestion import netcdf_loader
    from src.export import figures as viz
    from src.preprocessing import anomaly
    from src.teleconnections import index_statistics, sst_indices

    # ---------------- 1. data -> monthly regional SST ----------------
    if args.from_regional_csv:
        monthly = pd.read_csv(args.from_regional_csv, index_col=0, parse_dates=True)
        monthly.index = monthly.index.to_period("M").to_timestamp()
        extra_cols = [c for c in settings.EXTRA_REGIONS if c in monthly.columns]       # optional extra predictors in the CSV
        extra_monthly = monthly[extra_cols] if extra_cols else None
        monthly = monthly[[c for c in monthly.columns if c not in extra_cols]]
        log.info("Loaded monthly regional SST from %s (%s -> %s)", args.from_regional_csv,
                 monthly.index.min().strftime("%Y-%m"), monthly.index.max().strftime("%Y-%m"))
        meta = {"source": str(args.from_regional_csv)}
    else:
        extra_monthly = None
        ds = netcdf_loader.open_dataset(settings.INPUT_NC)
        meta = netcdf_loader.inspect_dataset(ds)
        monthly, report = sst_indices.regional_monthly_sst(ds, meta, settings.REGIONS)
        report.to_csv(settings.out("indices") / "data_quality_report_monthly.csv")
        if settings.EXTRA_REGIONS and not args.no_extra_predictors:
            try:
                extra_monthly, rep2 = sst_indices.regional_monthly_sst(ds, meta, settings.EXTRA_REGIONS, "extra_regional_monthly_sst")
                rep2.to_csv(settings.out("indices") / "data_quality_report_extra_regions.csv")
            except Exception as e:                       # e.g. an SST file that does not cover the Atlantic
                log.warning("Extra predictor regions skipped: %s", e)
        ds.close()
    (settings.out("indices") / "dataset_inspection.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---------------- 2. anomalies and indices ----------------
    anom, clim, baseline = anomaly.monthly_anomalies(monthly)
    idx = sst_indices.build_indices(anom)
    comb = sst_indices.save_indices(idx, monthly, clim, baseline)
    df = sst_indices.indices_frame(idx)
    if extra_monthly is not None and not extra_monthly.empty:
        sst_indices.save_predictors(sst_indices.extra_predictors(extra_monthly, df))
    log.info("Index record: %s -> %s (%d months); QA: max|DMI-(WIOD-EIOD)| = %.2e", df.index[0].strftime("%Y-%m"),
             df.index[-1].strftime("%Y-%m"), len(df), float((df["DMI"] - (df["WIOD"] - df["EIOD"])).abs().max()))

    # ---------------- 3. events and statistics ----------------
    events = sst_indices.save_events(idx)
    res = index_statistics.run_analysis(df, monthly, events)

    # ---------------- 4. forecasting ----------------
    hc = perf = sel = fcdf = None
    if not args.skip_forecast:
        from src.export import csv_export
        from src.forecasting import sst_forecast
        from src.verification import cross_validation, hindcast, skill_scores
        names = sst_forecast.available_models()
        log.info("Models in use: %s", names)
        hc, tuned, S = hindcast.run_hindcast(df, names)
        perf = skill_scores.evaluate(hc)
        sel, flags = cross_validation.select_models(perf, hc)
        hindcast.save_validation(hc, perf, sel, S)
        fcdf = sst_forecast.final_forecast(df, sel, tuned, hc, flags)
        csv_export.save_forecasts(fcdf)
        n_ok = int((sel["Validated_Skill"] == "yes").sum())
        log.info("Selected models show validated skill vs climatology at %d/%d (index, lead) combinations.", n_ok, len(sel))

    # ---------------- 5. figures ----------------
    viz.make_all(df, res, events, clim, monthly, baseline, hc, perf, sel, fcdf)

    log.info("Finished in %.1f min. Outputs in %s", (time.time() - t0) / 60, settings.OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:                       # a missing input: short message, no traceback on the console
        logging.getLogger("sst.main").error("%s", e)
        sys.exit(2)
    except Exception:
        logging.getLogger("sst.main").exception("Pipeline failed")
        sys.exit(1)
