"""Gridded SPI / SPEI workflow: computation, classification, spatial statistics, maps, reports.

Origin: climate_workflow.py."""
from __future__ import annotations

from pathlib import Path
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from src import settings
from src.data_ingestion import netcdf_loader
from src.drought_indices import index_kernels
from src.export import geotiff_export
from src.export.netcdf_export import make_dataset, make_map_dataset, save_netcdf
from src.spatial import forecast_maps, grid_statistics


log = logging.getLogger("clim.workflow")


SUBDIRS = [
    "climate_indices/spi/classified", "climate_indices/spei/classified",
    "spatial_analysis/drought_frequency", "spatial_analysis/drought_duration",
    "spatial_analysis/drought_severity", "spatial_analysis/wetness_frequency",
    "spatial_analysis/regional_statistics",
    "maps/spi", "maps/spei", "maps/drought", "maps/wetness",
    "statistics/trends", "statistics/validation", "reports",
]


def setup_logging(outdir: Path) -> Path:
    logfile = outdir / "reports" / f"climate_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)):
        h.setFormatter(fmt)
        root.addHandler(h)
    return logfile


def _season_months(season: str) -> list[int]:
    if season not in settings.CLIM_SEASONS:
        raise ValueError(f"Unknown season {season}; choose from {list(settings.CLIM_SEASONS)}")
    return settings.CLIM_SEASONS[season]


def compute_index(kind: str, scales: list[int], data: dict, ref_start: int, ref_end: int):
    """Return ({scale: array(time, lat, lon) float32}, diagnostics dict)."""
    P = data["prcp"]
    time_index = P.indexes["time"]
    nt, nlat, nlon = P.shape
    months = np.asarray(time_index.month)
    years = np.asarray(time_index.year)
    ref = (years >= ref_start) & (years <= ref_end)
    if ref.sum() < 20 * 12 * 0.75:
        log.warning("Reference period %d-%d covers only %d months of the record", ref_start, ref_end, ref.sum())
    dim = np.asarray(time_index.days_in_month)
    out = {s: np.full((nt, nlat, nlon), np.nan, dtype="float32") for s in scales}
    fail = np.full((nlat, nlon), np.nan, dtype="float32") if kind == "spei" else None
    lat_vals = P["lat"].values

    for i0 in range(0, nlat, settings.BLOCK_LAT):
        sl = slice(i0, min(nlat, i0 + settings.BLOCK_LAT))
        nb = sl.stop - sl.start
        p = P.isel(lat=sl).values.astype("float64").reshape(nt, -1)
        if kind == "spei":
            tmin = data["tmin"].isel(lat=sl).values.astype("float64").reshape(nt, -1)
            tmax = data["tmax"].isel(lat=sl).values.astype("float64").reshape(nt, -1)
            ra_m = index_kernels.monthly_mean_ra(lat_vals[sl])                       # (12, nb)
            ra = np.repeat(ra_m[months - 1][:, :, None], nlon, axis=2).reshape(nt, -1)   # matches (nt, nb, nlon) flatten
            pet = index_kernels.hargreaves_pet(tmin, tmax, ra, dim)
            base = p - pet                                                # climatic water balance (mm/month)
            del tmin, tmax, ra, pet
        else:
            base = p
        for s in scales:
            acc = index_kernels.accumulate(base, s)
            if kind == "spi":
                z = index_kernels.spi_from_accumulated(acc, months, ref, settings.MIN_POSITIVE_SAMPLES, settings.MIN_REF_SAMPLES,
                                           settings.SPI_CLIP)
            else:
                z = index_kernels.spei_from_accumulated(acc, months, ref, settings.MIN_REF_SAMPLES, settings.SPI_CLIP,
                                            settings.SPEI_NORMAL_FALLBACK)
                if s == scales[0]:
                    fail[sl, :] = index_kernels.spei_fit_failure_fraction(acc, months, ref, settings.MIN_REF_SAMPLES).reshape(nb, nlon)
            out[s][:, sl, :] = z.reshape(nt, nb, nlon).astype("float32")
        log.info("%s: rows %d-%d of %d done", kind.upper(), sl.start, sl.stop, nlat)
    return out, {"fit_failure_fraction": fail, "ref_start": ref_start, "ref_end": ref_end}


def process_index(kind, scale, z, like, args, outdir, summary, drought_rows):
    name = f"{kind}_{scale}"
    label = f"{kind.upper()}-{scale}"
    time_index = like.indexes["time"]
    lat, lon = like["lat"].values, like["lon"].values
    attrs = {"long_name": f"{label} standardized index", "units": "1",
             "reference_period": f"{args.ref_start}-{args.ref_end}",
             "distribution": settings.SPI_DISTRIBUTION if kind == "spi" else settings.SPEI_DISTRIBUTION,
             "pet_method": args.pet_method if kind == "spei" else "n/a"}
    save_netcdf(make_dataset(z, name, like, attrs), outdir / "climate_indices" / kind / f"{name}.nc")

    zr = z.astype("float64")
    codes = index_kernels.classify(zr.reshape(z.shape[0], -1)).reshape(z.shape)

    # ---- classification ----
    if args.classification:
        cattrs = {"long_name": f"{label} wetness/drought class", "flag_values": np.arange(1, 8, dtype="uint8"),
                  "flag_meanings": " ".join(n.replace(" ", "_") for n in index_kernels.CLASS_NAMES.values()),
                  "missing_code": 0,
                  "comment": "1 Extremely Dry ... 4 Near Normal ... 7 Extremely Wet; 0 = missing"}
        save_netcdf(make_dataset(codes, f"{name}_class", like, cattrs, "uint8"),
              outdir / "climate_indices" / kind / "classified" / f"{name}_class.nc", int_var=f"{name}_class")
        sub_c = grid_statistics.subset_region(make_dataset(codes, "c", like, {}, "uint8")["c"], args.region)
        df = grid_statistics.category_area_timeseries(sub_c.values, time_index, sub_c["lat"].values, sub_c["lon"].values)
        agg = grid_statistics.aggregate_category_stats(df)
        rd = outdir / "spatial_analysis" / "regional_statistics"
        for part, d in agg.items():
            d.to_csv(rd / f"{name}_{args.region}_category_area_{part}.csv")
        freq = {}
        for season in settings.CLIM_SEASONS:
            fm = grid_statistics.category_frequency_maps(codes, time_index, season)
            for i, c in enumerate(range(1, 8)):
                freq[f"freq_{season}_{index_kernels.CLASS_NAMES[c].replace(' ', '_')}"] = fm[i]
        save_netcdf(make_map_dataset(freq, like, {"long_name": f"{label} class frequency (% of time steps)"}),
              rd / f"{name}_class_frequency_maps.nc")
        if args.make_maps:
            forecast_maps.stacked_area_plot(df, f"{label} – share of {args.region} area by class",
                                   outdir / "maps" / kind / f"{name}_area_by_class.png")
            last = int(np.where(np.isfinite(zr).any(axis=(1, 2)))[0][-1])
            forecast_maps.map_field(codes[last], lat, lon, f"{label} classes – {time_index[last]:%B %Y}",
                           outdir / "maps" / kind / f"{name}_class_last", kind="class", formats=args.formats)

    # ---- regional summary (always) ----
    zda = make_dataset(z, name, like, {})[name]
    for rs in grid_statistics.regional_statistics(zda, label).to_dict("records"):
        summary.append(rs)

    # ---- spatial analysis ----
    if args.spatial_analysis:
        sp = {}
        dr = grid_statistics.spell_maps(zr, settings.DROUGHT_THRESHOLD, below=True)
        we = grid_statistics.spell_maps(zr, settings.WET_THRESHOLD, below=False)
        pers_d = grid_statistics.persistence_map(zr, settings.DROUGHT_THRESHOLD, True)
        pers_w = grid_statistics.persistence_map(zr, settings.WET_THRESHOLD, False)
        slope, pval = grid_statistics.linear_trend_map(zr, time_index, args.season if args.season in settings.CLIM_SEASONS else "ALL")
        sd = outdir / "spatial_analysis"
        save_netcdf(make_map_dataset({"drought_frequency_pct": dr["frequency_pct"], "n_drought_events": dr["n_events"],
                            "drought_persistence_pct": pers_d},
                           like, {"long_name": f"{label} drought frequency (index <= {settings.DROUGHT_THRESHOLD})"}),
              sd / "drought_frequency" / f"{name}_drought_frequency.nc")
        save_netcdf(make_map_dataset({"mean_duration_months": dr["mean_duration"], "max_duration_months": dr["max_duration"]},
                           like, {"long_name": f"{label} drought spell duration"}),
              sd / "drought_duration" / f"{name}_drought_duration.nc")
        save_netcdf(make_map_dataset({"mean_severity": dr["mean_severity"], "max_severity": dr["max_severity"],
                            "mean_intensity": dr["mean_intensity"]},
                           like, {"long_name": f"{label} drought severity (sum |index| per spell)"}),
              sd / "drought_severity" / f"{name}_drought_severity.nc")
        save_netcdf(make_map_dataset({"wetness_frequency_pct": we["frequency_pct"], "n_wet_events": we["n_events"],
                            "mean_duration_months": we["mean_duration"], "max_duration_months": we["max_duration"],
                            "wetness_persistence_pct": pers_w},
                           like, {"long_name": f"{label} wetness frequency (index >= {settings.WET_THRESHOLD})"}),
              sd / "wetness_frequency" / f"{name}_wetness_frequency.nc")
        save_netcdf(make_map_dataset({"trend_per_decade": slope, "p_value": pval}, like,
                           {"long_name": f"{label} linear trend (units/decade), season={args.season}"}),
              outdir / "statistics" / "trends" / f"{name}_trend.nc")
        for r in settings.STUDY_REGIONS:
            try:
                sub = grid_statistics.subset_region(make_map_dataset({"f": dr["frequency_pct"]}, like)["f"], r)
            except ValueError:
                continue
            drought_rows.append({"Index": label, "Region": r,
                                 "mean_drought_frequency_pct": round(float(np.nanmean(sub.values)), 3),
                                 "max_cell_drought_frequency_pct": round(float(np.nanmax(sub.values)), 3)})
        if args.geotiff:
            geotiff_export.write_geotiff(dr["frequency_pct"], lat, lon, sd / "drought_frequency" / f"{name}_drought_frequency.tif")
            geotiff_export.write_geotiff(z[-1], lat, lon, outdir / "climate_indices" / kind / f"{name}_last.tif")
        if args.make_maps:
            md = outdir / "maps" / "drought"
            forecast_maps.map_field(dr["frequency_pct"], lat, lon, f"{label} drought frequency", md / f"{name}_frequency",
                           kind="sequential", label="% of months (index ≤ -1)", formats=args.formats)
            forecast_maps.map_field(dr["mean_duration"], lat, lon, f"{label} mean drought duration", md / f"{name}_duration",
                           kind="sequential", label="months", formats=args.formats, cmap="PuRd")
            forecast_maps.map_field(dr["mean_severity"], lat, lon, f"{label} mean drought severity",
                           md / f"{name}_severity", kind="sequential", label="Σ|index| per spell",
                           formats=args.formats, cmap="OrRd")
            forecast_maps.map_field(we["frequency_pct"], lat, lon, f"{label} wetness frequency",
                           outdir / "maps" / "wetness" / f"{name}_frequency", kind="sequential",
                           label="% of months (index ≥ +1)", formats=args.formats, cmap="GnBu")
            forecast_maps.map_field(slope, lat, lon, f"{label} trend ({args.season})",
                           outdir / "maps" / kind / f"{name}_trend", kind="diverging",
                           label="index units / decade", formats=args.formats)

    # ---- maps for last month and target years ----
    if args.make_maps:
        md = outdir / "maps" / kind
        valid_t = np.where(np.isfinite(zr).any(axis=(1, 2)))[0]
        if len(valid_t):
            t = int(valid_t[-1])
            forecast_maps.map_field(z[t], lat, lon, f"{label} – {time_index[t]:%B %Y}", md / f"{name}_last", kind="index",
                           formats=args.formats, subtitle=f"reference {args.ref_start}-{args.ref_end}")
        for y in args.target_years:
            sel = np.where((time_index.year == y) & np.isin(time_index.month, _season_months(args.season)))[0]
            if len(sel) == 0:
                log.warning("No data for target year %d season %s", y, args.season)
                continue
            f = np.nanmean(zr[sel], axis=0)
            forecast_maps.map_field(f, lat, lon, f"{label} – {args.season} {y}", md / f"{name}_{args.season}_{y}",
                           kind="index", formats=args.formats,
                           subtitle=f"mean of monthly {label} values, {args.season}")
            forecast_maps.map_field(index_kernels.classify(f).astype("float"), lat, lon, f"{label} class – {args.season} {y}",
                           md / f"{name}_{args.season}_{y}_class", kind="class", formats=args.formats)


def validate(kind, z_by_scale, data, args, outdir):
    """Reference-period sanity checks plus a comparison with the `climate_indices` package if installed."""
    rows = []
    P = data["prcp"]
    years = P.indexes["time"].year
    ref = (years >= args.ref_start) & (years <= args.ref_end)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    for s, z in z_by_scale.items():
        zr = z[ref]
        rows.append({"Index": f"{kind.upper()}-{s}", "check": "reference-period mean (should be ~0)",
                     "value": float(np.nanmean(zr))})
        rows.append({"Index": f"{kind.upper()}-{s}", "check": "reference-period std (should be ~1)",
                     "value": float(np.nanstd(zr))})
        rows.append({"Index": f"{kind.upper()}-{s}", "check": "fraction of valid cells*months",
                     "value": float(np.isfinite(z).mean())})
    try:
        from climate_indices import compute, indices
    except Exception:
        rows.append({"Index": kind.upper(), "check": "external comparison", "value": np.nan,
                     "note": "package 'climate_indices' not installed; comparison skipped"})
        pd.DataFrame(rows).to_csv(outdir / "statistics" / "validation" / f"{kind}_validation.csv", index=False)
        return
    if kind == "spi":                                   # compare 30 random cells for each scale
        ny, nx = P.shape[1:]
        cells = [(int(rng.integers(ny)), int(rng.integers(nx))) for _ in range(30)]
        for s, z in z_by_scale.items():
            diffs = []
            for (i, j) in cells:
                ser = P.values[:, i, j].astype("float64")
                if not np.isfinite(ser).all() or (ser > 0).sum() < 30:
                    continue
                ext = indices.spi(ser, s, indices.Distribution.gamma, int(years[0]), args.ref_start,
                                  args.ref_end, compute.Periodicity.monthly)
                m = np.isfinite(ext) & np.isfinite(z[:, i, j])
                if m.sum() > 12:
                    diffs.append(np.abs(ext[m] - z[m, i, j]).mean())
            rows.append({"Index": f"SPI-{s}", "check": "mean |difference| vs climate_indices.spi",
                         "value": float(np.mean(diffs)) if diffs else np.nan, "note": f"{len(diffs)} cells"})
    pd.DataFrame(rows).to_csv(outdir / "statistics" / "validation" / f"{kind}_validation.csv", index=False)


def run(args) -> None:
    outdir = Path(args.output_dir or settings.CLIM_OUTPUT_DIR)
    for s in SUBDIRS:
        (outdir / s).mkdir(parents=True, exist_ok=True)
    logfile = setup_logging(outdir)
    t0 = time.time()
    kinds = [] if not args.index else (["spi", "spei"] if args.index == "both" else [args.index])
    scales = [int(s) for s in str(args.scales).split(",") if s.strip()]
    args.target_years = [int(y) for y in str(args.target_years).split(",") if y.strip()] if args.target_years else []
    args.formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    args.ref_start = args.ref_start or settings.REF_START
    args.ref_end = args.ref_end or settings.REF_END
    args.ref_start_year0, args.ref_end_year0 = args.ref_start, args.ref_end
    if args.region not in settings.STUDY_REGIONS:
        raise ValueError(f"Unknown region {args.region}; choose from {list(settings.STUDY_REGIONS)}")
    if args.pet_method.lower() != "hargreaves":
        raise ValueError("Only the Hargreaves-Samani PET method is implemented (--pet-method hargreaves)")
    log.info("Output: %s | log: %s", outdir, logfile)
    log.info("Indices=%s scales=%s ref=%d-%d region=%s season=%s", kinds, scales, args.ref_start, args.ref_end,
             args.region, args.season)

    in_dir = Path(args.input_dir or settings.CLIM_INPUT_DIR)
    summary, drought_rows, diag, qa = [], [], {}, {}
    if kinds:
        data, qa = netcdf_loader.load_inputs(in_dir, need_temperature="spei" in kinds, prcp_var=args.prcp_var,
                                  tmin_var=args.tmin_var, tmax_var=args.tmax_var, start=args.start, end=args.end)
        (outdir / "reports" / "data_quality_report.json").write_text(json.dumps(qa, indent=2, default=str))
        like = data["prcp"]
        for kind in kinds:
            zs, d = compute_index(kind, scales, data, args.ref_start, args.ref_end)
            diag[kind] = d
            if kind == "spei" and d["fit_failure_fraction"] is not None:
                ff = d["fit_failure_fraction"]
                log.info("SPEI log-logistic fit invalid for %.2f%% of cell-months of the first scale (%s); see "
                         "statistics/validation/spei_fit_failures.nc", 100 * float(np.nanmean(ff)),
                         "normal fallback used" if settings.SPEI_NORMAL_FALLBACK else "set to NaN")
                save_netcdf(make_map_dataset({"fit_failure_fraction": ff}, like), outdir / "statistics" / "validation" /
                      "spei_fit_failures.nc")
            for s in scales:
                process_index(kind, s, zs[s], like, args, outdir, summary, drought_rows)
            validate(kind, zs, data, args, outdir)
            del zs
        del data

    if args.teleconnection:
        from src.teleconnections import correlation_maps
        correlation_maps.run(args, outdir)

    if summary:
        pd.DataFrame(summary).to_csv(outdir / "reports" / "climate_indices_summary.csv", index=False)
    if drought_rows:
        pd.DataFrame(drought_rows).to_csv(outdir / "reports" / "drought_frequency.csv", index=False)
    write_report(outdir, args, kinds, scales, qa, time.time() - t0)
    log.info("Finished in %.1f min", (time.time() - t0) / 60)


def write_report(outdir: Path, args, kinds, scales, qa, seconds):
    lines = [
        "SPI / SPEI ANALYSIS REPORT", "=" * 60,
        f"Run time: {seconds / 60:.1f} min", f"Indices: {kinds}  Scales: {scales}",
        f"Reference period (all indices): {args.ref_start}-{args.ref_end}",
        f"SPI distribution: {settings.SPI_DISTRIBUTION} with mixed zero-probability term; "
        f"SPEI distribution: {settings.SPEI_DISTRIBUTION} (unbiased PWM); PET: Hargreaves-Samani",
        f"Index values capped at |{settings.SPI_CLIP}|.", "",
        "Classification (codes): 1 Extremely Dry (Z<=-2); 2 Severely Dry (-2<Z<=-1.5); 3 Moderately Dry "
        "(-1.5<Z<=-1); 4 Near Normal (-1<Z<1); 5 Moderately Wet (1<=Z<1.5); 6 Severely Wet (1.5<=Z<2); "
        "7 Extremely Wet (Z>=2); 0 missing.", "", "Input data quality:"]
    for k_, v in qa.items():
        lines.append(f"  {k_}: file={v['file']} var={v['variable']} units_in='{v['units_in']}' -> {v['units_out']}; "
                     f"period={v['analysis_period']}; shape(time,lat,lon)={v['shape']}; "
                     f"missing months={len(v.get('missing_months', []))}; ~{100 * v['missing_fraction']:.1f}% NaN")
    lines += ["", "Limitations:",
              "  * SPI zero-handling uses q + (1-q)G(x); in very arid cells with many zero months SPI is not "
              "meaningful (cells with < %d non-zero reference values are set to NaN)." % settings.MIN_POSITIVE_SAMPLES,
              "  * SPEI cell-months where the log-logistic fit is invalid (non-positive L-skewness) are standardized with "
              "a normal fit (spei_normal_fallback in config/forecast_settings.yaml) or are NaN; their share is mapped in "
              "statistics/validation/spei_fit_failures.nc.",
              "  * Trend p-values assume independent annual values (no autocorrelation correction)."]
    if args.teleconnection:
        lines += ["", "Teleconnection: SST block-averaged to ~%.1f deg, rainfall to ~%.1f deg; anomalies from the "
                  "reference period %d-%d; trends not removed; significance uses effective sample size (lag-1 "
                  "autocorrelation) and Benjamini-Hochberg FDR (q=%.2f); MCA significance by %d year-order "
                  "permutations. Analogue/regression outlooks are statistical products validated only by "
                  "leave-one-year-out hindcasts (see forecasting/regression/loo_skill_by_lead_*.csv); they are not "
                  "operational forecasts." % (args.sst_res, args.rain_res, args.ref_start, args.ref_end, args.fdr,
                                              args.n_perm)]
    (outdir / "reports" / "analysis_report.txt").write_text("\n".join(lines))
