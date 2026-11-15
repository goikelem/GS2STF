"""SST-rainfall teleconnection workflow (lagged correlation, EOF, MCA, pattern regression,
ENSO composites, analogue years, MOS calibration, forecast skill).

Run with ``python main.py --teleconnection ...``.  Everything is computed from gridded data; the
rainfall response is evaluated cell by cell and as an area-weighted regional mean.  Outlooks are
statistical, validated by leave-one-out hindcasts, and are NOT operational forecasts.
"""
from __future__ import annotations

import logging
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

import climate_data as cd
import climate_maps as cmap
import climate_workflow as cw
import config
import data_processing as dp
import indices_calculation as ic
import spatial_analysis as sa
import teleconnection as tc

log = logging.getLogger("clim.tele")

SUBDIRS = [
    "sst_analysis/sst_anomalies", "sst_analysis/nino34", "sst_analysis/enso", "sst_analysis/teleconnection",
    "forecasting/analogue_years", "forecasting/regression", "forecasting/seasonal_outlooks",
    "maps/sst", "maps/teleconnection", "statistics/correlation", "statistics/regression", "reports",
]


class _Grid:
    """Minimal lat/lon holder accepted by cw._map_dataset."""
    def __init__(self, lat, lon):
        self._d = {"lat": types.SimpleNamespace(values=np.asarray(lat)),
                   "lon": types.SimpleNamespace(values=np.asarray(lon))}

    def __getitem__(self, k):
        return self._d[k]


# ----------------------------------------------------------------------------
# data helpers
# ----------------------------------------------------------------------------
def load_sst_coarse(path: Path, start, end, deg: float):
    """Read the global SST file month by month, convert to degC, block-average to ~``deg`` degrees.

    Reading month by month keeps memory small for daily 0.25-degree files.  Longitudes are returned
    on 0..360 and latitudes ascending."""
    import xarray as xr
    path = Path(path)
    ds = dp.open_dataset(path)
    meta = dp.inspect_dataset(ds)
    var, tn, latn, lonn = meta["variable"], meta["time"], meta["lat"], meta["lon"]
    res = meta["lat_res"] or 1.0
    f = max(1, int(round(deg / res)))
    tvals = pd.DatetimeIndex(ds[tn].values)
    months = pd.period_range(max(tvals.min(), pd.Timestamp(start)), min(tvals.max(), pd.Timestamp(end)), freq="M")
    log.info("SST: %s | %d months | block factor %d (%.2f deg -> ~%.1f deg)", var, len(months), f, res, res * f)
    pieces = []
    for p in months:
        da = ds[var].sel({tn: slice(p.start_time, p.end_time)})
        if da.sizes[tn] == 0:
            continue
        da = da.rename({tn: "time", latn: "lat", lonn: "lon"}).load().astype("float32")
        if meta["kelvin"]:
            da = da - 273.15
        da = da.where((da > -2.5) & (da < 40))
        if f > 1:
            da = da.coarsen(lat=f, lon=f, boundary="trim").mean()
        da = da.mean("time", skipna=True, keep_attrs=False).expand_dims(time=[p.start_time])
        pieces.append(da)
    ds.close()
    sst = xr.concat(pieces, dim="time")
    sst = sst.assign_coords(lon=(sst["lon"] % 360)).sortby("lon").sortby("lat")
    sst.name = "sst"
    sst.attrs.update({"units": "degC"})
    return sst, meta


def block_mean(arr: np.ndarray, lat, lon, f: int):
    """NaN-aware block mean of (n, ny, nx) by integer factor f (trim edges); returns arr, lat, lon."""
    if f <= 1:
        return arr, np.asarray(lat), np.asarray(lon)
    n, ny, nx = arr.shape
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    a = arr[:, :ny2, :nx2].reshape(n, ny2 // f, f, nx2 // f, f)
    ok = np.isfinite(a)
    cnt = ok.sum(axis=(2, 4))
    with np.errstate(all="ignore"):
        out = np.where(ok, a, 0).sum(axis=(2, 4)) / cnt
    out = np.where(cnt >= 0.5 * f * f, out, np.nan)
    la = np.asarray(lat)[:ny2].reshape(-1, f).mean(1)
    lo = np.asarray(lon)[:nx2].reshape(-1, f).mean(1)
    return out, la, lo


def read_enso_phases(excel: Path, season: str):
    """{year: 'EL'|'LA'|'N'} from sheet ENSO_<season> of the user's Excel file, or None."""
    sheet = f"ENSO_{season}"
    try:
        names = pd.ExcelFile(excel).sheet_names
        if sheet not in names:
            return None
        df = pd.read_excel(excel, sheet_name=sheet)
        df.columns = df.columns.astype(str).str.strip()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Tkey"] = df["Tkey"].astype(str).str.strip().str.upper()
        df = df.dropna(subset=["Year"])
        return {int(y): t for y, t in zip(df["Year"], df["Tkey"])}
    except Exception as e:
        log.warning("Could not read ENSO phases from %s (%s); using Nino3.4 from the SST file", excel, e)
        return None


def wmean(M: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean over columns of (n, N), NaN cells excluded per row."""
    ok = np.isfinite(M)
    with np.errstate(all="ignore"):
        return (np.where(ok, M, 0) * w).sum(1) / (ok * w).sum(1)


def tercile_category(v, ref):
    lo, hi = np.nanquantile(ref, [1 / 3, 2 / 3])
    return np.where(np.isnan(v), "", np.where(v <= lo, "Below normal", np.where(v >= hi, "Above normal", "Near normal")))


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def run(args) -> None:
    t0 = time.time()
    outdir = Path(args.output_dir or config.CLIM_OUTPUT_DIR)
    for s in SUBDIRS:
        (outdir / s).mkdir(parents=True, exist_ok=True)
    logfile = cw.setup_logging(outdir)
    in_dir = Path(args.input_dir or config.CLIM_INPUT_DIR)
    ref_start = args.ref_start or config.REF_START
    ref_end = args.ref_end or config.REF_END
    season = args.season
    if season not in config.TELE_SEASONS:
        raise ValueError(f"Unknown season {season}: {list(config.TELE_SEASONS)}")
    pred_season = args.predictor_season or config.DEFAULT_PREDICTOR.get(season, "ALL")
    lags = [int(x) for x in str(args.lags).split(",") if x.strip() != ""]
    target_years = [int(y) for y in str(args.target_years).split(",") if y.strip()] if args.target_years else []
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    nperm, k_an = int(args.n_perm), int(args.analogue_k)
    log.info("Teleconnection: target=%s predictor=%s region=%s lags=%s ref=%d-%d | log %s", season, pred_season,
             args.region, lags, ref_start, ref_end, logfile)

    # ------------------------------------------------------------ data
    sst_path = Path(args.sst_file) if args.sst_file else in_dir / config.SST_FILE
    if not sst_path.exists():
        raise FileNotFoundError(sst_path)
    start, end = args.start or config.ANALYSIS_START, args.end or config.ANALYSIS_END
    sst, _ = load_sst_coarse(sst_path, start, end, config.SST_COARSEN_DEG)
    rain, rqa = cd.load_variable(in_dir / config.PRCP_FILE, "prcp", args.prcp_var or config.PRCP_VAR)
    rain = sa.subset_region(rain, args.region)
    t0_, t1_ = max(sst.indexes["time"][0], rain.indexes["time"][0], pd.Timestamp(start)), \
        min(sst.indexes["time"][-1], rain.indexes["time"][-1], pd.Timestamp(end))
    rain, _ = cd._full_monthly(rain, t0_, t1_)
    sst = sst.reindex(time=rain.indexes["time"])
    tidx = rain.indexes["time"]
    T = len(tidx)
    months, years = np.asarray(tidx.month), np.asarray(tidx.year)
    ref = (years >= ref_start) & (years <= ref_end)
    slat, slon = sst["lat"].values, sst["lon"].values
    nS = (len(slat), len(slon))
    log.info("Common period %s -> %s (%d months) | SST grid %dx%d | rain grid %dx%d (%s)", tidx[0].strftime("%Y-%m"),
             tidx[-1].strftime("%Y-%m"), T, *nS, rain.sizes["lat"], rain.sizes["lon"], args.region)

    # ------------------------------------------------------------ SST anomalies + Nino3.4 + ENSO
    S = sst.values.astype("float64").reshape(T, -1)
    S_an = tc.monthly_anomalies(S, months, ref)
    cw._save(cw._dataset(S_an.reshape((T,) + nS).astype("float32"), "sst_anomaly", sst,
                         {"long_name": "SST anomaly", "units": "degC",
                          "reference_period": f"{ref_start}-{ref_end}"}),
             outdir / "sst_analysis" / "sst_anomalies" / "sst_anomaly_monthly.nc")
    wlat = np.cos(np.deg2rad(slat))
    w2d = np.repeat(wlat[:, None], len(slon), axis=1)
    reg = config.REGIONS["R34"]
    lat_m, lon_m = dp.region_masks(slat, slon, reg["lat"], reg["lon"])
    box = (lat_m[:, None] & lon_m[None, :]).ravel()
    nino = pd.Series(wmean(S_an[:, box], w2d.ravel()[box]), index=tidx, name="Nino34_anomaly")
    nino_oni = nino.rolling(3, center=True, min_periods=3).mean().rename("ONI_3mo")
    pd.concat([nino, nino_oni], axis=1).round(4).to_csv(outdir / "sst_analysis" / "nino34" / "nino34_monthly.csv",
                                                       index_label="Date")
    ev, ph = ic.enso_events(nino.dropna())
    ev.to_csv(outdir / "sst_analysis" / "enso" / "ENSO_events.csv", index=False)
    ph.rename("ENSO_phase").to_csv(outdir / "sst_analysis" / "enso" / "ENSO_monthly_phase.csv", index_label="Date")
    summ = ic.event_summary(ev, len(nino.dropna()) / 12, "ENSO (Nino3.4)")
    summ.to_csv(outdir / "reports" / "enso_frequency.csv", index=False)
    log.info("Nino3.4: %d ENSO events; ocean cells in box: %d", len(ev), int(np.isfinite(S[:, box]).any(0).sum()))

    # ------------------------------------------------------------ rainfall
    R = rain.values.astype("float64").reshape(T, -1)
    r_lat, r_lon = rain["lat"].values, rain["lon"].values
    nR = (len(r_lat), len(r_lon))
    area_r = sa.area_grid(rain).ravel()
    rain_reg = wmean(R, area_r)
    y_an = tc.monthly_anomalies(rain_reg[:, None], months, ref)[:, 0]

    # ------------------------------------------------------------ 1. lagged correlation (monthly)
    tmask = np.isin(months, config.TELE_SEASONS[season])
    res = tc.lagged_correlation(S_an, y_an, tmask, lags, min_n=20)
    fields = {}
    rows = []
    for L in lags:
        r_, p_ = res[L]["r"], res[L]["p"]
        sig = tc.fdr_bh(p_, config.FDR_ALPHA)
        fields.update({f"r_lag{L}": r_, f"p_lag{L}": p_, f"n_lag{L}": res[L]["n"], f"neff_lag{L}": res[L]["neff"],
                       f"fdr_sig_lag{L}": sig.astype("float32")})
        cs = tc.correlation_methods(nino.values[np.where(tmask)[0] - L] if True else None,
                                    y_an[np.where(tmask)[0]], nboot=1000) if (np.where(tmask)[0] - L >= 0).all() else \
            tc.correlation_methods(nino.values[np.where(tmask)[0][np.where(tmask)[0] - L >= 0] - L],
                                   y_an[np.where(tmask)[0][np.where(tmask)[0] - L >= 0]], nboot=1000)
        rows.append({"lag_months": L, "season": season, "region": args.region, **cs})
        if args.make_maps:
            cmap.map_field(r_.reshape(nS), slat, slon,
                           f"SST(t-{L}) vs {args.region} rainfall in {season}: correlation",
                           outdir / "maps" / "teleconnection" / f"lagcorr_{season}_lag{L}", kind="diverging",
                           label="Pearson r", vmax=0.6, formats=formats, stipple=sig.reshape(nS),
                           subtitle=f"dots: FDR-significant (alpha={config.FDR_ALPHA}), N_eff-adjusted")
    rmax, lagmax, pmax = tc.best_lag(res)
    fields.update({"r_at_best_lag": rmax, "best_lag_months": lagmax, "p_at_best_lag": pmax})
    cw._save(cw._map_dataset({k_: v.reshape(nS) for k_, v in fields.items()}, sst,
                             {"long_name": f"Lagged SST-rainfall correlation, target {season}, {args.region}"}),
             outdir / "sst_analysis" / "teleconnection" / f"lagged_correlation_{season}_{args.region}.nc")
    pd.DataFrame(rows).round(4).to_csv(outdir / "statistics" / "correlation" /
                                       f"nino34_vs_regional_rain_monthly_lags_{season}.csv", index=False)
    if args.make_maps:
        sigbest = tc.fdr_bh(pmax, config.FDR_ALPHA)
        cmap.map_field(rmax.reshape(nS), slat, slon, f"Maximum |r| over lags {min(lags)}-{max(lags)}",
                       outdir / "maps" / "teleconnection" / f"lagcorr_{season}_max", kind="diverging",
                       label="r at the lag of maximum |r|", vmax=0.6, formats=formats, stipple=sigbest.reshape(nS))
        cmap.map_field(lagmax.reshape(nS), slat, slon, "Lag of maximum |r| (months)",
                       outdir / "maps" / "teleconnection" / f"lagcorr_{season}_bestlag", kind="sequential",
                       label="months", cmap="viridis", vmin=min(lags), vmax=max(lags), formats=formats)

    # ------------------------------------------------------------ seasonal aggregates
    yp, Sp = tc.seasonal_aggregate(S_an, years, months, config.TELE_SEASONS[pred_season], "mean")
    yt, Rt = tc.seasonal_aggregate(R, years, months, config.TELE_SEASONS[season], "sum")
    _, nino_p = tc.seasonal_aggregate(nino.values[:, None], years, months, config.TELE_SEASONS[pred_season], "mean")
    _, nino_t = tc.seasonal_aggregate(nino.values[:, None], years, months, config.TELE_SEASONS[season], "mean")
    nino_p, nino_t = nino_p[:, 0], nino_t[:, 0]
    ref_years_t = (yt >= ref_start) & (yt <= ref_end)
    clim_t = np.nanmean(Rt[ref_years_t], axis=0)
    Ra = Rt - clim_t                                          # seasonal anomaly (mm)
    Rpct = 100 * Ra / np.where(clim_t > 0, clim_t, np.nan)    # % of normal departure
    rt_reg = wmean(Rt, area_r)                                # regional seasonal total (mm)
    clim_reg = np.nanmean(rt_reg[ref_years_t])
    rt_anom = rt_reg - clim_reg
    common = np.intersect1d(yp, yt)
    ip, it = np.searchsorted(yp, common), np.searchsorted(yt, common)
    log.info("Years with predictor (%s): %d | with target (%s): %d | common: %d", pred_season, len(yp), season,
             len(yt), len(common))
    if len(common) < 15:
        log.warning("Only %d common years: teleconnection statistics will be weak", len(common))

    table = pd.DataFrame({"Year": yt, f"{season}_total_mm": rt_reg, "anomaly_mm": rt_anom,
                          "pct_of_normal": 100 * rt_reg / clim_reg,
                          "tercile": tercile_category(rt_reg, rt_reg[ref_years_t])})
    table["Nino34_target_season"] = nino_t[:len(yt)] if len(nino_t) == len(yt) else np.nan
    table.round(3).to_csv(outdir / "sst_analysis" / "enso" / f"regional_rainfall_{season}_{args.region}.csv",
                          index=False)

    # ------------------------------------------------------------ 2. correlation maps (seasonal) + stats
    ok_years = np.isfinite(rt_anom[it]) & np.isfinite(nino_p[ip])
    corr_rows = []
    cs = tc.correlation_methods(nino_p[ip], rt_anom[it], nboot=2000)
    corr_rows.append({"predictor": f"Nino3.4 {pred_season}", "target": f"{args.region} rainfall {season}", **cs})
    pd.DataFrame(corr_rows).round(4).to_csv(outdir / "statistics" / "correlation" /
                                            f"nino34_{pred_season}_vs_rain_{season}.csv", index=False)
    lin = tc.linear_regression(nino_p[ip], rt_anom[it])
    pd.DataFrame([{"predictor": f"Nino3.4 {pred_season}", **lin}]).round(4).to_csv(
        outdir / "statistics" / "regression" / f"nino34_{pred_season}_to_rain_{season}.csv", index=False)
    r1 = float(tc.lag1_autocorr(rt_anom[it][:, None])[0])
    r_s, n_s = tc.pearson_map(Sp[ip], rt_anom[it], 10)
    r1x = tc.lag1_autocorr(Sp[ip])
    p_s = tc.r_to_p(r_s, tc.effective_n(n_s, r1x, r1))
    sig_s = tc.fdr_bh(p_s, config.FDR_ALPHA)
    cw._save(cw._map_dataset({"r": r_s.reshape(nS), "p": p_s.reshape(nS), "fdr_sig": sig_s.reshape(nS).astype("float32")},
                             sst, {"long_name": f"SST {pred_season} vs {args.region} rainfall {season} (seasonal)"}),
             outdir / "sst_analysis" / "teleconnection" / f"seasonal_corr_{pred_season}_to_{season}.nc")

    # pattern regression of gridded rainfall on Nino3.4 (predictor season)
    slope, r_g, p_g = tc.regression_map(Ra[it], nino_p[ip], 10)
    sig_g = tc.fdr_bh(p_g, config.FDR_ALPHA)
    cw._save(cw._map_dataset({"slope_mm_per_degC": slope.reshape(nR), "r": r_g.reshape(nR), "p": p_g.reshape(nR),
                              "fdr_sig": sig_g.reshape(nR).astype("float32")}, rain,
                             {"long_name": f"Rainfall {season} regressed on Nino3.4 {pred_season}"}),
             outdir / "forecasting" / "regression" / f"pattern_regression_{pred_season}_to_{season}.nc")

    if args.make_maps:
        cmap.map_field(r_s.reshape(nS), slat, slon, f"SST {pred_season} vs {args.region} rainfall {season}",
                       outdir / "maps" / "teleconnection" / f"seasonal_corr_{pred_season}_to_{season}",
                       kind="diverging", label="Pearson r", vmax=0.7, formats=formats, stipple=sig_s.reshape(nS))
        cmap.map_field(slope.reshape(nR), r_lat, r_lon, f"Rainfall {season} per °C of Nino3.4 {pred_season}",
                       outdir / "maps" / "teleconnection" / f"pattern_regression_{pred_season}_to_{season}",
                       kind="diverging", label="mm per °C", formats=formats, stipple=sig_g.reshape(nR))

    # ------------------------------------------------------------ 3. EOF and MCA
    wS = np.sqrt(np.clip(w2d.ravel(), 0, None))
    e = tc.eof(Sp[ip], wS, nmodes=config.MCA_MODES)
    cw._save(cw._map_dataset({f"eof{i + 1}": e["patterns"][i].reshape(nS) for i in range(len(e["var_frac"]))},
                             sst, {"long_name": f"EOFs of SST anomaly {pred_season} (degC per std of PC)"}),
             outdir / "sst_analysis" / "teleconnection" / f"sst_eof_{pred_season}.nc")
    pd.DataFrame(e["pcs"], index=common, columns=[f"PC{i + 1}" for i in range(e["pcs"].shape[1])]).assign(
        **{}).to_csv(outdir / "sst_analysis" / "teleconnection" / f"sst_eof_{pred_season}_pcs.csv", index_label="Year")
    pd.DataFrame({"mode": np.arange(1, len(e["var_frac"]) + 1), "variance_fraction": e["var_frac"]}).round(4).to_csv(
        outdir / "sst_analysis" / "teleconnection" / f"sst_eof_{pred_season}_variance.csv", index=False)

    f_r = max(1, int(round(config.RAIN_COARSEN_DEG / (abs(np.median(np.diff(r_lat))) if len(r_lat) > 1 else 1))))
    Ra_c, rc_lat, rc_lon = block_mean(Ra[it].reshape((len(common),) + nR), r_lat, r_lon, f_r)
    nRc = Ra_c.shape[1:]
    wR = np.sqrt(np.repeat(np.cos(np.deg2rad(rc_lat))[:, None], len(rc_lon), axis=1).ravel())
    m = tc.mca_permutation(Sp[ip], Ra_c.reshape(len(common), -1), wS, wR, nperm=nperm, nmodes=config.MCA_MODES)
    modes = pd.DataFrame({"mode": np.arange(1, len(m["singular"]) + 1), "singular_value": m["singular"],
                          "squared_covariance_fraction": m["scf"], "corr_expansion_coefficients": m["r_ab"],
                          "p_value_permutation": m["p_value"], "n_years": len(common), "n_permutations": nperm})
    modes.round(4).to_csv(outdir / "sst_analysis" / "teleconnection" / f"mca_modes_{pred_season}_to_{season}.csv",
                          index=False)
    pd.DataFrame({**{f"SST_a{i + 1}": m["a"][:, i] for i in range(m["a"].shape[1])},
                  **{f"Rain_b{i + 1}": m["b"][:, i] for i in range(m["b"].shape[1])}}, index=common).round(4).to_csv(
        outdir / "sst_analysis" / "teleconnection" / f"mca_expansion_coefficients_{pred_season}_to_{season}.csv",
        index_label="Year")
    kmode = len(m["singular"])
    cw._save(cw._map_dataset({**{f"sst_homogeneous_mode{i + 1}": m["hom_x"][i].reshape(nS) for i in range(kmode)},
                              **{f"sst_heterogeneous_mode{i + 1}": m["het_x"][i].reshape(nS) for i in range(kmode)}},
                             sst, {"long_name": "MCA SST patterns (correlation maps)"}),
             outdir / "sst_analysis" / "teleconnection" / f"mca_sst_patterns_{pred_season}_to_{season}.nc")
    cw._save(cw._map_dataset({**{f"rain_homogeneous_mode{i + 1}": m["hom_y"][i].reshape(nRc) for i in range(kmode)},
                              **{f"rain_heterogeneous_mode{i + 1}": m["het_y"][i].reshape(nRc) for i in range(kmode)}},
                             _Grid(rc_lat, rc_lon), {"long_name": "MCA rainfall patterns (correlation maps)"}),
             outdir / "sst_analysis" / "teleconnection" / f"mca_rain_patterns_{pred_season}_to_{season}.nc")
    log.info("MCA: SCF=%s, r(a,b)=%s, p=%s", np.round(m["scf"], 3), np.round(m["r_ab"], 2), np.round(m["p_value"], 3))
    if args.make_maps:
        for i in range(min(2, kmode)):
            cmap.map_field(m["hom_x"][i].reshape(nS), slat, slon,
                           f"MCA mode {i + 1} SST {pred_season} (SCF {100 * m['scf'][i]:.0f}%, p={m['p_value'][i]:.3f})",
                           outdir / "maps" / "teleconnection" / f"mca_mode{i + 1}_sst", kind="diverging",
                           label="correlation with SST expansion coefficient", vmax=1.0, formats=formats)
            cmap.map_field(m["het_y"][i].reshape(nRc), rc_lat, rc_lon,
                           f"MCA mode {i + 1} rainfall {season}: correlation with SST coefficient",
                           outdir / "maps" / "teleconnection" / f"mca_mode{i + 1}_rain", kind="diverging",
                           label="correlation", vmax=1.0, formats=formats)

    # ------------------------------------------------------------ 4. ENSO-related rainfall response
    phases = read_enso_phases(Path(args.enso_excel or in_dir / config.ENSO_EXCEL), season)
    src = "Excel sheet ENSO_" + season
    if phases is None:
        thr = config.ENSO_THRESHOLD
        phases = {int(y): ("EL" if v >= thr else "LA" if v <= -thr else "N") for y, v in zip(yt, nino_t)
                  if np.isfinite(v)} if len(nino_t) == len(yt) else {}
        src = f"Nino3.4 {season} mean, threshold +-{thr} degC"
    log.info("ENSO phases from %s: %s", src, pd.Series(phases).value_counts().to_dict())
    resp_rows = []
    for ph_code, ph_name in (("EL", "El Nino"), ("LA", "La Nina")):
        member = np.array([phases.get(int(y)) == ph_code for y in yt])
        member &= np.isfinite(rt_reg)
        diff, pv, nm, no = tc.composite(Ra, member)
        diff_pct, _, _, _ = tc.composite(Rpct, member)
        sig = tc.fdr_bh(pv, config.FDR_ALPHA)
        cw._save(cw._map_dataset({"anomaly_mm": diff.reshape(nR), "anomaly_pct_of_normal": diff_pct.reshape(nR),
                                  "p_value": pv.reshape(nR), "fdr_sig": sig.reshape(nR).astype("float32")}, rain,
                                 {"long_name": f"{ph_name} composite rainfall response ({season}), {src}",
                                  "n_years": nm}),
                 outdir / "sst_analysis" / "enso" / f"enso_rain_{ph_code}_{season}.nc")
        a_ = rt_anom[member]; b_ = rt_anom[~member & np.isfinite(rt_reg)]
        from scipy import stats as _st
        tt = _st.ttest_ind(a_, b_, equal_var=False) if len(a_) > 2 and len(b_) > 2 else None
        resp_rows.append({"phase": ph_name, "season": season, "region": args.region, "n_years": nm,
                          "years": " ".join(str(int(y)) for y in yt[member]), "mean_anomaly_mm": np.nanmean(a_),
                          "mean_pct_of_normal": 100 * np.nanmean(a_) / clim_reg,
                          "p_value_welch": float(tt.pvalue) if tt is not None else np.nan,
                          "phase_source": src})
        if args.make_maps and nm >= 3:
            cmap.map_field(diff_pct.reshape(nR), r_lat, r_lon, f"{ph_name}: {season} rainfall anomaly ({nm} years)",
                           outdir / "maps" / "teleconnection" / f"enso_rain_{ph_code}_{season}", kind="diverging",
                           label="% of normal", vmax=60, formats=formats, stipple=sig.reshape(nR),
                           subtitle="dots: FDR-significant (Welch t-test vs all other years)")
    pd.DataFrame(resp_rows).round(3).to_csv(outdir / "sst_analysis" / "enso" / f"enso_rainfall_response_{season}.csv",
                                            index=False)

    # ------------------------------------------------------------ 5. analogues, MOS, hindcast skill
    band = (slat >= config.ANALOGUE_LAT[0]) & (slat <= config.ANALOGUE_LAT[1])
    cells = np.repeat(band[:, None], len(slon), axis=1).ravel()
    wA = w2d.ravel()[cells]
    Fp = Sp[:, cells]                                           # predictor fields, all predictor years
    has_rain = np.isfinite(rt_anom)
    y_rain = np.full(len(yp), np.nan)
    for i, y in enumerate(yp):
        j = np.where(yt == y)[0]
        if len(j):
            y_rain[i] = rt_anom[j[0]]
    pred_an = tc.analogue_loo_hindcast(Fp, yp, y_rain, wA, k=k_an)
    x_nino = np.array([nino_p[np.where(yp == y)[0][0]] if y in yp else np.nan for y in yp])
    pred_mos = tc.loo_linear_hindcast(x_nino, y_rain)
    sk_rows = []
    for name_, pr in (("analogue_mean (k=%d)" % k_an, pred_an), ("MOS linear on Nino3.4 %s" % pred_season, pred_mos)):
        sk_rows.append({"method": name_, "target": f"{args.region} {season} rainfall anomaly (mm)",
                        "validation": "leave-one-out cross-validation", **tc.forecast_skill(y_rain, pr)})
    pd.DataFrame(sk_rows).round(4).to_csv(outdir / "forecasting" / "regression" / f"hindcast_skill_{pred_season}_to_{season}.csv",
                                         index=False)
    pd.DataFrame({"Year": yp, "observed_anomaly_mm": y_rain, "analogue_hindcast_mm": pred_an,
                  "mos_hindcast_mm": pred_mos}).round(2).to_csv(
        outdir / "forecasting" / "regression" / f"hindcast_series_{pred_season}_to_{season}.csv", index=False)

    # outlook targets: requested years plus the latest year with a predictor but no (complete) target season
    latest_open = [int(y) for y in yp if y not in set(yt[np.isfinite(rt_reg)])]
    outlook_years = sorted(set(target_years) | set(latest_open[-1:]))
    skill = {r_["method"]: r_ for r_ in sk_rows}
    cand = np.isfinite(y_rain)
    ref_t = rt_reg[ref_years_t]
    out_rows, an_rows = [], []
    for y in outlook_years:
        if y not in yp:
            log.warning("Outlook year %d: no complete %s predictor season; skipped", y, pred_season)
            continue
        i = int(np.where(yp == y)[0][0])
        ay, ac = tc.rank_analogues(Fp[cand], yp[cand], Fp[i], wA, exclude_year=y, k=k_an)
        arain = np.array([y_rain[np.where(yp == a)[0][0]] for a in ay])
        for rk, (a, c_, rv) in enumerate(zip(ay, ac, arain), 1):
            an_rows.append({"target_year": y, "rank": rk, "analogue_year": int(a), "pattern_correlation": c_,
                            f"analogue_{season}_anomaly_mm": rv,
                            "analogue_phase": phases.get(int(a), "")})
        mos_pred = np.nan
        m_ok = np.isfinite(x_nino) & np.isfinite(y_rain) & (yp != y)
        if m_ok.sum() >= 8 and np.isfinite(x_nino[i]):
            b_, a_ = np.polyfit(x_nino[m_ok], y_rain[m_ok], 1)
            mos_pred = a_ + b_ * x_nino[i]
        an_mean = float(np.mean(arain)) if len(arain) else np.nan
        obs = rt_anom[np.where(yt == y)[0][0]] if (y in yt and np.isfinite(rt_reg[np.where(yt == y)[0][0]])) else np.nan
        out_rows.append({
            "outlook_year": y, "season": season, "region": args.region, "predictor": f"SST {pred_season} {y}",
            "analogue_years": " ".join(str(int(a)) for a in ay), "analogue_mean_anomaly_mm": an_mean,
            "analogue_mean_pct_of_normal": 100 * (clim_reg + an_mean) / clim_reg,
            "analogue_category": str(tercile_category(np.array([clim_reg + an_mean]), ref_t)[0]),
            "mos_anomaly_mm": mos_pred,
            "mos_category": str(tercile_category(np.array([clim_reg + mos_pred]), ref_t)[0]) if np.isfinite(mos_pred) else "",
            "observed_anomaly_mm": obs,
            "analogue_loo_correlation": skill[next(iter(skill))].get("pearson", np.nan),
            "analogue_loo_rmsess": skill[next(iter(skill))].get("rmsess_vs_climatology", np.nan),
            "mos_loo_correlation": list(skill.values())[1].get("pearson", np.nan),
            "mos_loo_rmsess": list(skill.values())[1].get("rmsess_vs_climatology", np.nan),
            "status": "STATISTICAL OUTLOOK - not an operational forecast; see hindcast skill columns",
        })
        if args.make_maps:
            cmap.map_field(Sp[i].reshape(nS), slat, slon, f"SST anomaly {pred_season} {y}",
                           outdir / "maps" / "sst" / f"sst_anomaly_{pred_season}_{y}", kind="diverging",
                           label="°C", vmax=2.0, cmap="RdBu_r", formats=formats)
    if an_rows:
        pd.DataFrame(an_rows).round(3).to_csv(outdir / "forecasting" / "analogue_years" /
                                              f"analogues_{pred_season}_to_{season}.csv", index=False)
    if out_rows:
        pd.DataFrame(out_rows).round(3).to_csv(outdir / "forecasting" / "seasonal_outlooks" /
                                               f"outlook_{season}_{args.region}.csv", index=False)

    # ------------------------------------------------------------ report
    lines = ["SST-RAINFALL TELECONNECTION REPORT", "=" * 60,
             f"Target: {args.region} {season} rainfall | predictor SST: {pred_season} | years used: {len(common)}",
             f"Reference period for anomalies: {ref_start}-{ref_end}; SST block-averaged to ~{config.SST_COARSEN_DEG} deg; "
             f"rainfall MCA grid ~{config.RAIN_COARSEN_DEG} deg", "",
             f"Nino3.4 {pred_season} vs regional rainfall {season}: r={cs.get('pearson', np.nan):.2f} "
             f"(95% bootstrap CI {cs.get('ci_low', np.nan):.2f}..{cs.get('ci_high', np.nan):.2f}; "
             f"p={cs.get('pearson_p', np.nan):.3f}, N_eff-adjusted p={cs.get('pearson_p_neff', np.nan):.3f})",
             "MCA: " + "; ".join(f"mode {i + 1}: SCF={m['scf'][i]:.2f}, r={m['r_ab'][i]:.2f}, p={m['p_value'][i]:.3f}"
                                 for i in range(kmode)), "",
             "Leave-one-out hindcast skill:"] + [
        f"  {r_['method']}: r={r_.get('pearson', np.nan):.2f}, RMSE-skill vs climatology={r_.get('rmsess_vs_climatology', np.nan):.2f}, "
        f"tercile hit rate={r_.get('tercile_hit_rate', np.nan):.2f} (random 0.33)" for r_ in sk_rows] + [
        "", "Caveats:",
        f"  * Only ~{len(common)} annual samples: correlations/MCA are noisy; permutation p-values and FDR control are used, "
        "but results should be interpreted as exploratory.",
        "  * Lagged-correlation significance uses an effective sample size from lag-1 autocorrelation; with "
        "season-restricted samples this is approximate.",
        "  * Analogue and MOS outlooks are statistical products validated only by leave-one-out hindcasts; they are not "
        "operational forecasts and carry no calibrated probabilities.",
        f"  * ENSO phases come from: {src}."]
    (outdir / "reports" / f"teleconnection_report_{season}.txt").write_text("\n".join(lines))
    log.info("Teleconnection analysis finished in %.1f min", (time.time() - t0) / 60)
