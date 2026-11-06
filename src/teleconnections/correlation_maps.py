"""SST-rainfall teleconnection workflow: lagged correlation maps, ENSO/IOD rainfall response, MCA, analogue years and statistical outlooks.

Origin: teleconnection.py (run)."""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from src import settings
from src.data_ingestion import netcdf_loader
from src.data_ingestion.netcdf_loader import load_sst_monthly
from src.export.netcdf_export import write_nc
from src.preprocessing.anomaly import season_total, window_mean
from src.preprocessing.regridding import coarsen_rain
from src.spatial import forecast_maps
from src.teleconnections import enso, sst_indices, tele_kernels
from src.teleconnections.sst_indices import box_mean


log = logging.getLogger("clim.tele")


DIRS = ["sst_analysis/sst_anomalies", "sst_analysis/nino34", "sst_analysis/enso", "sst_analysis/teleconnection",
        "forecasting/analogue_years", "forecasting/regression", "forecasting/seasonal_outlooks",
        "maps/sst", "maps/teleconnection", "statistics/correlation", "statistics/regression", "reports"]


def run(args, outdir: Path):
    for d in DIRS:
        (outdir / d).mkdir(parents=True, exist_ok=True)
    ref0, ref1 = args.ref_start, args.ref_end
    months_season = settings.CLIM_SEASONS[args.season]
    first_month = min(months_season)
    lags = [int(x) for x in str(args.lags).split(",") if x.strip()]
    region_box = settings.STUDY_REGIONS[args.region]
    start = args.start or settings.ANALYSIS_START
    end = args.end or settings.ANALYSIS_END
    in_dir = Path(args.input_dir or settings.CLIM_INPUT_DIR)
    sst_path = Path(args.sst_file) if args.sst_file else in_dir / settings.SST_FILE
    if not sst_path.exists():
        raise FileNotFoundError(f"SST file not found: {sst_path}")

    # ---------------- data ----------------
    sst, sti, slat, slon = load_sst_monthly(sst_path, args.sst_res, "1980-01", end)
    data, _ = netcdf_loader.load_inputs(in_dir, need_temperature=False, prcp_var=args.prcp_var, start=start, end=end)
    rain, rlat, rlon, rf = coarsen_rain(data["prcp"], args.rain_res)
    rti = data["prcp"].indexes["time"]
    log.info("Rainfall block-averaged by factor %d -> grid %dx%d", rf, len(rlat), len(rlon))

    sref = (sti.year >= ref0) & (sti.year <= ref1)
    rref = (rti.year >= ref0) & (rti.year <= ref1)
    nt, ny, nx = sst.shape
    sst_anom = tele_kernels.monthly_anomalies(sst.reshape(nt, -1), np.asarray(sti.month), sref).reshape(sst.shape)
    rt = rain.shape[0]
    rain_flat = rain.reshape(rt, -1)
    rain_anom = tele_kernels.monthly_anomalies(rain_flat, np.asarray(rti.month), rref)                   # mm
    rain_std = tele_kernels.monthly_anomalies(rain_flat, np.asarray(rti.month), rref, standardize=True)   # z-scores

    # ---------------- climate indices from the SST grid ----------------
    r34 = box_mean(sst_anom, slat, slon, settings.REGIONS["R34"])
    wiod = box_mean(sst_anom, slat, slon, settings.REGIONS["WIOD"])
    eiod = box_mean(sst_anom, slat, slon, settings.REGIONS["EIOD"])
    idx = pd.DataFrame({"R34": r34, "WIOD": wiod, "EIOD": eiod, "DMI": wiod - eiod}, index=sti)
    idx.index.name = "Date"
    idx.round(4).to_csv(outdir / "sst_analysis/nino34/climate_indices_from_sst_grid.csv")
    ev, phase_m = enso.enso_events(idx["R34"].dropna())
    ev.to_csv(outdir / "sst_analysis/enso/ENSO_events_from_grid.csv", index=False)
    phase_m.rename("ENSO_phase").to_csv(outdir / "sst_analysis/enso/ENSO_monthly_phase.csv", index_label="Date")
    write_nc({"sst_anomaly": (("time", "lat", "lon"), sst_anom)},
              {"time": sti, "lat": slat, "lon": slon}, outdir / "sst_analysis/sst_anomalies/sst_anomaly_monthly.nc",
              {"reference_period": f"{ref0}-{ref1}", "units": "degC"})

    # ---------------- rainfall indices ----------------
    lm, om = sst_indices.region_masks(rlat, rlon, region_box["lat"], region_box["lon"])
    reg = np.zeros((len(rlat), len(rlon)), bool); reg[np.ix_(lm, om)] = True
    wcell = (np.cos(np.deg2rad(rlat))[:, None] * np.ones((1, len(rlon))))
    wreg = np.where(reg, wcell, 0.0).ravel()

    def regional(a2d):   # (T, cells) -> (T,) weighted mean over the study region
        ok = np.isfinite(a2d) & (wreg[None, :] > 0)
        w = np.where(ok, wreg[None, :], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w.sum(1) > 0, (np.where(ok, a2d, 0) * w).sum(1) / w.sum(1), np.nan)

    rain_index_m = regional(rain_std)            # monthly standardized regional rainfall anomaly

    # ======================================================================
    # 1) Lagged SST -> regional rainfall correlation maps
    # ======================================================================
    # align SST to the rainfall time axis
    common = rti.intersection(sti)
    s_pos = sti.get_indexer(common)
    r_pos = rti.get_indexer(common)
    X = sst_anom.reshape(nt, -1)[s_pos]
    yv = rain_index_m[r_pos]
    sel = np.isin(common.month, months_season)
    lc = tele_kernels.lagged_correlation(X, yv, lags, sel, fdr_alpha=args.fdr)
    fields = {}
    for i, L in enumerate(lags):
        for nm in ("r", "n", "n_eff", "p"):
            fields[f"{nm}_lag{L}"] = (("lat", "lon"), np.asarray(lc[nm][i], float).reshape(ny, nx))
        fields[f"sig_fdr_lag{L}"] = (("lat", "lon"), lc["sig"][i].astype(float).reshape(ny, nx))
    fields["r_max"] = (("lat", "lon"), lc["r_max"].reshape(ny, nx))
    fields["lag_of_max_abs_r"] = (("lat", "lon"), lc["lag_of_max"].reshape(ny, nx))
    write_nc(fields, {"lat": slat, "lon": slon},
              outdir / f"sst_analysis/teleconnection/sst_{args.season}_rainfall_lagged_correlation.nc",
              {"description": f"Correlation of monthly SST anomaly at lag L with {args.region} standardized rainfall "
                              f"anomaly in {args.season} months; p-values use effective sample size; "
                              f"FDR level {args.fdr}"})
    if args.make_maps:
        for i, L in enumerate(lags):
            forecast_maps.map_field(lc["r"][i].reshape(ny, nx), slat, slon, f"SST lead {L} mo vs {args.region} {args.season} rainfall",
                           outdir / f"maps/teleconnection/sst_rain_corr_lag{L}", kind="diverging", vmax=0.8,
                           label="Pearson r", formats=args.formats, stipple=lc["sig"][i].reshape(ny, nx),
                           subtitle=f"dots: FDR-significant (q={args.fdr}); ref {ref0}-{ref1}")
        forecast_maps.map_field(lc["r_max"].reshape(ny, nx), slat, slon, "Maximum |r| over lags", outdir / "maps/teleconnection/sst_rain_corr_max",
                       kind="diverging", vmax=0.8, label="r at the strongest lag", formats=args.formats)
        forecast_maps.map_field(lc["lag_of_max"].reshape(ny, nx), slat, slon, "Lag (months) of maximum |r|",
                       outdir / "maps/teleconnection/sst_rain_corr_lag_of_max", kind="sequential", label="months",
                       vmin=min(lags), vmax=max(lags), cmap="viridis", formats=args.formats)
    summ = pd.DataFrame({"lag": lags, "n_cells_sig_fdr": [int(s.sum()) for s in lc["sig"]],
                         "max_abs_r": [float(np.nanmax(np.abs(r))) for r in lc["r"]]})
    summ.to_csv(outdir / "statistics/correlation/sst_rainfall_lagged_summary.csv", index=False)

    # ======================================================================
    # 2) Rainfall grid vs Nino3.4 / DMI at lags
    # ======================================================================
    for nm in ("R34", "DMI"):
        iv = idx[nm].reindex(common).values
        lci = tele_kernels.lagged_correlation_vs_index(rain_std[r_pos], iv, lags, sel, fdr_alpha=args.fdr)
        f2 = {}
        for i, L in enumerate(lags):
            f2[f"r_lag{L}"] = (("lat", "lon"), np.asarray(lci["r"][i]).reshape(len(rlat), len(rlon)))
            f2[f"sig_fdr_lag{L}"] = (("lat", "lon"), lci["sig"][i].astype(float).reshape(len(rlat), len(rlon)))
            f2[f"n_eff_lag{L}"] = (("lat", "lon"), np.asarray(lci["n_eff"][i]).reshape(len(rlat), len(rlon)))
        f2["r_max"] = (("lat", "lon"), lci["r_max"].reshape(len(rlat), len(rlon)))
        f2["lag_of_max_abs_r"] = (("lat", "lon"), lci["lag_of_max"].reshape(len(rlat), len(rlon)))
        write_nc(f2, {"lat": rlat, "lon": rlon},
                  outdir / f"sst_analysis/teleconnection/rainfall_{args.season}_vs_{nm}_lagged_correlation.nc")
        if args.make_maps:
            for L in (lags[0], lags[len(lags) // 2], lags[-1]):
                i = lags.index(L)
                forecast_maps.map_field(lci["r"][i].reshape(len(rlat), len(rlon)), rlat, rlon,
                               f"{args.season} rainfall vs {nm} (index leads {L} mo)",
                               outdir / f"maps/teleconnection/rain_vs_{nm}_lag{L}", kind="diverging", vmax=0.8,
                               label="Pearson r", formats=args.formats,
                               stipple=lci["sig"][i].reshape(len(rlat), len(rlon)))

    # ======================================================================
    # 3) Annual season arrays: rainfall totals, predictor windows, indices
    # ======================================================================
    all_years = sorted(set(rti.year) & set(sti.year))
    rain_tot, tot_years = [], []
    for y in all_years:
        t = season_total(rain_flat, rti, y, months_season)
        if t is not None and np.isfinite(t).any():
            rain_tot.append(t); tot_years.append(y)
    rain_tot = np.array(rain_tot)
    tot_years = np.array(tot_years)
    ref_rows = (tot_years >= ref0) & (tot_years <= ref1)
    clim_tot = np.nanmean(rain_tot[ref_rows], axis=0)
    rain_tot_anom = rain_tot - clim_tot                                             # mm
    with np.errstate(invalid="ignore", divide="ignore"):
        rain_pct = 100 * rain_tot / clim_tot                                         # % of normal
    reg_anom = regional(rain_tot_anom)                                               # (years,)
    pd.DataFrame({"Year": tot_years, f"{args.region}_{args.season}_rain_anomaly_mm": reg_anom,
                  f"{args.region}_{args.season}_rain_pct_of_normal": regional(rain_pct)}).round(3).to_csv(
        outdir / f"statistics/correlation/{args.region}_{args.season}_rainfall_anomaly_by_year.csv", index=False)

    # ENSO index (seasonal mean of the target-season months) and phase per year
    nino_seas = np.array([float(np.nanmean(idx["R34"][(idx.index.year == y) & idx.index.month.isin(months_season)]))
                          if ((idx.index.year == y) & idx.index.month.isin(months_season)).sum() == len(months_season)
                          else np.nan for y in tot_years])
    dmi_seas = np.array([float(np.nanmean(idx["DMI"][(idx.index.year == y) & idx.index.month.isin(months_season)]))
                         if ((idx.index.year == y) & idx.index.month.isin(months_season)).sum() == len(months_season)
                         else np.nan for y in tot_years])
    phase = tele_kernels.phase_from_index(nino_seas, settings.ENSO_THRESHOLD)
    pd.DataFrame({"Year": tot_years, "R34_season_mean": nino_seas, "ENSO_phase": phase, "DMI_season_mean": dmi_seas,
                  "regional_rain_anomaly_mm": reg_anom}).round(3).to_csv(
        outdir / f"sst_analysis/enso/ENSO_phase_by_year_{args.season}.csv", index=False)
    freq = pd.Series(phase).value_counts().rename_axis("ENSO_phase").reset_index(name="n_years")
    freq["pct_of_years"] = (100 * freq["n_years"] / freq["n_years"].sum()).round(1)
    freq.insert(0, "Season", args.season)
    freq.to_csv(outdir / "reports/enso_frequency.csv", index=False)

    # ======================================================================
    # 4) ENSO rainfall response (composites + significance)
    # ======================================================================
    el, la, ne = phase == "El Nino", phase == "La Nina", phase == "Neutral"
    rows = []
    if el.sum() >= 3 and ne.sum() >= 3:
        comp = {}
        for nm, grp in (("ElNino", el), ("LaNina", la)):
            if grp.sum() < 3:
                log.warning("Fewer than 3 %s years for %s; composite skipped", nm, args.season)
                continue
            d, p, sig = tele_kernels.composite_difference(rain_tot_anom, grp, ne, args.fdr)
            comp[nm] = (d, p, sig)
            pct = 100 * np.nanmean(rain_tot[grp], 0) / clim_tot - 100 * np.nanmean(rain_tot[ne], 0) / clim_tot
            comp[nm + "_pct"] = pct
            rows.append({"Season": args.season, "Region": args.region, "Phase": nm, "n_years": int(grp.sum()),
                         "years": " ".join(map(str, tot_years[grp])),
                         "regional_mean_anom_vs_neutral_mm": float(regional(d[None, :])[0]),
                         "regional_mean_pct_vs_neutral": float(regional(pct[None, :])[0]),
                         "pct_cells_FDR_significant": 100 * float(np.nanmean(sig[wreg > 0]))})
        out = {}
        for nm in [n for n in comp if not n.endswith("_pct")]:
            d, p, sig = comp[nm]
            out[f"{nm}_minus_neutral_mm"] = (("lat", "lon"), d.reshape(len(rlat), len(rlon)))
            out[f"{nm}_minus_neutral_pct_of_normal"] = (("lat", "lon"), comp[nm + "_pct"].reshape(len(rlat), len(rlon)))
            out[f"{nm}_p_value"] = (("lat", "lon"), p.reshape(len(rlat), len(rlon)))
            out[f"{nm}_sig_fdr"] = (("lat", "lon"), sig.astype(float).reshape(len(rlat), len(rlon)))
        write_nc(out, {"lat": rlat, "lon": rlon}, outdir / f"sst_analysis/enso/enso_rainfall_composites_{args.season}.nc")
        if args.make_maps:
            for nm in [n for n in comp if not n.endswith("_pct")]:
                d, p, sig = comp[nm]
                forecast_maps.map_field(d.reshape(len(rlat), len(rlon)), rlat, rlon,
                               f"{args.season} rainfall: {nm} minus Neutral", outdir / f"maps/teleconnection/enso_{nm}_{args.season}",
                               kind="diverging", label="mm", formats=args.formats,
                               stipple=sig.reshape(len(rlat), len(rlon)), subtitle=f"dots: FDR-significant (q={args.fdr})")
    else:
        log.warning("Too few El Nino/Neutral years for composites (El Nino=%d, Neutral=%d)", el.sum(), ne.sum())
    pd.DataFrame(rows).round(3).to_csv(outdir / f"sst_analysis/enso/enso_rainfall_response_{args.season}.csv", index=False)

    # Pair statistics: seasonal indices vs regional rainfall anomaly
    ps = []
    for nm, v in (("Nino3.4", nino_seas), ("DMI", dmi_seas)):
        s = tele_kernels.pair_statistics(v, reg_anom, n_boot=args.n_perm, seed=settings.RANDOM_SEED)
        s.update({"predictor": nm, "predictand": f"{args.region} {args.season} rainfall anomaly", "season": args.season})
        ps.append(s)
    pd.DataFrame(ps).round(4).to_csv(outdir / "statistics/correlation/index_vs_regional_rainfall_statistics.csv", index=False)

    # ======================================================================
    # 5) Maximum covariance analysis: SST predictor window vs seasonal rainfall
    # ======================================================================
    lead = args.lead
    sst_flat = sst_anom.reshape(nt, -1)
    pred_years, Xs = [], []
    for y in tot_years:
        w = window_mean(sst_flat, sti, int(y), first_month, lead, args.predictor_months)
        if w is not None:
            pred_years.append(y); Xs.append(w)
    pred_years = np.array(pred_years)
    Xs = np.array(Xs)
    keep = np.isin(tot_years, pred_years)
    Yr = rain_tot_anom[keep]
    yrs = tot_years[keep]
    wx = np.sqrt(np.maximum(np.cos(np.deg2rad(np.repeat(slat, nx))), 0))
    wy = np.sqrt(np.maximum(np.cos(np.deg2rad(np.repeat(rlat, len(rlon)))), 0))
    mc = tele_kernels.mca(Xs, Yr, n_modes=3, n_perm=args.n_perm, seed=settings.RANDOM_SEED, wx=wx, wy=wy)
    nm_ = len(mc["scf"])
    pd.DataFrame({"mode": np.arange(1, nm_ + 1), "singular_value": mc["singular_values"], "SCF_pct": 100 * mc["scf"],
                  "EC_correlation": mc["ec_corr"], "p_singular_value_perm": mc["p_singular"],
                  "p_SCF_perm": mc["p_scf"], "n_years": len(yrs), "lead_months": lead,
                  "predictor_months": args.predictor_months, "season": args.season}).round(4).to_csv(
        outdir / f"sst_analysis/teleconnection/mca_modes_{args.season}_lead{lead}.csv", index=False)
    pd.DataFrame({"Year": yrs, **{f"SST_EC{i + 1}": mc["U"][:, i] for i in range(nm_)},
                  **{f"Rain_EC{i + 1}": mc["V"][:, i] for i in range(nm_)}}).round(4).to_csv(
        outdir / f"sst_analysis/teleconnection/mca_expansion_coefficients_{args.season}_lead{lead}.csv", index=False)
    write_nc({"sst_homogeneous_corr": (("mode", "lat", "lon"), mc["hom_x"].reshape(nm_, ny, nx)),
               "sst_singular_vector": (("mode", "lat", "lon"), mc["pattern_x"].reshape(nm_, ny, nx))},
              {"mode": np.arange(1, nm_ + 1), "lat": slat, "lon": slon},
              outdir / f"sst_analysis/teleconnection/mca_sst_patterns_{args.season}_lead{lead}.nc")
    write_nc({"rain_homogeneous_corr": (("mode", "lat", "lon"), mc["hom_y"].reshape(nm_, len(rlat), len(rlon))),
               "rain_singular_vector": (("mode", "lat", "lon"), mc["pattern_y"].reshape(nm_, len(rlat), len(rlon)))},
              {"mode": np.arange(1, nm_ + 1), "lat": rlat, "lon": rlon},
              outdir / f"sst_analysis/teleconnection/mca_rain_patterns_{args.season}_lead{lead}.nc")
    if args.make_maps:
        for i in range(min(nm_, 2)):
            ttl = f"MCA mode {i + 1} (SCF {100 * mc['scf'][i]:.0f}%, r_EC={mc['ec_corr'][i]:.2f}, p={mc['p_scf'][i]:.3f})"
            forecast_maps.map_field(mc["hom_x"][i].reshape(ny, nx), slat, slon, ttl + " – SST",
                           outdir / f"maps/teleconnection/mca{i + 1}_sst_{args.season}_lead{lead}", kind="diverging",
                           vmax=1.0, label="homogeneous correlation", formats=args.formats)
            forecast_maps.map_field(mc["hom_y"][i].reshape(len(rlat), len(rlon)), rlat, rlon, ttl + f" – {args.season} rainfall",
                           outdir / f"maps/teleconnection/mca{i + 1}_rain_{args.season}_lead{lead}", kind="diverging",
                           vmax=1.0, label="homogeneous correlation", formats=args.formats)

    # ======================================================================
    # 6) Leave-one-year-out skill by lead (analogue + regression), 0..6 months
    # ======================================================================
    skill_rows = []
    for L in range(0, 7):
        P, ok_years = [], []
        for i, y in enumerate(tot_years):
            w = window_mean(sst_flat, sti, int(y), first_month, L, args.predictor_months)
            if w is not None and np.isfinite(reg_anom[i]):
                P.append(w); ok_years.append(i)
        if len(P) < 15:
            continue
        P = np.array(P); yy = reg_anom[ok_years]
        n34 = np.array([box_mean(P.reshape(-1, ny, nx)[j][None], slat, slon, settings.REGIONS["R34"])[0]
                        for j in range(len(P))])
        for method, feat in (("analogue (SST pattern)", P), ("regression on Nino3.4", n34[:, None]),
                             ("regression on Nino3.4 + DMI", None)):
            if feat is None:
                dm = np.array([box_mean(P.reshape(-1, ny, nx)[j][None], slat, slon, settings.REGIONS["WIOD"])[0]
                               - box_mean(P.reshape(-1, ny, nx)[j][None], slat, slon, settings.REGIONS["EIOD"])[0]
                               for j in range(len(P))])
                feat = np.column_stack([n34, dm])
            res = tele_kernels.loo_skill(feat, yy, top_k=args.top_k, weights=np.sqrt(np.maximum(np.cos(np.deg2rad(np.repeat(slat, nx))), 0)) ** 2
                               if method.startswith("analogue") else None,
                               method="analogue" if method.startswith("analogue") else "regression")
            res.pop("hindcast", None)
            skill_rows.append({"season": args.season, "lead_months": L, "method": method, **res})
    skill = pd.DataFrame(skill_rows).round(4)
    skill["note"] = "leave-one-year-out hindcast; statistical outlook, not an operational forecast"
    skill.to_csv(outdir / f"forecasting/regression/loo_skill_by_lead_{args.season}.csv", index=False)

    # ======================================================================
    # 7) Analogue years and seasonal outlook for the target years
    # ======================================================================
    targets = [int(y) for y in str(args.target_years).split(",") if y.strip()] or [int(sti.year.max())]
    lib_years, lib_fields, lib_rain, lib_idx = [], [], [], []
    for i, y in enumerate(tot_years):
        w = window_mean(sst_flat, sti, int(y), first_month, lead, args.predictor_months)
        if w is not None:
            lib_years.append(int(y)); lib_fields.append(w); lib_rain.append(i)
    lib_years = np.array(lib_years); lib_fields = np.array(lib_fields); lib_rain = np.array(lib_rain)
    w_area = np.cos(np.deg2rad(np.repeat(slat, nx)))
    ana_rows, outl_rows = [], []
    for ty in targets:
        tgt = window_mean(sst_flat, sti, ty, first_month, lead, args.predictor_months)
        if tgt is None:
            log.warning("Target year %d: SST predictor window (lead %d) not available; skipped", ty, lead)
            continue
        top, idxs = tele_kernels.rank_analogues(tgt, lib_fields, lib_years, w_area, args.top_k, exclude_year=ty)
        for r in top:
            r.update({"target_year": ty, "season": args.season, "lead_months": lead})
            ana_rows.append(r)
        ay = [r["year"] for r in top]
        rows_a = [int(np.where(tot_years == y)[0][0]) for y in ay]
        comp_mm = np.nanmean(rain_tot_anom[rows_a], axis=0)
        comp_pct = 100 * np.nanmean(rain_tot[rows_a], axis=0) / clim_tot
        # fraction of analogue years with below / above normal regional rainfall
        ra = reg_anom[rows_a]
        outl_rows.append({"target_year": ty, "season": args.season, "lead_months": lead, "region": args.region,
                          "analogue_years": " ".join(map(str, ay)),
                          "analogue_mean_regional_anomaly_mm": float(np.nanmean(ra)),
                          "analogue_mean_pct_of_normal": float(regional(comp_pct[None, :])[0]),
                          "n_analogues_below_normal": int((ra < 0).sum()), "n_analogues_above_normal": int((ra > 0).sum()),
                          "analogue_spread_sd_mm": float(np.nanstd(ra, ddof=1)) if len(ra) > 1 else np.nan,
                          "note": "analogue composite = statistical outlook; see loo_skill_by_lead for validated skill"})
        write_nc({"analogue_mean_anomaly_mm": (("lat", "lon"), comp_mm.reshape(len(rlat), len(rlon))),
                   "analogue_mean_pct_of_normal": (("lat", "lon"), comp_pct.reshape(len(rlat), len(rlon)))},
                  {"lat": rlat, "lon": rlon}, outdir / f"forecasting/seasonal_outlooks/analogue_outlook_{args.season}_{ty}_lead{lead}.nc",
                  {"analogue_years": " ".join(map(str, ay)), "disclaimer": "statistical analogue outlook, not an operational forecast"})
        if args.make_maps:
            forecast_maps.map_field(comp_mm.reshape(len(rlat), len(rlon)), rlat, rlon,
                           f"Analogue outlook {args.season} {ty}: rainfall anomaly (lead {lead} mo)",
                           outdir / f"maps/teleconnection/analogue_outlook_{args.season}_{ty}", kind="diverging",
                           label="mm", formats=args.formats,
                           subtitle="analogues: " + ", ".join(map(str, ay)) + " – statistical outlook")
            forecast_maps.map_field(tgt.reshape(ny, nx), slat, slon, f"SST anomaly predictor for {args.season} {ty} (lead {lead} mo)",
                           outdir / f"maps/sst/sst_predictor_{args.season}_{ty}", kind="diverging", vmax=2.0,
                           label="°C", formats=args.formats)
    pd.DataFrame(ana_rows).round(4).to_csv(outdir / f"forecasting/analogue_years/analogue_years_{args.season}_lead{lead}.csv", index=False)
    pd.DataFrame(outl_rows).round(3).to_csv(outdir / f"forecasting/seasonal_outlooks/seasonal_outlook_{args.season}_lead{lead}.csv", index=False)

    # summary
    summary = pd.DataFrame([{"item": "SST months", "value": len(sti)}, {"item": "rainfall seasons", "value": len(tot_years)},
                            {"item": "El Nino / La Nina / Neutral years", "value": f"{int(el.sum())}/{int(la.sum())}/{int(ne.sum())}"},
                            {"item": "MCA mode-1 SCF %", "value": round(100 * float(mc['scf'][0]), 1)},
                            {"item": "MCA mode-1 p (perm)", "value": round(float(mc['p_scf'][0]), 4)}])
    summary.to_csv(outdir / "reports/summary_statistics.csv", index=False)
    log.info("Teleconnection analysis finished")
