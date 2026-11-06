# SST Climate Indices: Analysis and 12-Month Forecasting

Computes ENSO Nino 3.4 (`R34`), Western Indian Ocean (`WIOD`), Eastern Indian Ocean (`EIOD`) and the Dipole Mode Index (`DMI = WIOD - EIOD`) from global SST, analyses their variability, and produces validated 1-12 month forecasts.

## Install and run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                 # full run, settings in config.py
python main.py --fast          # quick run (fewer trees/CV folds, refit every 24 months)
python main.py --force         # ignore cached intermediates
python main.py --input X.nc --output DIR
python main.py --from-regional-csv monthly_regional_sst.csv   # skip NetCDF step (columns Date,R34,WIOD,EIOD)
```

Defaults: input `/home/goitomk/DATA/SST/input/global_sst_1981-01-01_present.nc`, output `/home/goitomk/DATA/SST/output` (subfolders `indices analysis figures forecast models validation logs cache`). The input file is only read, never modified. Intermediate results (regional monthly SST, tuned hyper-parameters, per-model hindcasts) are cached in `output/cache`, so an interrupted run resumes and repeated runs are fast.

## Configuration (`config/*.yaml`)
Baseline period, region boxes (EIOD longitudes are editable), ENSO/IOD thresholds, training/validation dates, enabled models, lags, rolling windows, prediction-interval level, random seed.

## Method summary
- **Data**: variable, time, lat and lon are auto-detected; 0-360 and -180-180 longitudes and ascending/descending latitudes are handled; duplicate timestamps dropped; Kelvin converted to degC; values outside -2.5..40 degC masked. Only the three regions are read, year by year (memory-safe), reduced with **cos(latitude) weights** (NaN/land excluded per time step), then averaged to monthly. A month is complete if >= 50% of its days exist and the final month has data through its last day.
- **Anomalies**: relative to the monthly climatology of 1991-2020 (falls back to the full record, with a warning, if not covered).
- **Regions**: Nino 3.4 5S-5N, 170W-120W; WIOD 10S-10N, 50-70E; EIOD 10S-4N, 90-110E. `R34` is the Nino 3.4 SST anomaly, **not** the Southern Oscillation Index (that would need Tahiti and Darwin sea-level pressure).
- **Events**: ENSO: 3-month running mean of R34, |anom| >= 0.5 degC for >= 5 consecutive months (strong >= 1.5). IOD: monthly DMI, |DMI| >= 0.4 degC for >= 3 consecutive months (strong >= 0.8). Start/end, duration, peak, intensity tables in `analysis/`.
- **Statistics**: climatologies, annual/seasonal stats, OLS trends with AR(1)-adjusted significance plus Theil-Sen/Mann-Kendall, ACF/PACF, correlation matrices and lead-lag correlations with effective-sample-size significance, seasonal (JJAS, OND, DJF, MAM) relationships, IOD seasonal behaviour, strong-event years.
- **Forecasting**: persistence, seasonal climatology, AR (BIC), ARIMA, SARIMA, VAR (on R34/WIOD/EIOD; DMI derived as WIOD-EIOD), Random Forest, XGBoost, SVR, Gradient Boosting. ML models use a direct strategy (one model per index and lead) with lags 0-11 of all indices, rolling means/stds, ENSO-IOD interaction terms, tendencies and month harmonics.
- **No leakage**: predictors are trailing functions of data up to the origin; training rows for lead h require origin + h <= forecast origin; hyper-parameters are tuned by rolling-origin CV (with an h-month embargo) on data before the validation period only. `leakage_check` runs automatically.
- **Validation**: chronological hindcast over valid dates from `VALIDATION_START`, with periodic refits. The first 60% of validation dates **select** the model per index and lead (simplest model within 2% of best RMSE); the last 40% is an **independent test period** on which metrics (RMSE, MAE, MAPE where |obs| >= 0.5, correlation, R2, MBE = mean(forecast - obs)) are reported for the selected models.
- **Final forecast**: selected models retrained on all data; intervals from empirical quantiles of that model's hindcast errors (90% default). `Validated_Skill` is `yes` only if the selected model beat climatology in RMSE on the test period and had a significant positive correlation; otherwise the reason is stated. Forecasts are not observations.

## Outputs
`indices/` index CSVs (incl. `all_climate_indices_monthly.csv`), regional SST, climatology, inspection JSON; `analysis/` tables; `validation/` hindcasts, metrics, selection, protocol; `forecast/climate_indices_forecast_12_months.csv` (+ per-index files; extra columns: initialization date, interval level, validated-skill flag); `models/` per-index joblib bundles, preprocessing config, `model_metadata.json`; `figures/` PNGs.

## Assumptions and limitations
- Anomalies use a 1991-2020 baseline that overlaps the validation period (standard practice for indices, but a mild information leak for climatology-based comparisons). The SST trend is not removed, so Nino 3.4 is not "relative".
- Regional anomaly equals the weighted mean of grid anomalies except where ice/land masks vary in time.
- ENSO phase here uses a fixed threshold on the monthly index; it can differ from operational (ONI/BoM) classifications.
- Skill beyond a few months is generally limited; trust the `Validated_Skill` column, not the plotted curve.

## Testing status
The pipeline downstream of the NetCDF step was run end to end on synthetic monthly data (all figures and tables produced; DMI = WIOD - EIOD and zero baseline mean verified). Region masks and cos-lat weighting were unit-checked for 0-360 and -180-180 grids. The NetCDF reader (`data_processing.open_dataset/inspect_dataset/regional_monthly_sst`) and the ARIMA/SARIMA/VAR/XGBoost code paths could not be run in the build environment (packages unavailable); run `python main.py --fast` once on your machine and check `logs/`.

---

# Part 2 – Gridded SPI / SPEI, drought classification and spatial analysis

Runs **instead of** the SST workflow whenever `--index` is given. Everything is computed **cell by cell** on the
CHIRPS grid; precipitation and temperature are never averaged before an index is computed.

## Inputs / outputs (defaults in `config/*.yaml`)
Inputs in `/home/goitomk/temp/input_data`: `prcp-chirps-ethio-198101_202607.nc`, `tmin-era5-chirts-ethio-198101_202607.nc`,
`tmax-era5-chirts-ethio-198101_202607.nc`. Outputs in `/home/goitomk/temp/output_data`:

```
climate_indices/{spi,spei}/spi_3.nc ...      + classified/spi_3_class.nc (uint8 codes)
spatial_analysis/{drought_frequency,drought_duration,drought_severity,wetness_frequency}/*.nc
spatial_analysis/regional_statistics/*.csv|nc  (class area % and km2: monthly, seasonal, annual; class frequency maps)
statistics/{trends,validation}/                (trend maps, reference-period checks, SPEI fit diagnostics)
maps/{spi,spei,drought,wetness}/*.png|pdf      reports/ (CSV summaries, data_quality_report.json, analysis_report.txt, log)
```

## Example commands
```bash
python main.py --index both --scales 1,3,6,12 --season JJAS --target-years 1997,2015 \
    --classification --spatial-analysis --make-maps
python main.py --index spi --scales 3 --classification                       # SPI-3 only
python main.py --index spei --pet-method hargreaves --tmax-var tmx --tmin-var tmn --scales 1,3
python main.py --index both --ref-start 1981 --ref-end 2010 --formats png,pdf --geotiff
python tests/test_kernels.py                                                  # unit tests (or: pytest tests)
```

## Method
- **Data QA** (`climate_data.py`): coordinates/variables auto-detected (override with `--prcp-var/--tmin-var/--tmax-var`);
  units checked (Kelvin -> degC, negative/fill precipitation masked, mm/day -> mm/month); daily input aggregated to
  months; gaps reindexed to NaN and reported; temperature regridded bilinearly to the CHIRPS grid if grids differ;
  a JSON quality report is written. Common period default 1981-01 .. 2026-07.
- **One reference period** for SPI, SPEI (default = the SST baseline 1991-2020; `--ref-start/--ref-end`).
- **SPI**: trailing `k`-month sums; gamma fitted per cell and calendar month (Thom ML approximation) on the reference
  period with the mixed zero term `H = q + (1-q)G`; `SPI = Phi^-1(H)`; capped at +-3.09.
- **SPEI**: Hargreaves-Samani PET from tmin/tmax and FAO-56 extraterrestrial radiation (per cell latitude, month-mean Ra,
  actual days in month); `P - PET` accumulated; three-parameter **log-logistic** fitted per cell and calendar month by
  unbiased PWMs (Vicente-Serrano et al. 2010). Cell-months with an invalid fit (non-positive L-skewness) use a normal fit
  of the same reference sample (`SPEI_NORMAL_FALLBACK`; share mapped in `statistics/validation/spei_fit_failures.nc`).
- **Classes** (`index_kernels.classify`, uint8): 1 Extremely Dry (Z<=-2), 2 Severely Dry (-2<Z<=-1.5), 3 Moderately Dry
  (-1.5<Z<=-1), 4 Near Normal (-1<Z<1), 5 Moderately Wet (1<=Z<1.5), 6 Severely Wet (1.5<=Z<2), 7 Extremely Wet (Z>=2),
  0 missing. Area share (%) and km2 per class use spherical cell areas.
- **Spell statistics**: drought = index <= -1, wetness = index >= +1; frequency (% of months), number of events, mean/max
  duration, severity (sum of |index| per spell), intensity, persistence P(event next month | event now), OLS trend per
  decade (p-values assume independent annual values).
- **Validation**: reference-period mean ~0 / std ~1 per index; SPI is compared with `climate_indices.spi` on 30 random
  cells when that package is installed; unit tests check the gamma fit against `scipy.stats.gamma.fit`, log-logistic
  parameter recovery, class boundaries (no gaps/overlaps), accumulation, PET, cell areas and spell statistics.

---

# Part 3 – SST–rainfall teleconnection, ENSO rainfall response, MCA, analogues

```bash
python main.py --teleconnection --season JJAS --target-years 1997,2015,2026 --lags 0,1,2,3,4,5,6 --lead 3 --make-maps
python main.py --index both --classification --spatial-analysis --teleconnection --make-maps   # everything
```
Needs `sst_era5_198001_202607.nc` (+ the CHIRPS file) in the input folder (`--sst-file`, `--input-dir`).
SST is block-averaged to ~2 deg (`--sst-res`) and rainfall to ~0.5 deg (`--rain-res`) for the global/MCA analyses
(processed one year at a time, so daily global files are memory-safe). All anomalies use the same reference period.

| Product | File(s) |
|---|---|
| Lagged SST -> regional rainfall correlation (r, n, n_eff, p, FDR mask, max-|r| map, lag of max) | `sst_analysis/teleconnection/sst_<S>_rainfall_lagged_correlation.nc`, `statistics/correlation/sst_rainfall_lagged_summary.csv` |
| Rainfall grid vs Nino3.4 / DMI at lags 0-6 | `sst_analysis/teleconnection/rainfall_<S>_vs_{R34,DMI}_lagged_correlation.nc` |
| Maximum covariance analysis (singular values, SCF, ECs, EC correlation, homogeneous maps, permutation p) | `mca_modes_*.csv`, `mca_expansion_coefficients_*.csv`, `mca_{sst,rain}_patterns_*.nc` |
| ENSO phase by year, frequency, events, rainfall composites (El Nino/La Nina minus Neutral, Welch t + FDR) | `sst_analysis/enso/*`, `reports/enso_frequency.csv` |
| Pearson/Spearman/Kendall, slope, R2, RMSE, MAE, bias, bootstrap CI, permutation p (index vs regional rainfall) | `statistics/correlation/index_vs_regional_rainfall_statistics.csv` |
| Analogue years, analogue rainfall outlook (NetCDF + CSV + maps) | `forecasting/analogue_years/`, `forecasting/seasonal_outlooks/` |
| Leave-one-year-out skill by lead 0-6 months (analogue, regression on Nino3.4, Nino3.4+DMI; MSE skill score vs climatology) | `forecasting/regression/loo_skill_by_lead_<S>.csv` |

**Definitions.** Lead = months between the end of the SST predictor window (`--predictor-months`, default 3) and the first
month of the target season (lead 0 = window ends the month before the season). Significance: effective sample size from
lag-1 autocorrelations (Bretherton et al. 1999), Benjamini-Hochberg FDR across cells (`--fdr`), MCA by year-order
permutations. Rainfall of the season = sum of its months, anomalies in mm and % of normal.

**Important.** Analogue and regression outputs are *statistical outlooks* validated by leave-one-year-out hindcasts, not
operational forecasts. Check the `mse_skill_score` column: values <= 0 mean no skill relative to climatology at that lead.

## Status
- Tested (synthetic data, 22 unit tests + end-to-end runs): SPI/SPEI kernels, classification, spell statistics, spatial
  statistics, maps; lagged correlation, MCA, composites, analogues, leave-one-out skill, and the whole teleconnection
  pipeline (which recovered a planted El Nino -> dry JJAS signal with the right sign, region and lag dependence).
- **Not run here** (xarray/netCDF4 unavailable in the build sandbox): all NetCDF reading/writing code (`climate_data.py`,
  `load_sst_monthly`, `_save`/`_write_nc`) and the optional comparison with the `climate_indices` package. Run one small job
  first (`python main.py --index spi --scales 3 --classification`, then `--teleconnection --season JJAS`) and inspect `reports/`.
- Not included: forecasts of rainfall from dynamical models, Kendall/Spearman maps at grid level, block-bootstrap, and
  non-ENSO/IOD modes (e.g. Atlantic). SST trends are not removed before correlation.
