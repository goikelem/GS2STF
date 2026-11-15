# User guide

For installation and an overview see `README.md`; for design decisions and the module map see `docs/DEVELOPMENT_NOTES.md`. This guide is about using the tool and reading its output.

## 1. Which command for which question

| You want | Command (add `--sst-indices file.csv`; see section 2) | Main output |
|---|---|---|
| Seasonal station forecast (JJAS, ONDJ, ...) | `python main.py --stations obs.csv --station-season JJAS --station-forecast --init-month 5 --target-year 2026` | `station_forecast_*.csv/.xlsx` |
| Rainfall occurrence (wet days) | add `--wet-day-threshold 1.0` | rows with `Variable = wet_day_fraction` |
| Weeks 2-4 or 3-6 station forecast | `--stations obs.csv --subseasonal --init-date 2026-06-01 --lead-weeks 2 4` | `subseasonal_forecast_*.csv/.xlsx` |
| Seasonal forecast map (grid) | `--grid-forecast --grid-variable precipitation --season JJAS --init-month 5 --target-year 2026 --grid-region Ethiopia` | `.nc`, `.tif`, maps |
| Ocean indices (Nino 3.4, DMI, ...) and their forecasts | `python main.py` (SST workflow) | `indices/`, `forecast/` |
| A tour on synthetic data | `sh examples/run_demo.sh` | `demo_out/` |

## 2. What you need to supply

* **Station observations**: daily CSV(s) or a folder of CSVs; long format `Station,Date,Lat,Lon,Variable,Value` or simple format
  `Station,Date,lat,long,value` (see "Station input handling" in the README). Day-first dates must be declared (`--station-dayfirst`).
* **Ocean predictor table**: monthly CSV with a `Date` column and one column per index (R34, DMI, TNA, ...), as written by the SST workflow
  (`indices/all_predictors_monthly.csv`). Values are anomalies in degrees C. The table must cover the months before the initialization date.
* Optional: dynamical-model files (README section "Dynamical-model guidance"), a polygon file for clipping grids.
* `--init-month` is the LAST month of data you want used (5 = May, i.e. a forecast issued at the start of June). `--target-year` is the
  season label: for cross-year seasons (ONDJ) it is the START year by default (`season_year_label` in `config/datasets.yaml`).

## 3. Ideas you need to read the output

* **Initialization date.** Seasonal: the last day of `--init-month` (init month 5 -> `2026-05-31`); only data up to then are used.
  Subseasonal: the day you give; only data BEFORE it are used. **Lead.** Seasonal `lead_months` = calendar months from the init month to
  the first month of the season (May -> JJAS = 1). Subseasonal `lead_weeks` = weeks after the init date (`2-4` = days 8-28).
* **Terciles.** Below / near / above normal are the lowest, middle and highest third of the TRAINING years (all years before the target).
  `Tercile_lower` / `Tercile_upper` are the cut points in the variable's units. Climatology = every category 33%.
* **Climatology member.** "Climatology" is the benchmark: the training-year mean with 1/3-1/3-1/3 probabilities. Every method must beat it
  out of sample (by `--min-gain`) at each station, otherwise that station keeps climatology: a forecast labelled `climatology` means "no
  predictor showed skill here", not an error.
* **Skill score (RPSS).** 0 = no better than climatology, 1 = perfect, negative = worse. Computed on hindcasts: each past year is forecast
  using only earlier years (rolling origin).
* **Ensemble.** When several methods run, the final forecast is a weighted combination. Weights come from each member's past hindcast skill
  only; a member that has not beaten climatology gets zero weight.

## 4. Output dictionary

### Forecast table (`station_forecast_*.csv`, `subseasonal_forecast_*.csv`)
| Column | Meaning |
|---|---|
| `Station`, `lat`, `lon` | station and its (median) coordinates |
| `Forecast_Period` | `JJAS_2026` (seasonal) or `W2-4_2026-06-01` (subseasonal: weeks 2-4 after that date) |
| `Variable` | `precipitation` (mm per season / per window, expected total), `tmax`/`tmin`/`tmean` (degrees C, mean), `wet_day_fraction` (0-1) |
| `Forecast` | expected value on the original scale (a mean, not a median) |
| `Anomaly` | `Forecast - Climatology` |
| `Climatology` | mean of the training years |
| `P_below`, `P_normal`, `P_above` | probabilities of the three tercile categories (sum to 1) |
| `Tercile_lower`, `Tercile_upper` | category cut points from the training years |
| `Predictors` | single method: the predictors used (`R34+TNA`; `climatology` = none beat it). Ensemble: `member(predictors)` for members with weight |
| `N_train_years` | years used for training |
| `Method` | method name, or `ensemble[skill,pooled]` |
| `init_date`, `lead_months` / `lead_weeks`, `target_start`, `target_end`, `resolution` | forecast metadata (initialization, lead, target window) |
| `Ensemble_spread` | weighted standard deviation of the members' forecasts: disagreement between methods, NOT total forecast uncertainty (use the probabilities for that) |
| `Forecast_<member>`, `Weight_<member>` | each member's forecast and its weight (weights sum to 1); `NaN` = member not available for that station |
| `Forecast_wet_days`, `Climatology_wet_days`, `wet_day_threshold_mm` | wet-day fraction converted to days (rows of `wet_day_fraction` only) |

Subseasonal and dynamical-member rows with empty forecast values carry the reason in `Predictors` (`dry-season (not forecastable)`, `insufficient data`, `no model forecast / too short`); an empty seasonal row means the station had too few years or no usable predictors.

### Skill files
* `hindcast_skill_<variable>_*.csv` (one row per station): `n` years scored, `correlation`, `rmse`, `mae`, `bias`, `rmse_climatology`,
  `rmse_skill` (1 - rmse/rmse_climatology), `n_prob`, `rpss`, `hss` (Heidke, 0 = chance), `brier_ss_below|normal|above` (vs 1/3), `roc_auc_*`
  (0.5 = no discrimination), `crps` (absolute, only for untransformed variables such as temperature). Per-station values rest on ~30
  seasons: read them as patterns, not as guarantees for one station.
* `hindcast_pooled_*.csv`: the same scores pooled over stations, with 90% bootstrap intervals (`rpss_lo_90`, `rpss_hi_90`, ...).
  **A forecast system is only worth using where the lower bound of pooled RPSS is above 0.**
* `method_comparison_*.csv`: RPSS of every method, the ensemble and climatology with intervals, and `rpss_diff_vs_climatology` (paired
  difference with its interval; `diff_lo_90 > 0` = significantly better than climatology).
* `hindcast_reliability_*.csv`: forecast probability bins against observed frequency; a calibrated system has
  `observed_frequency` close to `mean_forecast_prob`. Bins with small `n` are noise.
* `hindcast_event_scores_*.csv` (wet-day runs): accuracy, POD, FAR, CSI, bias score, POFD and Brier skill for the events below-normal and
  above-normal (forecast YES when the probability >= 0.5); `target = ALL` is pooled.
* `subseasonal_skill_summary_*.csv`: pooled RPSS with interval for each variable and lead window.

### Supporting files
`predictor_diagnostics_*.csv` (correlations and `VIF` of the candidate predictors; VIF above 5-10 = collinear), `station_coverage.csv`
(record length, % missing, longest gap, years with >= 90% of days), `station_data_quality_report.json` (everything the loader repaired or
dropped), `station_monthly.csv`, `station_<season>_seasonal_anomalies.csv`, `forecast_run_metadata.json` / `subseasonal_run_metadata.json`
(settings of the run).

## 5. Reading a result: a checklist
1. Open `station_data_quality_report.json` and `station_coverage.csv`: are the dropped values, unit warnings and gaps what you expect?
2. Open `hindcast_pooled_*.csv` / `method_comparison_*.csv`: is pooled RPSS clearly above 0 (interval too)? If not, the forecast adds
   little beyond climatology; say so when you communicate it.
3. Check `Predictors` / `Weight_*`: do the selected predictors make physical sense (e.g. Nino 3.4 and Indian Ocean for JJAS Ethiopia)?
4. Look at the probabilities, not only `Forecast`: values near 33/33/33 mean "no signal"; 50%+ in one category with positive hindcast skill
   is a real lean. Compare `hindcast_reliability_*.csv` to see whether such probabilities were trustworthy in the past.
5. For subseasonal runs expect low skill at weeks 2-6; trust the multi-week window more than a single week.

## 6. Common messages
| Message | Meaning / fix |
|---|---|
| `Dates like 01/02/2000 are ambiguous` | add `--station-dayfirst` or `--station-monthfirst` (or `--station-date-format`) |
| `Cannot find columns for [...]` | map them: `--station-columns station=ID,date=DAY,lat=Y,lon=X,value=RR` |
| `Predictors for JJAS_2026 are incomplete` | the predictor table stops before the init month: extend it |
| `SST index table not found` | run the SST workflow first or pass `--sst-indices` |
| `Only N training years ...; need >= 15` | too short a record (`--min-train-years` for subseasonal; seasonal needs 15) |
| `dry-season (not forecastable)` | more than 25% of training windows had no rain: terciles undefined, no forecast |
| `init_month ... differs from the forecast's init month` | the dynamical-model file was issued in another month: supply matching files |
| unit warnings ("tenths of degrees / mm") | values look ten times too large: convert the file |

## 7. Limits you should know about
Skill numbers in this package's tests come from synthetic data. The tool uses ocean indices (and persistence for weeks) only unless you supply
dynamical-model files; it does not run a dynamical model. Probabilities assume Gaussian errors on the modelling scale. Grid forecasts for weeks 2-6,
rainfall amounts by GLM, and tree-boosting / deep-learning methods are not implemented. The NetCDF-reading step for real gridded files has not been
run by the author (no xarray in the build environment): run the grid workflow on a small clipped file first.
