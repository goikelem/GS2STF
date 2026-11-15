# Development notes (module map, design decisions, change log)

Started as a restructuring of the flat `SST_Climate_Indices` project into the layout of the project specification (the original SST-index
analysis, SPI/SPEI and SST-rainfall teleconnection code behaves as before: verified output-for-output) and now adds seasonal and subseasonal
forecasting of stations and grids with tercile probabilities, an ensemble, hindcast verification and dynamical-model inputs. What is still
missing is listed under *Not implemented yet*.

```bash
pip install -r requirements.txt
python tests/test_demo.py                 # 1 min: the package works on your machine (synthetic data)
sh examples/run_demo.sh                   # tour of the outputs on synthetic data -> demo_out/
python main.py --help
```
**First run on your own data** (adapt the paths; the example uses `/home/goitomk/temp/input_data`, input files are found by pattern, so
`prcp-chirps-ethio-198101_202609.nc` or a later end date both work):
```bash
# 1. ocean indices + extra predictors from the SST NetCDF  (--fast = quicker validation settings; drop it for the full run)
python main.py --input /home/goitomk/temp/input_data/sst_era5_198101_202610.nc --output /home/goitomk/temp/sst_output --fast
#    -> /home/goitomk/temp/sst_output/indices/all_predictors_monthly.csv   (check indices/dataset_inspection.json first)

# 2. a REAL forecast with the data you have: OND 2026 from data through September (SST runs to October, CHIRPS to September)
python main.py --grid-forecast --grid-variable precipitation --season OND --init-month 9 --target-year 2026 \
    --sst-indices /home/goitomk/temp/sst_output/indices/all_predictors_monthly.csv \
    --grid-region Ethiopia --grid-coarsen 5 --methods ridge,pcr --grid-output /home/goitomk/temp/grid_out

# 3. stations (daily CSV or a folder of CSVs)
python main.py --stations /path/to/stations.csv --station-season OND --station-forecast --init-month 9 --target-year 2026 \
    --sst-indices /home/goitomk/temp/sst_output/indices/all_predictors_monthly.csv --station-output /home/goitomk/temp/station_out

# other workflows
python main.py --index both --classification --spatial-analysis --make-maps    # SPI / SPEI (files from config/datasets.yaml)
python main.py --teleconnection --season JJAS                                  # SST-rainfall teleconnections
```
Tests: `python tests/test_kernels.py; python tests/test_tele_kernels.py; python tests/test_station.py; python tests/test_seasonal.py; python tests/test_ensemble.py; python tests/test_grid.py; python tests/test_predictors.py; python tests/test_subseasonal.py; python tests/test_dynamical.py; python tests/test_glm.py; python tests/test_demo.py; python tests/test_station_messy.py; python tests/test_leakage_e2e.py; python tests/test_robustness.py`

**New here? Read `docs/USER_GUIDE.md`** (which command for which question, output dictionary, how to read results, common messages).
Method descriptions, outputs and assumptions of the original project: `docs/METHODS_original_README.md`.

## Configuration
`config/datasets.yaml` (paths, file names), `config/regions.yaml` (SST index boxes, study regions),
`config/forecast_settings.yaml` (baseline, events, seasons, models, validation split, SPI/SPEI and teleconnection
parameters). `src/settings.py` loads them and exposes the same names the old `config.py` had.
Dataset paths are still the original absolute paths: edit them for your machine.

## Where the old modules went
| Old module | New location |
|---|---|
| `config.py` | `config/*.yaml` + `src/settings.py` |
| `data_processing.py` | `data_ingestion/netcdf_loader.py` (open/inspect), `preprocessing/anomaly.py`, `teleconnections/sst_indices.py` (regional SST) |
| `climate_data.py` | `data_ingestion/netcdf_loader.py`, `data_ingestion/data_validation.py` |
| `indices_calculation.py` | `teleconnections/sst_indices.py`, `events.py`, `enso.py`, `iod.py` |
| `statistical_analysis.py` | `teleconnections/lag_analysis.py`, `teleconnections/index_statistics.py` |
| `teleconnection.py` | `teleconnections/correlation_maps.py` (the `run` workflow), plus helpers in `data_ingestion/`, `preprocessing/`, `export/` |
| `tele_kernels.py` | `teleconnections/tele_kernels.py` |
| `forecasting.py` | `forecasting/machine_learning.py`, `forecasting/sst_forecast.py`, `export/csv_export.py` |
| `model_validation.py` | `verification/skill_scores.py`, `cross_validation.py`, `hindcast.py` |
| `climate_workflow.py`, `index_kernels.py` | `drought_indices/workflow.py`, `drought_indices/index_kernels.py` |
| `spatial_analysis.py`, `climate_maps.py` | `spatial/grid_statistics.py`, `spatial/forecast_maps.py` |
| `visualization.py` | `export/figures.py` |
| NetCDF / GeoTIFF writers | `export/netcdf_export.py`, `export/geotiff_export.py` |

Small renames made when helpers became shared across modules: `_same_grid`/`_full_monthly` -> `same_grid`/`full_monthly`,
`_factor` -> `block_factor`, `_dataset`/`_map_dataset`/`_save` -> `make_dataset`/`make_map_dataset`/`save_netcdf`, `_write_nc` -> `write_nc`.

Extra modules not in the specification: `forecasting/dynamical.py`, `preprocessing/grid_processing.py`, `drought_indices/` (SPI/SPEI, kept because it is part of your project),
`teleconnections/events.py`, `teleconnections/index_statistics.py`, `export/figures.py`.

## Station input handling (what the loader tolerates, and what it refuses)
`--stations` takes one CSV or a FOLDER of CSV files (e.g. one per station; combined, with cross-file duplicates and coordinate disagreements
reported). Tolerated: byte-order mark, UTF-8 or Latin-1, `,` `;` tab or `|` separators (detected from the header), decimal commas (`10,4`),
header spellings such as `Station ID` / `Latitude`, a value column named after the variable (`Rain`, `Tmax` -> variable inferred),
missing tokens (`NA`, `-`, empty, ...), `T`/trace rain (counted as 0 mm and reported), sentinel codes (-99, -999, ...).
Refused, with a message naming the problem (and the offending file in a folder): unrecognised columns (use `--station-columns`),
a date column that is mostly unreadable, and **ambiguous dates**. `01/02/2000` could be 1 Feb or 2 Jan; the loader decides only when some
field exceeds 12 (then it says how it decided) and otherwise stops and asks for `--station-dayfirst`, `--station-monthfirst` or
`--station-date-format '%d.%m.%Y'` instead of guessing. (Ethiopian-calendar dates are not converted: convert them first.)
Counted in `station_data_quality_report.json`: unreadable values (with examples), trace values, dropped dates, negative rain, impossible
values (beyond +-70 degC or 2000 mm/day, set to missing), duplicates, stations with conflicting coordinates, tmin > tmax days. Probable unit
problems (e.g. temperatures in tenths of degrees, rain in tenths of mm) raise a warning. `station_coverage.csv` lists, per station and
variable, the first / last valid date, % missing, longest gap and the number of years with >= 90% of days observed.

## Station pathway (implemented)
`data_ingestion/station_csv_loader.py` reads the long format (Station, Date, Lat, Lon, Variable, Value), the simple format
(Station, Date, lat, long, value) and separate files per variable, with automatic or user-mapped columns, sentinel codes
(-99, -999, ...) to NaN, negative rain to NaN, duplicate and coordinate-conflict handling, and a data-quality report.
`preprocessing/station_processing.py` does daily -> monthly -> seasonal aggregation (cross-year seasons such as ONDJ and
user-defined month lists; precipitation summed, temperature averaged; completeness rules in `config/datasets.yaml`),
anomalies and percent-of-normal against a reference period, and grid-to-station predictor extraction (nearest / bilinear,
with the matched cell and offset kept). Try it: `python main.py --stations obs.csv --station-season JJAS`.
Tests: `python tests/test_station.py`.

## Seasonal forecasting (implemented: four methods + ensemble)
`forecasting/seasonal.py` (engine and methods), `forecasting/ensemble.py` (combination, selection), `precipitation.py` /
`temperature.py` (thin wrappers) and `verification/probabilistic_scores.py` (scoring, method comparison). Run: `python main.py --stations obs.csv --station-season JJAS --station-forecast --init-month 5 --target-year 2027`
(needs the SST index table from the SST workflow, or `--sst-indices`).
* Every forecast carries `init_date`, `lead_months`, `target_start`, `target_end` and `Forecast_Period`. Predictors are
  window means of the indices (R34, WIOD, EIOD, DMI) ending at the initialization month; nothing later is read (tested).
* Methods (`--methods ridge,pcr,cca,rf`, default all four): **ridge** multiple linear regression with predictor subset chosen by exact
  leave-one-out error; **pcr** (EOF/PCA of the predictors + regression); **cca** (reduced-rank canonical correlation with
  cross-validated truncation, multi-target); **rf** (random forest, out-of-bag error scale). Each method uses training years only and
  falls back to climatology for any target where it does not beat climatology out of sample. Precipitation uses a square-root scale.
* Ensemble (`--ensemble-mode skill|equal|select`): weights come only from the members' hindcast RPS in EARLIER seasons
  (climatology competes as a member, shrinking the ensemble when nothing is skilful); `select` picks the best past member.
  Skill/select need the rolling scheme. Output keeps every member's forecast and weight. `method_comparison_*.csv` ranks all
  methods by RPSS with bootstrap intervals and paired differences to climatology. The ensemble is not guaranteed to beat the best
  single method (on a purely linear synthetic signal ridge wins); it mainly protects against picking the wrong one.
* Output: Forecast, Anomaly, below / near / above-normal probabilities, terciles, predictors used, number of training years;
  CSV and Excel in the specification's station format. Grids: `seasonal.forecast_gridded` (numpy arrays; NetCDF export not built).
* Hindcasts: rolling-origin (default) or blocked; terciles and selection from training years only. Scores: correlation, RMSE,
  MAE, bias, RMSE skill vs climatology (and persistence), Heidke, RPSS, Brier skill, ROC area, Gaussian CRPS, reliability
  tables, block-bootstrap confidence intervals.
* Limits: probabilities assume a Gaussian predictive distribution; skill numbers in the tests come from synthetic data with a
  planted signal and say nothing about real skill. Persistence needs monthly target data (done automatically for stations).
  XGBoost, LightGBM, GLM, LSTM/CNN and dynamical-model forecasts are not included (XGBoost/LightGBM were not installed in the build sandbox, so they were not added untested).
Tests: `python tests/test_seasonal.py`, `python tests/test_ensemble.py`. The DataArray wrapper `extract_from_dataarray` is untested (no xarray in the sandbox);
the numpy function it calls is tested.

## Gridded seasonal forecasts (implemented)
`python main.py --grid-forecast --grid-variable precipitation --season JJAS --init-month 5 --target-year 2027 --grid-region Ethiopia`
(also `--bbox LON_MIN LON_MAX LAT_MIN LAT_MAX`, `--clip-file region.geojson|.shp`, `--grid-coarsen N`, `--methods`, `--grid-variable tmax|tmin`).
Pipeline: NetCDF -> monthly grid (existing loader, needs xarray) -> seasonal field (`preprocessing/grid_processing.py`; complete seasons
only, same start/end-year labels as stations) -> optional coarsening and region clipping (`spatial/shapefile_clip.py`: bounding box, named
region, GeoJSON polygons, minimal built-in .shp reader for WGS84 polygons, holes and 0-360 longitudes handled) -> the same methods,
ensemble and leak-free hindcasts as the station pathway, run on every cell (`seasonal.run_grid_forecast`).
Outputs in `outputs/spatial_forecasts/`: NetCDF-3 (scipy only; forecast, anomaly, climatology, tercile probabilities, dominant category,
hindcast RPSS / correlation / RMSE-skill maps, init date / lead / target period as attributes), a multi-band GeoTIFF (EPSG:4326, named bands,
needs only `tifffile`), anomaly / tercile-probability / hindcast-skill PNG maps, method comparison CSV and a metadata JSON.
Station maps: `spatial/station_maps.py`. Scale: ~1,600 cells take ~4 s (ridge) or ~40 s (ridge+pcr+cca); coarsen large grids first.
Tests: `python tests/test_grid.py` (grid aggregation, clipping incl. a hand-built shapefile, forecast, NetCDF/GeoTIFF read-back, maps, CLI function).
Not exercised in the build sandbox: reading real NetCDF files (needs xarray), real shapefiles from GIS software (reader tested only against
a shapefile written to the ESRI spec in the test), cartopy-style basemaps (maps are plain Matplotlib, optional polygon outlines).
Per-cell skill from ~30 seasons is noisy: read the skill maps as patterns, not as cell-level guarantees.

## Ocean predictors, SST as a target, and guarding against chance predictors
* **Extra predictors** (`config/regions.yaml` -> `extra_index_regions`): Nino 1+2, Nino 3, Nino 4, tropical Indian Ocean basin (approximate box),
  tropical North and South Atlantic, plus the derived Atlantic gradient `ATL_GRAD = TNA - TSA`. The SST workflow computes them from the NetCDF
  (or reads them as extra columns of `--from-regional-csv`) and writes `indices/all_predictors_monthly.csv`; the core index files and the
  index-forecasting pipeline are unchanged (disable with `--no-extra-predictors`). Seasonal forecasts use this table when it exists;
  `--predictors R34,DMI,TNA` restricts the candidates, `--sst-indices FILE` overrides the file.
  No textbook centred ONI is added: a centred 3-month mean at month m would use month m+1. The forecast code averages the last
  `--predictor-months` months up to the initialization month instead, which is the leak-free equivalent.
* **Multicollinearity**: every forecast run writes `predictor_diagnostics_*.csv` (correlation matrix + VIF; DMI = WIOD - EIOD is collinear by
  construction, so avoid offering all three to methods that need independent predictors).
* **Chance predictors**: with many candidate predictors, leave-one-out selection finds spurious winners. On pure noise, 9 predictors gave
  non-climatology fits ~50% of the time and RPSS about -0.08 (worse than climatology). A model must now cut climatology's leave-one-out error
  by `--min-gain` (default 0.08) or the station/cell keeps climatology: noise RPSS improved to about -0.04, a strong real signal is unchanged,
  a weak one loses a little. The default was tuned on synthetic data only: re-check it on your real hindcasts.
* **SST target**: `--grid-forecast --grid-variable sst --bbox ...` forecasts seasonal SST (degC; the SST file is global, so clip it).
  The anomaly is the forecast minus the training-year mean, so a long-term warming trend shows up as a positive anomaly.
  Predictors still end at the initialization month, never the verification period.

## Subseasonal (weekly) station forecasts (implemented)
`python main.py --stations daily_obs.csv --subseasonal --init-date 2026-06-01 --lead-weeks 2 4 --predictors R34,DMI,TNA`
(`--lead-weeks 3 6` for weeks 3-6; `--methods ridge,pcr,rf`; `--pool-weeks`, `--recent-weeks`, `--min-train-years`, `--no-weekly-breakdown`).
Code: `forecasting/subseasonal.py`. The forecast is issued on the init date using data BEFORE it; weeks 2-4 means days 8-28 (the
aggregated window) and the run also gives each single week 2, 3, 4 (weekly resolution), each with its own hindcast skill.
* Samples: the same window for every earlier year and for init dates shifted by +/- `pool-weeks` weeks (about 3x more training rows).
  A training row is used only if its whole target window ended before the init date and, in hindcasts, only if it belongs to an
  earlier year (rolling origin, tested by corrupting the test year and everything after it).
* Predictors: ocean indices from COMPLETE months before the init date (a June 1 forecast uses Mar-May) and the station's own anomaly
  over the last `recent-weeks` weeks (persistence). Targets are standardised per calendar shift (precipitation on a square-root scale).
* Errors are estimated by leaving out whole years, not single rows: neighbouring init dates of one year are near-duplicates, and
  row-wise leave-one-out made the first version overconfident (probabilities too sharp, RPSS negative despite positive correlation).
  The closed form was verified against brute-force refits. Seasonal forecasts are unaffected (groups are not used there).
* Dry-season guard: precipitation is not forecast where more than 25% of training windows are dry (terciles undefined); the row says why.
* Output: `subseasonal_forecast_<date>_W<a>-<b>.csv/.xlsx` (one row per station x lead window, with init_date, lead_weeks,
  target_start, target_end), per-station hindcast skill, a pooled skill summary with bootstrap intervals, method comparison, run metadata.
* Expect low skill at weeks 2-6. Only ocean indices and persistence are used (no MJO, no dynamical-model input); synthetic tests show the
  mechanics work, not that real skill exists. Judge real data by the hindcast RPSS interval. Subseasonal forecasts for grids are not built.

## Dynamical-model guidance and bias correction (implemented)
The dynamical model itself is not run here: you export its forecasts (and its re-forecast archive for past years) to CSV, all issued in
the same month as your forecast. `forecasting/dynamical.py` documents the formats; loading fails if an `init_month` column differs from
the forecast's `--init-month` (mixing lead times would silently break the training relationship).
* `--dynamical-indices file.csv` (`season_year,init_month,R34,DMI,...`): the model's forecast of ocean indices for the target season,
  added as predictors `DYN_*`. This is the "forecast SST" input of the specification: what the model predicted at the initialization
  date, never the SST observed during the verification period.
* `--dynamical-stations file.csv` (`Station,season_year,init_month,value`, optional `Variable`; `--dynamical-variable` otherwise):
  the model's forecast of the station variable. It becomes an extra ensemble member `dynamical`: observations regressed on the model
  forecast (removes mean/scale bias, i.e. model-output statistics), fitted on EARLIER years only, kept only if it beats climatology by
  `--min-gain`, with tercile probabilities from the calibrated regression. The skill-weighted ensemble decides how much to trust it.
* Grids accept `--dynamical-indices`; station-level model guidance for grid cells is not built.
* `preprocessing/bias_correction.py` is a utility library (linear scaling, empirical quantile mapping with additive / multiplicative tails,
  per-group correctors such as per calendar month, and `rolling_correct`, which corrects every year with parameters from earlier years
  only). It is tested but not called by the command-line workflows: use it to correct a gridded product (CHIRPS, ERA5) against stations
  or to pre-process model output before writing the CSV files above.
* Ensemble weights now also ignore any member that has not beaten climatology in the hindcast history (previously such members kept
  nearly the weight of climatology). Example from the tests: with a skilful model and ocean predictors that carry no signal, the
  ensemble RPSS rose from 0.19 to 0.25 (best single member 0.34): it still hedges toward climatology.

## Rainfall occurrence (binomial GLM) and event verification (implemented)
`--wet-day-threshold 1.0` adds a forecast of the fraction of days with rain >= 1 mm in the season (`Variable = wet_day_fraction`, plus
`Forecast_wet_days` / `Climatology_wet_days`). `forecasting/glm.py` fits a ridge-penalised binomial GLM (logit link, IRLS), so the forecast stays
inside (0, 1); predictor subset and the "must beat climatology by `--min-gain`" rule work as for every other method; it joins the ensemble as
method `glm` (it refits the model n x subsets times per station, so it is for stations, not big grids). Wet-day seasons need every day
observed (a missing day drops the season), and the threshold is inclusive (>= 1.0 mm counts as wet).
`verification.probabilistic_scores.event_scores` gives accuracy, POD, FAR, CSI, bias score, POFD, hits / misses / false alarms and the Brier
skill score for the events below-normal and above-normal (forecast YES when the category probability >= 0.5), pooled and per station
(`hindcast_event_scores_*.csv`). On the planted synthetic signal GLM and ridge scored the same (mid-range fractions are close to linear);
the GLM's advantage is staying within bounds for very dry or very wet stations.

## Try it on synthetic data
`sh examples/run_demo.sh` writes planted-signal demo inputs (`examples/make_demo_data.py`: ocean predictors + 6 stations, daily rain and Tmax)
and runs a seasonal forecast (rainfall, Tmax, wet-day fraction; ridge, pcr, cca + ensemble) and a weeks 2-4 subseasonal forecast into
`demo_out/`. The planted relationships make skill look far better than real data will deliver: it is a tour of formats and outputs.

## Leakage audit
Besides the component tests, `tests/test_leakage_e2e.py` audits the real command-line workflows end to end: seasonal station forecasts
(ridge, PCR, ensemble weights, wet-day GLM), subseasonal forecasts and gridded forecasts are each run twice, once with data that stop at the
initialization date and once with the future filled with absurd values (observations of the target period, ocean indices after the init
month). The forecast tables must be identical, and a negative control (corrupting an index month inside the allowed window) must change the
grid forecast, so the comparison cannot be blind. Hindcast skill tables are not compared: a hindcast of the target year legitimately sees it.
If you add a predictor or data source, extend this test with it.

## Not implemented yet
No module of the specified layout is a placeholder any more. Still absent: subseasonal forecasts for grids, GLM for rainfall amounts (only occurrence is modelled),
XGBoost / LightGBM / LSTM / CNN for seasonal and weekly targets, atmospheric-circulation and soil-moisture predictors, station-level
model guidance for grids, running any dynamical model, and real-data validation of everything above.

## Deterministic forecasts are means, not medians
Precipitation is modelled on a square-root scale, and squaring a square-root-scale forecast gives the MEDIAN, which is too low for skewed data.
For skewed series a climatology "forecast" showed a negative anomaly (about -16% for a gamma series of shape 2; weekly totals are that skewed).
`seasonal.retransform_mean` now adds the variance term (sqrt: mean^2 + sigma^2; log1p: exp(mean + sigma^2/2) - 1), so `Forecast` is the expected
total and `Anomaly = Forecast - Climatology` has no built-in bias (tests for seasonal and subseasonal). Tercile probabilities were never affected
(terciles are invariant under monotone transforms). Forecasts produced by earlier versions of this package are biased low for skewed rainfall.

## Things to know
* `legacy/` holds `teleconnection_workflow.py` and `test_teleconnection.py`. They target an older teleconnection API
  (`tc.eof`, `tc.mca_permutation`, ...) that no longer exists, are not imported anywhere, and the test file already
  failed before the restructuring. They are kept unchanged for reference.
* `regions.yaml` now also contains **East_Africa** and **Upper_Blue_Nile**. These are approximate bounding boxes added
  from the specification; replace with shapefile clipping when that is built.
* In the old `config.py`, `N_PERM`, `FDR_ALPHA` and `SST_FILE` were defined twice; the later (effective) values were
  kept. Names used only by the legacy workflow live in the `legacy:` block of `forecast_settings.yaml`.
* The gridded (NetCDF) code paths could not be run in the sandbox used for the restructuring (no xarray/netCDF4).
  They were checked statically (all modules import, no undefined names or missing attributes) but not executed.
