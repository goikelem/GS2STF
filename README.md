# S2S Teleconnection Forecasting

Restructured from the flat `SST_Climate_Indices` project into the layout in the project specification.
**This step is a restructuring only: no algorithm was changed.** The SST-index analysis and forecasting,
SPI/SPEI workflow and SST-rainfall teleconnection code behave as before. The new S2S forecasting
capabilities from the specification are listed under *Not implemented yet*.

```bash
pip install -r requirements.txt
python main.py --help
python main.py --from-regional-csv monthly_regional_sst.csv --fast        # SST index forecasting, no NetCDF
python main.py --index both --classification --spatial-analysis --make-maps   # SPI / SPEI
python main.py --teleconnection --season JJAS                              # SST-rainfall teleconnections
python tests/test_kernels.py; python tests/test_tele_kernels.py; python tests/test_station.py; python tests/test_seasonal.py; python tests/test_ensemble.py
```
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

Extra modules not in the specification: `drought_indices/` (SPI/SPEI, kept because it is part of your project),
`teleconnections/events.py`, `teleconnections/index_statistics.py`, `export/figures.py`.

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

## Not implemented yet (placeholder modules containing only a docstring)
`preprocessing/bias_correction`,
`forecasting/subseasonal`, `spatial/station_maps`,
`spatial/shapefile_clip`. Also absent: GLM, XGBoost/LightGBM for seasonal targets, LSTM/CNN, gridded NetCDF/GeoTIFF forecast export, shapefile clipping,
Atlantic / circulation / soil-moisture predictors.

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
