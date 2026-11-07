Here is the complete, fully polished, and comprehensive `README.md` containing every section of your project documentation, beautifully formatted for GitHub.

You can copy this entire block directly into your repository's `README.md` file:

```markdown
<div align="center">

# 🌊 S2S Teleconnection Forecasting

**A robust, modular climate data pipeline and forecasting toolkit restructured for scalability and precision.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Refactored](https://img.shields.io/badge/status-restructured-success.svg)](#)

</div>

---

> **⚠️ Restructuring Notice:** This repository has been cleanly reorganized from the flat `SST_Climate_Indices` project into a structured, modular specification layout. **This step is a restructuring only: no core algorithms were changed.** The SST-index analysis and forecasting, SPI/SPEI workflow, and SST-rainfall teleconnection code behave identically to the original project. New S2S forecasting capabilities are listed under *Not implemented yet*.

---

## 🚀 Quick Start

Get up and running with dependencies and CLI commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Display help menu
python main.py --help

# 1. SST index forecasting (no NetCDF dependency required)
python main.py --from-regional-csv monthly_regional_sst.csv --fast

# 2. SPI / SPEI drought indices and spatial analysis
python main.py --index both --classification --spatial-analysis --make-maps

# 3. SST-rainfall teleconnections
python main.py --teleconnection --season JJAS

# Run all test suites
python tests/test_kernels.py
python tests/test_tele_kernels.py
python tests/test_station.py
python tests/test_seasonal.py
python tests/test_ensemble.py

```

*For detailed methodologies, outputs, and assumptions from the original project, refer to [`docs/METHODS_original_README.md`](https://www.google.com/search?q=docs/METHODS_original_README.md).*

---

## ⚙️ Configuration

Configuration files are centrally managed under the `config/` directory. `src/settings.py` loads these parameters while exposing the same names used in the legacy `config.py`.

| Configuration File | Purpose & Details |
| --- | --- |
| `config/datasets.yaml` | File paths and dataset definitions *(Note: Dataset paths remain original absolute paths; edit these for your local machine)*. |
| `config/regions.yaml` | Regional SST index boxes and study regions *(Now includes **East_Africa** and **Upper_Blue_Nile** bounding boxes)*. |
| `config/forecast_settings.yaml` | Baseline settings, target events, seasons, models, validation splits, SPI/SPEI, and teleconnection parameters. |

---

## 📂 Where the Old Modules Went

The flat project structure has been refactored into the following modular layout:

| Old Flat Module | New Modular Location |
| --- | --- |
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

### Small Renames & Extra Modules

* **Renames:** Helper functions shared across modules were standardized: `_same_grid`/`_full_monthly` $\rightarrow$ `same_grid`/`full_monthly`, `_factor` $\rightarrow$ `block_factor`, `_dataset`/`_map_dataset`/`_save` $\rightarrow$ `make_dataset`/`make_map_dataset`/`save_netcdf`, `_write_nc` $\rightarrow$ `write_nc`.
* **Extra modules not in specification:** `drought_indices/` (SPI/SPEI, retained from original project), `teleconnections/events.py`, `teleconnections/index_statistics.py`, `export/figures.py`.

---

## 🚉 Station Pathway (Implemented)

* **Data Ingestion (`data_ingestion/station_csv_loader.py`)**: Supports long format (`Station, Date, Lat, Lon, Variable, Value`), simple format (`Station, Date, lat, long, value`), and separate files per variable. Includes automatic or user-mapped columns, sentinel code handling (`-99`, `-999`, ... $\rightarrow$ `NaN`), negative rain filtering, duplicate/coordinate-conflict handling, and a complete data-quality report.
* **Preprocessing (`preprocessing/station_processing.py`)**: Handles daily $\rightarrow$ monthly $\rightarrow$ seasonal aggregation (including cross-year seasons such as `ONDJ` and user-defined month lists; precipitation summed, temperature averaged; completeness rules defined in `config/datasets.yaml`), anomaly and percent-of-normal calculations against a reference period, and grid-to-station predictor extraction (nearest / bilinear, keeping matched cell and offset).
* **Try it:** `python main.py --stations obs.csv --station-season JJAS`
* **Tests:** `python tests/test_station.py`

---

## 🌦️ Seasonal Forecasting (Implemented)

Four core forecasting methods plus an ensemble engine (`forecasting/seasonal.py`, `forecasting/ensemble.py`, thin wrappers `precipitation.py` / `temperature.py`, and `verification/probabilistic_scores.py`).

* **Run Command:**
```bash
python main.py --stations obs.csv --station-season JJAS --station-forecast --init-month 5 --target-year 2027

```


*(Requires the SST index table from the SST workflow or `--sst-indices`).*

### Core Specifications

* **Metadata:** Every forecast carries `init_date`, `lead_months`, `target_start`, `target_end`, and `Forecast_Period`. Predictors use window means of indices (`R34`, `WIOD`, `EIOD`, `DMI`) ending at the initialization month (no future data read).
* **Methods (`--methods ridge,pcr,cca,rf` — default all four):**
* **`ridge`**: Multiple linear regression with predictor subset chosen by exact leave-one-out error.
* **`pcr`**: Principal Component Regression (EOF/PCA of predictors + regression).
* **`cca`**: Reduced-rank canonical correlation with cross-validated truncation (multi-target).
* **`rf`**: Random Forest with out-of-bag error scaling.
*Each method uses training years only and falls back to climatology for any target where it does not beat climatology out of sample. Precipitation uses a square-root scale.*


* **Ensemble (`--ensemble-mode skill|equal|select`):** Weights derive strictly from members' hindcast `RPS` in *earlier* seasons (climatology competes as a member, shrinking the ensemble when nothing is skillful); `select` picks the best past member. Outputs preserve every member's forecast and weight. *(Note: The ensemble protects against picking the wrong single model rather than outperforming the single best model on purely linear synthetic signals).*
* **Outputs:** Forecasts, anomalies, below/near/above-normal probabilities, terciles, predictors used, and training year counts exported in CSV and Excel station format. Gridded outputs use `seasonal.forecast_gridded` (numpy arrays; NetCDF export not built).
* **Hindcasts & Skill Scores:** Rolling-origin (default) or blocked. Scores include correlation, RMSE, MAE, bias, RMSE skill vs. climatology & persistence, Heidke score, RPSS, Brier skill score, ROC area, Gaussian CRPS, reliability tables, and block-bootstrap confidence intervals.
* **Limitations:** Probabilities assume a Gaussian predictive distribution. Skill numbers in tests use synthetic data with a planted signal and do not reflect real-world skill. Persistence requires monthly target data (handled automatically for stations). XGBoost, LightGBM, GLM, LSTM/CNN, and dynamical-model forecasts are excluded (not installed in the build sandbox).
* **Tests:** `python tests/test_seasonal.py`, `python tests/test_ensemble.py`. *(Note: The DataArray wrapper `extract_from_dataarray` is untested due to the sandbox lacking `xarray`, though the underlying numpy function is tested).*

---

## 🗺️ Roadmap: Not Implemented Yet

Placeholder modules containing only docstrings:

* `preprocessing/bias_correction`
* `forecasting/subseasonal`
* `spatial/station_maps`
* `spatial/shapefile_clip`

**Also absent:** GLM, XGBoost/LightGBM for seasonal targets, LSTM/CNN, gridded NetCDF/GeoTIFF forecast export, shapefile clipping, and Atlantic / circulation / soil-moisture predictors.

---

## 📌 Important Developer Notes

* **`legacy/` Directory:** Holds `teleconnection_workflow.py` and `test_teleconnection.py` targeting an older teleconnection API (`tc.eof`, `tc.mca_permutation`, ...) that no longer exists. They are unimported and kept purely as reference.
* **Regions:** `regions.yaml` contains approximate bounding boxes for **East_Africa** and **Upper_Blue_Nile** added from the specification; replace with shapefile clipping when implemented.
* **Legacy Config Values:** In the old `config.py`, `N_PERM`, `FDR_ALPHA`, and `SST_FILE` were defined twice; the later (effective) values were retained. Names used exclusively by the legacy workflow reside in the `legacy:` block of `forecast_settings.yaml`.
* **Sandbox Verification:** Gridded (NetCDF) code paths could not be executed in the restructuring build sandbox (`xarray`/`netCDF4` unavailable). They were checked statically (all modules import correctly, no undefined names or missing attributes) but not executed at runtime.

```

```
