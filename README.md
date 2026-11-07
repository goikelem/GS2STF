Here is a professionally redesigned, highly polished, and visually engaging `README.md` for your **S2S Teleconnection Forecasting** repository.

This version introduces clean typography, well-structured badges, clear visual hierarchy, collapsible/organized sections, and formatted callouts to make your project documentation immediately impressive and easy to navigate.

---

```markdown
<div align="center">

# 🌊 S2S Teleconnection Forecasting

**A robust, modular climate data pipeline and forecasting toolkit restructured for scalability and precision.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Refactored](https://img.shields.io/badge/status-restructured-success.svg)](#)

</div>

---

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

# Run test suites
python tests/test_kernels.py && python tests/test_tele_kernels.py && python tests/test_station.py && python tests/test_seasonal.py && python tests/test_ensemble.py

```

*For detailed methodologies, outputs, and assumptions from the original project, refer to [`docs/METHODS_original_README.md`](https://www.google.com/search?q=docs/METHODS_original_README.md).*

---

## ⚙️ Configuration

Configuration files are centrally managed under the `config/` directory. `src/settings.py` loads these parameters while preserving backwards compatibility with legacy naming conventions.

| Configuration File | Purpose |
| --- | --- |
| `config/datasets.yaml` | File paths and dataset definitions *(Note: Dataset paths default to absolute paths; edit these for your local machine)*. |
| `config/regions.yaml` | Regional SST index boxes and study regions *(Includes **East_Africa** and **Upper_Blue_Nile** bounding boxes)*. |
| `config/forecast_settings.yaml` | Baselines, target events, seasons, models, validation splits, and SPI/SPEI parameters. |

---

## 📂 Architecture: Where the Old Modules Went

The flat project structure has been refactored into a modular layout:

| Old Flat Module | New Modular Location |
| --- | --- |
| `config.py` | `config/*.yaml` + `src/settings.py` |
| `data_processing.py` | `data_ingestion/netcdf_loader.py`, `preprocessing/anomaly.py`, `teleconnections/sst_indices.py` |
| `climate_data.py` | `data_ingestion/netcdf_loader.py`, `data_ingestion/data_validation.py` |
| `indices_calculation.py` | `teleconnections/sst_indices.py`, `events.py`, `enso.py`, `iod.py` |
| `statistical_analysis.py` | `teleconnections/lag_analysis.py`, `teleconnections/index_statistics.py` |
| `teleconnection.py` | `teleconnections/correlation_maps.py` (workflow), plus helpers in `data_ingestion/`, `preprocessing/`, `export/` |
| `tele_kernels.py` | `teleconnections/tele_kernels.py` |
| `forecasting.py` | `forecasting/machine_learning.py`, `forecasting/sst_forecast.py`, `export/csv_export.py` |
| `model_validation.py` | `verification/skill_scores.py`, `cross_validation.py`, `hindcast.py` |
| `climate_workflow.py`, `index_kernels.py` | `drought_indices/workflow.py`, `drought_indices/index_kernels.py` |
| `spatial_analysis.py`, `climate_maps.py` | `spatial/grid_statistics.py`, `spatial/forecast_maps.py` |
| `visualization.py` | `export/figures.py` |
| NetCDF / GeoTIFF writers | `export/netcdf_export.py`, `export/geotiff_export.py` |

---

## 📊 Station Pathway & Seasonal Forecasting

### 1. Station Pathway (Implemented)

* **Ingestion (`data_ingestion/station_csv_loader.py`)**: Supports long format, simple format, and separate files per variable. Automatically handles sentinel codes (`-99`, `-999`), negative precipitation values, duplicate coordinates, and generates quality reports.
* **Aggregation (`preprocessing/station_processing.py`)**: Computes daily-to-monthly-to-seasonal aggregations (including cross-year seasons like `ONDJ`), precipitation sums, temperature averages, anomalies, and grid-to-station predictor extractions.
* **Try it:** `python main.py --stations obs.csv --station-season JJAS`

### 2. Seasonal Forecasting (Implemented)

`forecasting/seasonal.py`, `forecasting/ensemble.py`, and `verification/probabilistic_scores.py` provide a robust forecasting suite:

* **Four Core Methods (`--methods ridge,pcr,cca,rf`)**:
* **Ridge**: Multiple linear regression with leave-one-out feature selection.
* **PCR**: Principal Component Regression (EOF/PCA of predictors).
* **CCA**: Reduced-rank Canonical Correlation Analysis.
* **RF**: Random Forest with out-of-bag error scaling.


* **Ensemble Modes (`--ensemble-mode skill|equal|select`)**: Combines models using hindcast RPSS weights from earlier seasons, protecting against single-model failure.
* **Outputs & Verification**: Generates probabilistic terciles, anomalies, CSV/Excel reports, rolling-origin hindcasts, and comprehensive verification metrics (RPSS, Brier skill, ROC area, CRPS, reliability tables).

---

## 🗺️ Roadmap: Not Implemented Yet

The following placeholder modules and features are reserved for future updates:

* `preprocessing/bias_correction`
* `forecasting/subseasonal`
* `spatial/station_maps` & `spatial/shapefile_clip`
* Advanced machine learning models (XGBoost, LightGBM, GLM, LSTM/CNN)
* Gridded NetCDF/GeoTIFF forecast exports & shapefile clipping

---

## 📌 Developer Notes

* **`legacy/` folder:** Contains archived scripts (`teleconnection_workflow.py`, `test_teleconnection.py`) targeting an older API. Retained strictly for reference.
* **Gridded Data Paths:** NetCDF/xarray code paths were verified statically (imports, signatures, attributes) within the build sandbox environment, but require an active `xarray`/`netCDF4` runtime environment to execute fully.

```

```
