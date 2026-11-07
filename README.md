<div align="center">

# 🌊 S2S Teleconnection Forecasting

### A modular climate data pipeline and seasonal forecasting toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7B731?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Restructured-2ECC71?style=for-the-badge)](#-restructuring-notice)
[![Forecasting](https://img.shields.io/badge/Seasonal_Forecasting-Implemented-0EA5E9?style=for-the-badge)](#-seasonal-forecasting)
[![Stations](https://img.shields.io/badge/Station_Pathway-Implemented-8E44AD?style=for-the-badge)](#-station-pathway)
[![Roadmap](https://img.shields.io/badge/Subseasonal-Planned-E67E22?style=for-the-badge)](#-roadmap)

**[🚀 Quick Start](#-quick-start)** ·
**[⚙️ Configuration](#️-configuration)** ·
**[📂 Module Map](#-where-the-old-modules-went)** ·
**[🚉 Stations](#-station-pathway)** ·
**[🌦️ Forecasting](#️-seasonal-forecasting)** ·
**[🗺️ Roadmap](#-roadmap)** ·
**[📌 Dev Notes](#-developer-notes)**

</div>

---

## 📢 Restructuring Notice

> [!IMPORTANT]
> This repository was reorganized from the flat `SST_Climate_Indices` project into a modular, specification-based layout.
> **This step is a restructuring only: no core algorithms were changed.** The SST-index analysis and forecasting, the SPI/SPEI workflow, and the SST-rainfall teleconnection code behave identically to the original project.
> New S2S capabilities are listed under [Roadmap](#-roadmap).

---

## ✨ Highlights

| | Feature | Description |
|:-:|---|---|
| 🌡️ | **SST indices** | Regional SST indices, ENSO / IOD, lag analysis and forecasting |
| 💧 | **Drought indices** | SPI / SPEI workflow with classification, spatial analysis and maps |
| 🔗 | **Teleconnections** | SST–rainfall correlation maps by season |
| 🚉 | **Station pathway** | Robust CSV ingestion, QC, aggregation, anomalies and predictor extraction |
| 🌦️ | **Seasonal forecasting** | Ridge, PCR, CCA and Random Forest, with a skill-weighted ensemble |
| ✅ | **Verification** | Rolling-origin / blocked hindcasts with deterministic and probabilistic scores |

---

## 🚀 Quick Start

```bash
# 📦 Install dependencies
pip install -r requirements.txt

# ❓ Display the help menu
python main.py --help
```

### 🧭 Workflows

```bash
# 1️⃣  SST index forecasting (no NetCDF dependency required)
python main.py --from-regional-csv monthly_regional_sst.csv --fast

# 2️⃣  SPI / SPEI drought indices and spatial analysis
python main.py --index both --classification --spatial-analysis --make-maps

# 3️⃣  SST–rainfall teleconnections
python main.py --teleconnection --season JJAS
```

### 🧪 Run the test suites

```bash
python tests/test_kernels.py
python tests/test_tele_kernels.py
python tests/test_station.py
python tests/test_seasonal.py
python tests/test_ensemble.py
```

> 📖 For the methodology, outputs and assumptions of the original project, see [`docs/METHODS_original_README.md`](docs/METHODS_original_README.md).

---

## ⚙️ Configuration

Configuration lives in the `config/` directory. `src/settings.py` loads it and exposes the same names used by the legacy `config.py`.

| 📄 File | 🎯 Purpose |
|---|---|
| `config/datasets.yaml` | File paths and dataset definitions. ⚠️ Paths remain the original absolute paths; edit them for your machine. |
| `config/regions.yaml` | Regional SST index boxes and study regions, now including **East_Africa** and **Upper_Blue_Nile**. |
| `config/forecast_settings.yaml` | Baselines, target events, seasons, models, validation splits, SPI/SPEI and teleconnection parameters. |

---

## 📂 Where the Old Modules Went

| 🗃️ Old flat module | 🧩 New modular location |
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

### 🔤 Small renames

| Old name | New name |
|---|---|
| `_same_grid`, `_full_monthly` | `same_grid`, `full_monthly` |
| `_factor` | `block_factor` |
| `_dataset`, `_map_dataset`, `_save` | `make_dataset`, `make_map_dataset`, `save_netcdf` |
| `_write_nc` | `write_nc` |

### ➕ Extra modules (not in the specification)

`drought_indices/` (SPI/SPEI, retained from the original project) · `teleconnections/events.py` · `teleconnections/index_statistics.py` · `export/figures.py`

---

## 🚉 Station Pathway

> **Status:** ✅ Implemented

### 📥 Data ingestion: `data_ingestion/station_csv_loader.py`

- 📑 **Formats:** long (`Station, Date, Lat, Lon, Variable, Value`), simple (`Station, Date, lat, long, value`), or separate files per variable
- 🔎 Automatic or user-mapped columns
- 🚫 Sentinel codes (`-99`, `-999`, …) converted to `NaN`
- 🌧️ Negative rainfall filtered
- 🧹 Duplicate and coordinate-conflict handling
- 📊 Complete data-quality report

### 🛠️ Preprocessing: `preprocessing/station_processing.py`

- 📆 **Aggregation:** daily → monthly → seasonal, including cross-year seasons such as `ONDJ` and user-defined month lists
  - Precipitation is **summed**, temperature is **averaged**
  - Completeness rules are defined in `config/datasets.yaml`
- 📉 Anomalies and percent-of-normal against a reference period
- 🎯 Grid-to-station predictor extraction (nearest / bilinear), keeping the matched cell and offset

```bash
# ▶️ Try it
python main.py --stations obs.csv --station-season JJAS

# 🧪 Test it
python tests/test_station.py
```

---

## 🌦️ Seasonal Forecasting

> **Status:** ✅ Implemented

Four core methods plus an ensemble engine, in `forecasting/seasonal.py`, `forecasting/ensemble.py`, the thin wrappers `precipitation.py` / `temperature.py`, and `verification/probabilistic_scores.py`.

```bash
python main.py --stations obs.csv --station-season JJAS --station-forecast --init-month 5 --target-year 2027
```

> [!NOTE]
> Requires the SST index table from the SST workflow, or pass `--sst-indices`.

### 🗂️ Forecast metadata

Every forecast carries `init_date`, `lead_months`, `target_start`, `target_end` and `Forecast_Period`.
Predictors are window means of indices (`R34`, `WIOD`, `EIOD`, `DMI`) ending at the initialization month, so **no future data is read**.

### 🧠 Methods

Select with `--methods ridge,pcr,cca,rf` (default: all four).

| 🏷️ Method | 📝 Description |
|:-:|---|
| 📈 **`ridge`** | Multiple linear regression, predictor subset chosen by exact leave-one-out error |
| 🧬 **`pcr`** | Principal Component Regression (EOF/PCA of predictors, then regression) |
| 🔀 **`cca`** | Reduced-rank canonical correlation with cross-validated truncation (multi-target) |
| 🌲 **`rf`** | Random Forest with out-of-bag error scaling |

Each method uses **training years only** and falls back to **climatology** for any target where it does not beat climatology out of sample. Precipitation uses a square-root scale.

### 🤝 Ensemble

Select with `--ensemble-mode skill|equal|select`.

| Mode | Behaviour |
|:-:|---|
| ⚖️ `skill` | Weights derive strictly from members' hindcast `RPS` in *earlier* seasons. Climatology competes as a member, shrinking the ensemble when nothing is skillful. |
| 🟰 `equal` | Equal weights across members |
| 🏆 `select` | Picks the best past member |

Outputs preserve every member's forecast and weight.

> [!NOTE]
> The ensemble protects against picking the wrong single model; it does not outperform the single best model on purely linear synthetic signals.

### 📤 Outputs

Forecasts, anomalies, below/near/above-normal probabilities, terciles, predictors used and training-year counts, exported as CSV and Excel in station format.
Gridded outputs use `seasonal.forecast_gridded` (numpy arrays; NetCDF export is not built).

### 🔁 Hindcasts and skill scores

Rolling-origin (default) or blocked hindcasts.

| 🎯 Deterministic | 🎲 Probabilistic |
|---|---|
| Correlation, RMSE, MAE, bias | RPSS, Brier skill score |
| RMSE skill vs. climatology and persistence | ROC area, Gaussian CRPS |
| Heidke score | Reliability tables |
| | Block-bootstrap confidence intervals |

### ⚠️ Limitations

- Probabilities assume a **Gaussian** predictive distribution.
- Skill numbers in tests use synthetic data with a planted signal and **do not reflect real-world skill**.
- Persistence requires monthly target data (handled automatically for stations).
- XGBoost, LightGBM, GLM, LSTM/CNN and dynamical-model forecasts are excluded (not installed in the build sandbox).

### 🧪 Tests

```bash
python tests/test_seasonal.py
python tests/test_ensemble.py
```

> The DataArray wrapper `extract_from_dataarray` is untested because the sandbox lacked `xarray`; the underlying numpy function is tested.

---

## 🗺️ Roadmap

### 🚧 Placeholder modules (docstrings only)

| 📦 Module | Status |
|---|:-:|
| `preprocessing/bias_correction` | 🟡 Planned |
| `forecasting/subseasonal` | 🟡 Planned |
| `spatial/station_maps` | 🟡 Planned |
| `spatial/shapefile_clip` | 🟡 Planned |

### ❌ Also not yet available

- GLM, XGBoost / LightGBM for seasonal targets, LSTM / CNN
- Gridded NetCDF / GeoTIFF forecast export
- Shapefile clipping
- Atlantic, circulation and soil-moisture predictors

---

## 📌 Developer Notes

| 🔖 Topic | 📝 Details |
|---|---|
| 🗄️ **`legacy/`** | Holds `teleconnection_workflow.py` and `test_teleconnection.py`, which target an older teleconnection API (`tc.eof`, `tc.mca_permutation`, …) that no longer exists. They are unimported and kept for reference only. |
| 🌍 **Regions** | `regions.yaml` contains approximate bounding boxes for **East_Africa** and **Upper_Blue_Nile**, added from the specification. Replace them with shapefile clipping when implemented. |
| 🧾 **Legacy config values** | In the old `config.py`, `N_PERM`, `FDR_ALPHA` and `SST_FILE` were defined twice; the later (effective) values were kept. Names used only by the legacy workflow live in the `legacy:` block of `forecast_settings.yaml`. |
| 🧰 **Sandbox verification** | Gridded (NetCDF) code paths could not be executed in the build sandbox (`xarray` / `netCDF4` unavailable). They were checked statically (all modules import, no undefined names or missing attributes) but **not executed at runtime**. |

---

<div align="center">

**🌍 Built for climate science · 🌧️ Seasonal outlooks · 🌊 Ocean–atmosphere teleconnections**

<sub>Released under the MIT License.</sub>

</div>
