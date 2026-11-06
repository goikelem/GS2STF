"""Central settings, loaded from config/*.yaml.

The YAML files are the single source of truth.  This module exposes the same upper-case
names the original config.py had, so code reads ``settings.BASELINE_START`` etc.  Values can
still be overridden at run time (main.py does this for --input / --output / --fast).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("S2S_CONFIG_DIR", PROJECT_ROOT / "config"))


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _boxes(d: dict) -> dict:
    return {k: {"lat": tuple(v["lat"]), "lon": tuple(v["lon"])} for k, v in d.items()}


_DS = _load("datasets.yaml")
_RG = _load("regions.yaml")
_FS = _load("forecast_settings.yaml")

# ---------------------------------------------------------------- paths (SST / index workflow)
INPUT_NC = _path(_DS["sst_workflow"]["input_nc"])
OUTPUT_DIR = _path(_DS["sst_workflow"]["output_dir"])
SUBDIRS = list(_FS["output"]["subdirs"])


def out(sub: str) -> Path:
    """Return (and create) an output sub-directory."""
    p = OUTPUT_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- baseline, regions, events, seasons
BASELINE_START = _FS["baseline"]["start"]
BASELINE_END = _FS["baseline"]["end"]
MIN_DAYS_FRACTION = _FS["baseline"]["min_days_fraction"]

REGIONS = _boxes(_RG["index_regions"])

ENSO_THRESHOLD = _FS["events"]["enso_threshold"]
ENSO_MIN_MONTHS = _FS["events"]["enso_min_months"]
ENSO_USE_RUNNING_MEAN = _FS["events"]["enso_use_running_mean"]
ENSO_STRONG = _FS["events"]["enso_strong"]
IOD_THRESHOLD = _FS["events"]["iod_threshold"]
IOD_MIN_MONTHS = _FS["events"]["iod_min_months"]
IOD_STRONG = _FS["events"]["iod_strong"]

SEASONS = {k: list(v) for k, v in _FS["seasons"]["sst"].items()}
CLIM_SEASONS = {k: list(v) for k, v in _FS["seasons"]["gridded"].items()}
CLIM_SEASONS["ALL"] = list(range(1, 13))
TELE_SEASONS = {**CLIM_SEASONS, **{k: list(v) for k, v in _FS["seasons"]["teleconnection_extra"].items()}}

# ---------------------------------------------------------------- index forecasting
_F = _FS["index_forecasting"]
INDICES = list(_F["indices"])
HORIZONS = list(_F["horizons"])
TRAIN_START = _F["train_start"]
TRAIN_END = _F["train_end"]
VALIDATION_START = _F["validation_start"]
VALIDATION_END = _F["validation_end"]
CV_FOLDS = _F["cv_folds"]
REFIT_EVERY = _F["refit_every"]
SELECTION_FRACTION = _F["selection_fraction"]
PARSIMONY_TOL = _F["parsimony_tol"]
RF_TREES = _F["rf_trees"]
FAST_MODE = _F["fast_mode"]
N_LAGS = _F["n_lags"]
ROLL_WINDOWS = list(_F["roll_windows"])
MODELS = list(_F["models"])
PI_LEVEL = _F["pi_level"]
RANDOM_SEED = _F["random_seed"]
N_JOBS = _F["n_jobs"]

FIG_DPI = _FS["output"]["fig_dpi"]
FORCE_RECOMPUTE = _FS["output"]["force_recompute"]

# ---------------------------------------------------------------- gridded workflow (SPI/SPEI, teleconnections)
_G = _DS["gridded_workflow"]
CLIM_INPUT_DIR = _path(_G["input_dir"])
CLIM_OUTPUT_DIR = _path(_G["output_dir"])
PRCP_FILE, TMIN_FILE, TMAX_FILE = _G["prcp_file"], _G["tmin_file"], _G["tmax_file"]
SST_FILE = _G["sst_file"]
ENSO_EXCEL = _G["enso_excel"]
PRCP_VAR, TMIN_VAR, TMAX_VAR = _G["prcp_var"], _G["tmin_var"], _G["tmax_var"]
ANALYSIS_START, ANALYSIS_END = _G["analysis_start"], _G["analysis_end"]

REF_START = BASELINE_START      # one reference period for SPI, SPEI and SST anomalies
REF_END = BASELINE_END

STUDY_REGIONS = _boxes(_RG["study_regions"])
DEFAULT_REGION = _RG["default_region"]

_D = _FS["drought_indices"]
SCALES = list(_D["scales"])
PET_METHOD = _D["pet_method"]
SPEI_DISTRIBUTION = _D["spei_distribution"]
SPI_DISTRIBUTION = _D["spi_distribution"]
PRCP_UNITS = _D["prcp_units"]
MIN_POSITIVE_SAMPLES = _D["min_positive_samples"]
MIN_REF_SAMPLES = _D["min_ref_samples"]
SPI_CLIP = _D["spi_clip"]
BLOCK_LAT = _D["block_lat"]
DROUGHT_THRESHOLD = _D["drought_threshold"]
WET_THRESHOLD = _D["wet_threshold"]
SPEI_NORMAL_FALLBACK = _D["spei_normal_fallback"]
CLASS_NAMES = {int(k): v for k, v in _D["class_names"].items()}

_T = _FS["teleconnection"]
LAGS = list(_T["lags"])
SST_RES_DEG = _T["sst_res_deg"]
RAIN_RES_DEG = _T["rain_res_deg"]
PREDICTOR_MONTHS = _T["predictor_months"]
OUTLOOK_LEAD = _T["outlook_lead"]
ANALOGUE_TOP_K = _T["analogue_top_k"]
N_PERM = _T["n_perm"]
FDR_ALPHA = _T["fdr_alpha"]

# ---------------------------------------------------------------- legacy-only names
_L = _FS["legacy"]
TELE_LAGS = list(_L["tele_lags"])
SST_COARSEN_DEG = _L["sst_coarsen_deg"]
RAIN_COARSEN_DEG = _L["rain_coarsen_deg"]
MCA_MODES = _L["mca_modes"]
ANALOGUE_K = _L["analogue_k"]
DEFAULT_PREDICTOR = dict(_L["default_predictor"])
ANALOGUE_LAT = tuple(_RG["analogue_lat"])

# ---------------------------------------------------------------- station observations
_ST = _DS["stations"]
STATION_DIR = _path(_ST["directory"])
STATION_MISSING_VALUES = tuple(_ST["missing_values"])
STATION_MIN_DAY_FRACTION = dict(_ST["min_day_fraction"])
STATION_MIN_REF_YEARS = _ST["min_ref_years"]
STATION_SEASON_YEAR_LABEL = _ST["season_year_label"]
STATION_SEASONS = {k: list(v) for k, v in _FS["seasons"]["station"].items()}
