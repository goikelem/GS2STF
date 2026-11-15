# Global Seasonal 2 Sub-seasonal Teleconnection Forecasting (GS2STF) Model

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen)

Seasonal and subseasonal forecasting of **rainfall, temperature, rainfall occurrence and sea-surface temperature** from ocean–atmosphere teleconnections (ENSO, the Indian Ocean Dipole, tropical Atlantic and Pacific indices), for **stations and gridded fields**, with **tercile probabilities** and an out-of-sample **hindcast verification record** for every forecast.

It was built for East Africa (Ethiopia, Tigray, the Upper Blue Nile basin), but nothing in the code is specific to one region: regions, seasons and predictors are set in YAML files.

> **Status.** A research prototype. All 105 automated tests use synthetic data with planted signals, so they show that the code works and does not leak future information, **not** that forecasts are skilful on your data. Validate on your own hindcasts before using any output operationally (see [Limitations](#limitations)).

---

## What it does

| Horizon | Question | Command |
|---|---|---|
| Subseasonal (weeks 2–4, 3–6, or any single week) | What will the next few weeks be like? | `--stations … --subseasonal --init-date … --lead-weeks 2 4` |
| Seasonal, station | What will JJAS / OND / ONDJ… be like at each station? | `--stations … --station-forecast --init-month … --target-year …` |
| Seasonal, gridded | Same, as maps (rainfall, Tmax, Tmin, SST) | `--grid-forecast --grid-variable … --season …` |
| Rainfall occurrence | How many days will be wet? | add `--wet-day-threshold 1.0` |
| Ocean indices | What will Niño 3.4 and the IOD do over 12 months? | `python main.py --input sst.nc` |
| Drought history | Was this season drier than normal (SPI / SPEI)? | `--index spi \| spei \| both` |
| Teleconnection analysis | Which ocean regions relate to my rainfall, at which lag? | `--teleconnection` |

**Methods:** ridge regression, principal-component regression, CCA, random forest, binomial GLM (occurrence), an optional calibrated dynamical-model member, and a skill-weighted multi-model **ensemble**, all benchmarked against **climatology**.

**Verification:** RPSS, Heidke, Brier skill, ROC area, CRPS, correlation / RMSE / bias, reliability tables, accuracy / POD / FAR / CSI for events, with 90% block-bootstrap intervals.

**Outputs:** CSV and Excel forecast tables, NetCDF and multi-band GeoTIFF grids, anomaly / tercile-probability / hindcast-skill maps, and machine-readable run metadata.

## Design principles

- **No future data.** Predictors use only data available at the initialization date; terciles, model selection, bias correction and ensemble weights use training years only. This is tested end to end: every workflow is run with clean data and again with the future replaced by absurd values, and the forecasts must be identical (`tests/test_leakage_e2e.py`).
- **Rolling-origin validation.** Each past year is forecast from earlier years only. Random splits are never used.
- **Climatology is the benchmark.** A model must beat climatology's leave-one-out error by a margin (`--min-gain`, default 8%) at each station or grid cell, otherwise that location keeps climatology. A forecast labelled `climatology` means "no predictor showed skill here".
- **Nothing is guessed silently.** Ambiguous dates stop the run; every repair of the input data is counted in a quality report.

## Quick start

```bash
git https://github.com/goikelem/GS2STF.git
cd GS2STF
pip install -r requirements.txt

python tests/test_demo.py        # about a minute: synthetic data, checks the install
sh examples/run_demo.sh          # a tour of the outputs on synthetic data -> demo_out/
python main.py --help
```

The demo's relationships are *planted*, so its skill looks far better than real data will give. Use it to learn the formats, not to judge the method.

### A first run on your own data

Run everything from the repository root. Paths below are examples.

```bash
# 1. Ocean predictors from a global SST NetCDF (--fast = quicker validation; drop it for the full run)
python main.py --input data/sst/sst_era5.nc --output outputs/sst --fast
#    -> outputs/sst/indices/all_predictors_monthly.csv

# 2. A gridded seasonal forecast: OND rainfall from data through September
python main.py --grid-forecast --grid-variable precipitation \
    --season OND --init-month 9 --target-year 2026 \
    --sst-indices outputs/sst/indices/all_predictors_monthly.csv \
    --grid-region Ethiopia --grid-coarsen 5 --methods ridge,pcr \
    --grid-output outputs/grid

# 3. Station forecasts (a daily CSV, or a folder of CSV files)
python main.py --stations data/stations/ \
    --station-season OND --station-forecast --init-month 9 --target-year 2026 \
    --sst-indices outputs/sst/indices/all_predictors_monthly.csv \
    --wet-day-threshold 1.0 --station-output outputs/stations

# 4. Weeks 2-4 for a given issue date
python main.py --stations data/stations/ --subseasonal \
    --init-date 2026-11-16 --lead-weeks 2 4 \
    --sst-indices outputs/sst/indices/all_predictors_monthly.csv \
    --station-output outputs/subseasonal
```

`data/` and `outputs/` are git-ignored, so your data and results are never committed.

## Input data

**Station observations** (daily, CSV): long format `Station,Date,Lat,Lon,Variable,Value`, or simple format `Station,Date,lat,long,value`. A single file or a folder of files. Rainfall in mm/day, temperature in °C.

The loader tolerates a byte-order mark, UTF-8 or Latin-1, `,` `;` tab or `|` separators, decimal commas, `NA` / `-` / sentinel codes, and `T` (trace) rainfall. It **refuses** ambiguous day/month dates until you say which (`--station-dayfirst` or `--station-monthfirst`), and it warns about probable unit problems (for example temperatures in tenths of degrees). Every run writes `station_data_quality_report.json` and `station_coverage.csv`.

**Gridded data** (NetCDF with time, latitude, longitude): rainfall (for example CHIRPS), Tmin and Tmax (for example ERA5 / CHIRTS), SST (for example ERA5). File names are matched by pattern, so `prcp-chirps-ethio-198101_202609.nc` is found even if the configured name has a different end date.

**Ocean predictors**: a monthly table of index anomalies (R34, WIOD, EIOD, DMI, Niño 1+2 / 3 / 4, tropical Indian Ocean, tropical North and South Atlantic, Atlantic gradient), written by the SST workflow. You can supply your own columns.

**Optional**: dynamical-model exports (CSV, same initialization month as the forecast), polygon files (`.geojson` / `.shp`, WGS84) for clipping, and `ENSO-Years.xlsx` for ENSO composites.

## Reading a forecast

Each forecast row has the expected value, its anomaly, **three tercile probabilities** (below / near / above normal), the tercile cut points, the predictors used, the ensemble weights, and the initialization date, lead and target period.

```text
Station, lat, lon, Forecast_Period, Variable, Forecast, Anomaly,
Climatology, P_below, P_normal, P_above, Tercile_lower, Tercile_upper,
Predictors, N_train_years, Method, init_date, lead_months, target_start, target_end, …
```

Before trusting a forecast, check the hindcast record (`hindcast_pooled_*.csv`, `method_comparison_*.csv`): the **lower end of the pooled RPSS interval should be above 0**. Probabilities near 33 / 33 / 33 mean "no signal".

## Documentation

| Document | Contents |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | which command for which question, the output dictionary, how to read results, common messages |
| [`docs/DEVELOPMENT_NOTES.md`](docs/DEVELOPMENT_NOTES.md) | module map, design decisions and every fix made during development |
| [`docs/METHODS_original_README.md`](docs/METHODS_original_README.md) | methods of the original SST-index project |
| `python main.py --help` | every option with its default |

## Repository layout

```text
main.py                  single entry point for all workflows
config/                  datasets.yaml, forecast_settings.yaml, regions.yaml
src/
  data_ingestion/        station CSV loader, NetCDF loader, validation
  preprocessing/         station and grid processing, anomalies, regridding, bias correction
  teleconnections/       SST indices, ENSO / IOD events, lag analysis, correlation maps
  forecasting/           seasonal, subseasonal, ensemble, GLM, dynamical, index forecasting
  verification/          hindcasts, cross-validation, skill and probabilistic scores
  spatial/               forecast and station maps, polygon clipping, grid statistics
  export/                NetCDF, GeoTIFF, CSV / Excel, figures
  drought_indices/       SPI / SPEI workflow
examples/                synthetic demo data generator and demo script
tests/                   automated tests, including the end-to-end leakage audit
docs/                    user guide and development notes
legacy/                  two superseded files kept for reference (not used)
```

## Configuration

Defaults live in three YAML files in `config/`: dataset paths and file names (`datasets.yaml`), seasons, methods and thresholds (`forecast_settings.yaml`), and ocean and study regions (`regions.yaml`). To add a season, add a line such as `NDJF: [11, 12, 1, 2]` under `seasons.station`. Set `S2S_CONFIG_DIR` to use a different config folder.

## Testing

```bash
python tests/test_demo.py              # quick install check
python tests/test_leakage_e2e.py       # end-to-end leakage audit
for t in tests/test_*.py; do python "$t"; done
```

Each file prints `PASS` or `FAIL` per test. If you add a predictor or data source, extend `tests/test_leakage_e2e.py` so the audit covers it.

## Limitations

- **Skill on real data is unverified.** All tests use synthetic data. Real skill will be lower and varies by region and season.
- **Reading real NetCDF files** has not been exercised by the original author's build environment. Try the grid workflow on a small clipped region first.
- Default thresholds (for example `--min-gain 0.08`) were set on synthetic experiments: re-check them on your hindcasts.
- No dynamical climate model is run; subseasonal forecasts use only ocean indices and persistence (no MJO).
- Probabilities assume Gaussian errors on the modelling scale (square root for rainfall); check `hindcast_reliability_*.csv`.
- **Not implemented:** subseasonal forecasts for grids, rainfall amounts by GLM (only occurrence), XGBoost / LightGBM / LSTM / CNN for seasonal and weekly targets, atmospheric-circulation and soil-moisture predictors, Ethiopian-calendar date conversion.
- The `East_Africa` and `Upper_Blue_Nile` regions are approximate boxes: use a polygon for exact limits.

## Data sources

The package reads, but does not distribute, products such as CHIRPS (rainfall), CHIRTS and ERA5 (temperature), and ERA5 (SST). Please follow each provider's terms and cite them when you publish results.

## Contributing

Issues and pull requests are welcome. Please run the full test suite and the leakage audit before submitting, and describe any change that can alter forecast values.

## License

No license has been chosen yet. Add a `LICENSE` file (for example MIT or Apache-2.0) before making the repository public, and update this section.
