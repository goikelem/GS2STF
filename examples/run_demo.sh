#!/bin/sh
# Tour of the station workflows on SYNTHETIC data (about 1-3 minutes).  Run from the project root:  sh examples/run_demo.sh
set -e
python examples/make_demo_data.py demo_data

echo "== 1. seasonal JJAS forecast issued 1 June 2026 (init month 5): all methods + ensemble, rainfall occurrence (wet days >= 1 mm)"
python main.py --stations demo_data/stations_long.csv --station-season JJAS --station-forecast \
    --init-month 5 --target-year 2026 --sst-indices demo_data/predictors.csv \
    --methods ridge,pcr,cca --wet-day-threshold 1.0 --station-output demo_out/seasonal

echo "== 2. subseasonal forecast issued 1 June 2026: weeks 2-4 (each single week and the 3-week window)"
python main.py --stations demo_data/stations_long.csv --subseasonal --init-date 2026-06-01 --lead-weeks 2 4 \
    --sst-indices demo_data/predictors.csv --methods ridge,pcr --station-output demo_out/subseasonal

echo "Done. Look at demo_out/seasonal and demo_out/subseasonal (forecast CSV/XLSX, hindcast skill, method comparison, diagnostics)."
