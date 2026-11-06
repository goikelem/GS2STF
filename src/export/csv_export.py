"""CSV export of forecasts.

Origin: forecasting.py."""
from __future__ import annotations

import logging

import pandas as pd

from src import settings
from src.forecasting.machine_learning import IDX


log = logging.getLogger("sst.forecast")


def save_forecasts(fc: pd.DataFrame):
    d = settings.out("forecast")
    cols = ["Date", "Lead_Month", "Index", "Forecast_Value", "Lower_Bound", "Upper_Bound", "Selected_Model",
            "Initialization_Date", "Interval_Level", "N_residuals_for_interval", "Validated_Skill"]
    fc[cols].to_csv(d / "climate_indices_forecast_12_months.csv", index=False)
    for k in IDX:
        fc[fc["Index"] == k][cols].to_csv(d / f"{k}_forecast_12_months.csv", index=False)
    log.info("Saved forecast CSVs to %s", d)


def save_station_forecasts(df, outdir, stem: str):
    """Write station forecasts as CSV and Excel (sheet 'forecast'); returns the two paths."""
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv, xlsx = outdir / f"{stem}.csv", outdir / f"{stem}.xlsx"
    df.round(4).to_csv(csv, index=False)
    df.round(4).to_excel(xlsx, index=False, sheet_name="forecast")
    return csv, xlsx
