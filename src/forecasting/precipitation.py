"""Precipitation forecasting: seasonal totals, anomalies and tercile probabilities per station.

Rainfall totals are modelled on a square-root scale (skewed, non-negative); tercile categories are unchanged by
this monotone transform.  See forecasting.seasonal for the method and the leak-free hindcast protocol.
"""
from __future__ import annotations

from src.forecasting import seasonal

VARIABLE = "precipitation"
TRANSFORM = "sqrt"


def forecast_stations(seasonal_table, stations, indices, spec, target_year, monthly=None, **kw):
    """Returns (forecast table in the specification's station format, verification Hindcast, {name: Hindcast})."""
    return seasonal.run_station_forecast(seasonal_table, stations, indices, spec, target_year, VARIABLE,
                                         transform=kw.pop("transform", TRANSFORM), monthly=monthly, **kw)
