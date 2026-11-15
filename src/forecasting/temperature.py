"""Temperature forecasting (tmax, tmin, tmean): seasonal mean temperature, anomalies and tercile probabilities.

Temperature is modelled untransformed.  See forecasting.seasonal for the method and hindcast protocol.
"""
from __future__ import annotations

from src.forecasting import seasonal

VARIABLES = ("tmax", "tmin", "tmean")


def forecast_stations(seasonal_table, stations, indices, spec, target_year, variable="tmax", monthly=None, **kw):
    if variable not in VARIABLES:
        raise ValueError(f"variable must be one of {VARIABLES}")
    return seasonal.run_station_forecast(seasonal_table, stations, indices, spec, target_year, variable,
                                         transform="none", monthly=monthly, **kw)
