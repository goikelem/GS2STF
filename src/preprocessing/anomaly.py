"""Climatology baselines, monthly anomalies and seasonal / windowed aggregation.

Origin: data_processing.py, teleconnection.py."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import settings


log = logging.getLogger("sst.data")


def baseline_period(index: pd.DatetimeIndex) -> tuple[int, int]:
    s, e = settings.BASELINE_START, settings.BASELINE_END
    if index.min().year <= s and index.max().year >= e:
        n = ((index.year >= s) & (index.year <= e)).sum()
        if n >= 12 * 20:
            return s, e
    log.warning("Baseline %d-%d not adequately covered; using full record %d-%d instead.", s, e,
                index.min().year, index.max().year)
    return int(index.min().year), int(index.max().year)


def monthly_anomalies(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, int]]:
    """Anomalies relative to the monthly climatology over the baseline period.

    Because cos(lat) weights are constant in time, the anomaly of the weighted regional mean equals
    the weighted mean of grid-point anomalies (apart from time-varying ice/land masks).
    """
    s, e = baseline_period(monthly.index)
    base = monthly[(monthly.index.year >= s) & (monthly.index.year <= e)]
    clim = base.groupby(base.index.month).mean()
    clim.index.name = "Month"
    anom = monthly.copy()
    for m in range(1, 13):
        sel = monthly.index.month == m
        anom.loc[sel] = monthly.loc[sel] - clim.loc[m]
    log.info("Anomalies computed vs %d-%d monthly climatology.", s, e)
    return anom, clim, (s, e)


def season_total(arr, ti, year, months):
    """Sum over the season months of ``year``; NaN unless every month is present."""
    rows = [np.where((ti.year == year) & (ti.month == m))[0] for m in months]
    if any(len(r) == 0 for r in rows):
        return None
    block = arr[[r[0] for r in rows]]
    return np.where(np.isfinite(block).all(axis=0), block.sum(axis=0), np.nan)


def window_mean(arr, ti, year, first_month, lead, n_months):
    """Mean of ``n_months`` monthly fields ending ``lead`` months before month ``first_month`` of ``year``."""
    end_abs = year * 12 + (first_month - 1) - 1 - lead            # absolute month index (year*12 + month0)
    rows = []
    for a in range(end_abs - n_months + 1, end_abs + 1):
        r = np.where((ti.year * 12 + ti.month - 1) == a)[0]
        if len(r) == 0:
            return None
        rows.append(r[0])
    block = arr[rows]
    return np.where(np.isfinite(block).all(axis=0), block.mean(axis=0), np.nan)
