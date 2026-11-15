"""Data-quality helpers: grid-compatibility check and gap-free monthly time axis.

Origin: climate_data.py."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd


log = logging.getLogger("clim.data")


def same_grid(a, b, tol=1e-4) -> bool:
    return (a.lat.size == b.lat.size and a.lon.size == b.lon.size
            and np.allclose(a.lat.values, b.lat.values, atol=tol)
            and np.allclose(a.lon.values, b.lon.values, atol=tol))


def full_monthly(da, start, end):
    """Select the analysis window and reindex to a gap-free monthly axis (gaps become NaN)."""
    full = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    missing = full.difference(da.indexes["time"])
    if len(missing):
        log.warning("%d monthly time steps missing in %s (filled with NaN): %s%s", len(missing), da.name,
                    [d.strftime("%Y-%m") for d in missing[:6]], " ..." if len(missing) > 6 else "")
    return da.reindex(time=full), [d.strftime("%Y-%m") for d in missing]
