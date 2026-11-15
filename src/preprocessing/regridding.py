"""Spatial coarsening (block averaging) of gridded fields.

Origin: teleconnection.py."""
from __future__ import annotations

import numpy as np


def block_factor(res_deg: float, target: float) -> int:
    return int(max(1, round(target / res_deg))) if res_deg and np.isfinite(res_deg) else 1


def coarsen_rain(da, target_res: float):
    f = block_factor(float(abs(np.median(np.diff(da["lat"].values)))), target_res)
    out = da.coarsen(lat=f, lon=f, boundary="trim").mean(skipna=True) if f > 1 else da
    return out.values.astype("float64"), out["lat"].values, out["lon"].values, f


def coarsen_array(data, lat, lon, factor: int, min_valid: float = 0.5):
    """Block-average the last two axes (lat, lon) of a numpy array by an integer ``factor``.

    Edge rows/columns that do not fill a block are trimmed.  A block needs at least ``min_valid`` of its cells to be
    finite (otherwise NaN), which keeps coastline / mask cells from being averaged from a single pixel.
    Returns (coarse data, coarse lat, coarse lon) with the block-mean coordinates.
    """
    import warnings
    data = np.asarray(data, dtype=float); lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    f = int(factor)
    if f < 1:
        raise ValueError("factor must be >= 1")
    if f == 1:
        return data, lat, lon
    nl, no = (data.shape[-2] // f) * f, (data.shape[-1] // f) * f
    if nl == 0 or no == 0:
        raise ValueError(f"factor {f} is larger than the grid {data.shape[-2:]}")
    d = data[..., :nl, :no].reshape(data.shape[:-2] + (nl // f, f, no // f, f))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(d, axis=(-3, -1))
    frac = np.isfinite(d).mean(axis=(-3, -1))
    mean = np.where(frac >= min_valid, mean, np.nan)
    return mean, lat[:nl].reshape(-1, f).mean(1), lon[:no].reshape(-1, f).mean(1)
