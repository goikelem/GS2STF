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
