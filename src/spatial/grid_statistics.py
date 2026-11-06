"""Spatially explicit statistics on gridded standardized indices (area means, trends, spells, category areas).

Origin: spatial_analysis.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import settings
from src.drought_indices import index_kernels
from src.teleconnections import sst_indices


def subset_region(da, region: str | dict):
    """Bounding-box subset; latitude may be ascending or descending."""
    box = settings.STUDY_REGIONS[region] if isinstance(region, str) else region
    lat_m, lon_m = sst_indices.region_masks(da["lat"].values, da["lon"].values, box["lat"], box["lon"])
    if not lat_m.any() or not lon_m.any():
        raise ValueError(f"Region {region} does not overlap the data grid")
    return da.isel(lat=np.where(lat_m)[0], lon=np.where(lon_m)[0])


def season_mask(time_index: pd.DatetimeIndex, season: str) -> np.ndarray:
    return np.isin(time_index.month, settings.CLIM_SEASONS[season])


def area_grid(da) -> np.ndarray:
    return index_kernels.cell_area_km2(da["lat"].values, da["lon"].values)


def regional_mean(da, weights: str = "area") -> pd.Series:
    """Area-weighted regional mean time series (NaN cells excluded per time step)."""
    a = area_grid(da)[None, :, :]
    v = da.values
    w = np.where(np.isfinite(v), a, 0.0)
    num = np.nansum(np.where(np.isfinite(v), v, 0.0) * w, axis=(1, 2))
    den = w.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.Series(out, index=da.indexes["time"], name=da.name)


def category_area_timeseries(codes: np.ndarray, time_index, lat, lon) -> pd.DataFrame:
    """Percent of valid study area and km^2 in each class for every time step."""
    area = index_kernels.cell_area_km2(lat, lon)[None, :, :]
    valid = codes > 0
    total = (valid * area).sum(axis=(1, 2))
    cols = {"valid_area_km2": total}
    for c, name in index_kernels.CLASS_NAMES.items():
        km2 = ((codes == c) * area).sum(axis=(1, 2))
        cols[f"km2_{name}"] = km2
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"pct_{name}"] = np.where(total > 0, 100 * km2 / total, np.nan)
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(time_index, name="Date"))
    return df


def aggregate_category_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Monthly, seasonal (JJAS/OND/FMAM) and annual means of the area-percent columns."""
    pct = [c for c in df.columns if c.startswith("pct_")]
    out = {"monthly": df[["valid_area_km2"] + pct + [c for c in df.columns if c.startswith("km2_")]].round(3)}
    out["annual"] = df[pct].groupby(df.index.year).mean().round(3)
    out["annual"].index.name = "Year"
    seas = []
    for s, months in settings.CLIM_SEASONS.items():
        if s == "ALL":
            continue
        sub = df[df.index.month.isin(months)]
        if s == "OND" or s == "JJAS" or s == "FMAM":
            g = sub[pct].groupby(sub.index.year).mean()
            g.insert(0, "Season", s)
            seas.append(g)
    out["seasonal"] = pd.concat(seas).round(3)
    out["seasonal"].index.name = "Year"
    return out


def category_frequency_maps(codes: np.ndarray, time_index, season: str = "ALL") -> np.ndarray:
    """% of valid time steps in each class at every cell: array (7, lat, lon)."""
    sel = season_mask(pd.DatetimeIndex(time_index), season)
    sub = codes[sel]
    valid = (sub > 0).sum(axis=0)
    out = np.full((7,) + sub.shape[1:], np.nan, dtype="float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, c in enumerate(range(1, 8)):
            out[i] = np.where(valid > 0, 100.0 * (sub == c).sum(axis=0) / valid, np.nan)
    return out


def spell_maps(z: np.ndarray, threshold: float, below: bool) -> dict[str, np.ndarray]:
    nt = z.shape[0]
    flat = z.reshape(nt, -1)
    res = index_kernels.spell_statistics(flat, threshold, below)
    return {n: v.reshape(z.shape[1:]).astype("float32") for n, v in res.items()}


def persistence_map(z: np.ndarray, threshold: float, below: bool = True) -> np.ndarray:
    """P(event in month t+1 | event in month t) per cell (percent); NaN if no events."""
    hit = (z <= threshold) if below else (z >= threshold)
    ok = np.isfinite(z)
    a, b = hit[:-1] & ok[:-1], hit[1:] & ok[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 100.0 * (a & b).sum(axis=0) / a.sum(axis=0)
    return np.where(a.sum(axis=0) > 0, p, np.nan).astype("float32")


def linear_trend_map(z: np.ndarray, time_index, season: str = "ALL"):
    """Cell-wise OLS trend of the seasonal-mean index (units per decade) and two-sided p-value.

    The p-value assumes independent annual values; serial correlation is not corrected here.
    """
    ti = pd.DatetimeIndex(time_index)
    sel = season_mask(ti, season)
    years = np.unique(ti.year[sel])
    ann = np.full((len(years),) + z.shape[1:], np.nan)
    for i, y in enumerate(years):
        m = sel & (ti.year == y)
        if m.sum() >= max(1, int(0.75 * len(settings.CLIM_SEASONS[season]))):
            ann[i] = np.nanmean(z[m], axis=0)
    x = years.astype(float)
    ok = np.isfinite(ann)
    n = ok.sum(axis=0).astype(float)
    xm = np.where(ok, x[:, None, None], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = np.nanmean(xm, axis=0), np.nanmean(ann, axis=0)
        sxx = np.nansum((xm - mx) ** 2, axis=0)
        sxy = np.nansum((xm - mx) * (ann - my), axis=0)
        slope = sxy / sxx
        resid = ann - (my + slope * (xm - mx))
        s2 = np.nansum(resid ** 2, axis=0) / (n - 2)
        se = np.sqrt(s2 / sxx)
        t = slope / se
    p = 2 * stats.t.sf(np.abs(t), np.maximum(n - 2, 1))
    bad = n < 10
    return (np.where(bad, np.nan, slope * 10).astype("float32"), np.where(bad, np.nan, p).astype("float32"))


def regional_statistics(z_da, label: str, regions=None) -> pd.DataFrame:
    """Mean / std / min / max / percent of months in drought, per region."""
    rows = []
    for r in (regions or settings.STUDY_REGIONS):
        try:
            sub = subset_region(z_da, r)
        except ValueError:
            continue
        s = regional_mean(sub)
        rows.append({"Index": label, "Region": r, "n_cells": int(np.isfinite(sub.isel(time=-1).values).sum()),
                     "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
                     "pct_months_mean_le_-1": 100 * float((s <= settings.DROUGHT_THRESHOLD).mean()),
                     "pct_months_mean_ge_+1": 100 * float((s >= settings.WET_THRESHOLD).mean())})
    return pd.DataFrame(rows).round(3)
