"""Station CSV loader.

Supported layouts
  * long format : Station, Date, Lat, Lon, Variable, Value   (several variables in one file)
  * simple format: Station, Date, lat, long, value            (one variable per file)
  * separate files per variable: ``load_station_files({"precipitation": "rain.csv", "tmax": ...})``

Columns are found automatically from common names, or mapped explicitly with ``column_map``
(keys: station, date, lat, lon, variable, value).  Everything the loader repairs or drops is
counted in ``StationData.report`` -- nothing is silently altered.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings

log = logging.getLogger("s2s.stations")

COLUMN_ALIASES = {
    "station": ["station", "station_id", "stationid", "stn", "id", "site", "name"],
    "date": ["date", "datetime", "time", "day"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "long", "longitude", "lng", "x"],
    "variable": ["variable", "var", "element", "param", "parameter"],
    "value": ["value", "val", "amount", "obs"],
}
VARIABLE_ALIASES = {
    "precipitation": {"precipitation", "precip", "prcp", "pr", "rain", "rainfall", "rr", "p"},
    "tmax": {"tmax", "tx", "tasmax", "maxtemp", "tmax_c", "temperature_max", "maximum_temperature"},
    "tmin": {"tmin", "tn", "tasmin", "mintemp", "tmin_c", "temperature_min", "minimum_temperature"},
    "tmean": {"tmean", "tavg", "tas", "tmp", "temperature", "temp", "tmean_c"},
}
OBS_COLUMNS = ["station", "date", "lat", "lon", "variable", "value"]


@dataclass
class StationData:
    observations: pd.DataFrame            # tidy: station, date, lat, lon, variable, value
    stations: pd.DataFrame                # one row per station: station, lat, lon, n_obs, start, end
    report: dict = field(default_factory=dict)


def normalise_variable(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    for std, aliases in VARIABLE_ALIASES.items():
        if key in aliases or key == std:
            return std
    raise ValueError(f"Unknown variable '{name}'. Known: {sorted(VARIABLE_ALIASES)} (or pass an alias).")


def _resolve_columns(df: pd.DataFrame, column_map: dict | None, need_variable: bool) -> dict:
    lower = {c.strip().lower(): c for c in df.columns}
    out = {}
    for role, aliases in COLUMN_ALIASES.items():
        if column_map and role in column_map:
            col = column_map[role]
            if col not in df.columns:
                raise KeyError(f"Mapped column '{col}' for '{role}' not in file; columns are {list(df.columns)}")
            out[role] = col
            continue
        hit = next((lower[a] for a in aliases if a in lower), None)
        if hit is not None:
            out[role] = hit
    missing = [r for r in ("station", "date", "lat", "lon", "value") if r not in out]
    if missing:
        raise KeyError(f"Cannot find columns for {missing}. File columns: {list(df.columns)}. "
                       f"Pass column_map={{'station': 'ID', ...}}.")
    if need_variable and "variable" not in out:
        raise KeyError("Long-format file needs a variable column (or pass variable=...).")
    return out


def load_station_csv(path, column_map: dict | None = None, variable: str | None = None,
                     missing_values=None, date_format: str | None = None, dayfirst: bool = False,
                     sep: str | None = None) -> StationData:
    """Read one station CSV (long or simple format) into the tidy observation table.

    ``variable`` is required for the simple format (no variable column); if omitted,
    'precipitation' is assumed and the assumption is recorded in the report.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Station file not found: {path}")
    missing_values = settings.STATION_MISSING_VALUES if missing_values is None else tuple(missing_values)
    raw = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c", dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    cols = _resolve_columns(raw, column_map, need_variable=variable is None and "variable" in
                            {c.strip().lower() for c in raw.columns})
    rep = {"file": str(path), "rows_read": int(len(raw)), "assumptions": []}

    df = pd.DataFrame({
        "station": raw[cols["station"]].astype(str).str.strip(),
        "date": pd.to_datetime(raw[cols["date"]], format=date_format, dayfirst=dayfirst, errors="coerce"),
        "lat": pd.to_numeric(raw[cols["lat"]], errors="coerce"),
        "lon": pd.to_numeric(raw[cols["lon"]], errors="coerce"),
        "value": pd.to_numeric(raw[cols["value"]], errors="coerce"),
    })
    if "variable" in cols and variable is None:
        df["variable"] = raw[cols["variable"]].map(normalise_variable)
    else:
        if variable is None:
            variable = "precipitation"
            rep["assumptions"].append("no variable column or argument: assumed 'precipitation'")
        df["variable"] = normalise_variable(variable)

    bad_date = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    rep["dropped_unparseable_dates"] = bad_date
    df["date"] = df["date"].dt.normalize()

    sentinel = df["value"].isin(missing_values)
    rep["sentinel_values_set_to_nan"] = int(sentinel.sum())
    df.loc[sentinel, "value"] = np.nan

    neg = (df["variable"] == "precipitation") & (df["value"] < 0)
    rep["negative_precipitation_set_to_nan"] = int(neg.sum())
    df.loc[neg, "value"] = np.nan

    bad_xy = ~(df["lat"].between(-90, 90) & df["lon"].between(-180, 360))
    rep["dropped_invalid_coordinates"] = int(bad_xy.sum())
    df = df[~bad_xy]

    dup = df.duplicated(["station", "date", "variable"], keep="first")
    rep["duplicate_records_dropped"] = int(dup.sum())
    df = df[~dup]

    # one coordinate pair per station (median); conflicts are reported
    g = df.groupby("station")[["lat", "lon"]]
    spread = (g.max() - g.min()).max(axis=1)
    conflicts = spread[spread > 1e-3].index.tolist()
    rep["stations_with_conflicting_coordinates"] = conflicts
    if conflicts:
        log.warning("%d station(s) have inconsistent coordinates; median used: %s", len(conflicts), conflicts[:5])
    med = g.median()
    df["lat"] = df["station"].map(med["lat"]); df["lon"] = df["station"].map(med["lon"])

    # tmin > tmax on the same day (flag only)
    w = df[df["variable"].isin(["tmin", "tmax"])].pivot_table(index=["station", "date"], columns="variable", values="value")
    rep["days_tmin_above_tmax"] = int((w["tmin"] > w["tmax"]).sum()) if {"tmin", "tmax"} <= set(w.columns) else 0

    df = df.sort_values(["variable", "station", "date"]).reset_index(drop=True)[OBS_COLUMNS]
    st = (df.dropna(subset=["value"]).groupby("station")
          .agg(lat=("lat", "first"), lon=("lon", "first"), n_obs=("value", "size"),
               start=("date", "min"), end=("date", "max")).reset_index())
    rep.update({"rows_kept": int(len(df)), "n_stations": int(st.shape[0]),
                "variables": sorted(df["variable"].unique().tolist())})
    log.info("%s: %d rows kept, %d stations, variables %s", path.name, len(df), st.shape[0], rep["variables"])
    return StationData(df, st, rep)


def load_station_files(files: dict, **kwargs) -> StationData:
    """Load separate files per variable, e.g. {"precipitation": "rain.csv", "tmin": "tn.csv", "tmax": "tx.csv"}."""
    parts = {v: load_station_csv(p, variable=v, **kwargs) for v, p in files.items()}
    obs = pd.concat([d.observations for d in parts.values()], ignore_index=True)
    st = (obs.dropna(subset=["value"]).groupby("station")
          .agg(lat=("lat", "median"), lon=("lon", "median"), n_obs=("value", "size"),
               start=("date", "min"), end=("date", "max")).reset_index())
    rep = {"files": {v: d.report for v, d in parts.items()}, "n_stations": int(st.shape[0]),
           "variables": sorted(obs["variable"].unique().tolist())}
    return StationData(obs, st, rep)
