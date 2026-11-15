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
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings

log = logging.getLogger("s2s.stations")

COLUMN_ALIASES = {                       # compared after lower-casing and removing spaces, '_', '-' and '.'
    "station": ["station", "stationid", "stationname", "stationcode", "stn", "id", "site", "code"],
    "date": ["date", "datetime", "time", "day", "obsdate", "observationdate"],
    "lat": ["lat", "latitude", "lati", "y"],
    "lon": ["lon", "long", "longitude", "longi", "lng", "x"],
    "variable": ["variable", "var", "element", "param", "parameter"],
    "value": ["value", "val", "amount", "obs", "observation"],
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", ".", "m", "missing", "nd"}
TRACE_TOKENS = {"t", "tr", "trace"}                  # trace rainfall: counted as 0 mm (reported)
HARD_LIMITS = {"precipitation": (0.0, 2000.0), "tmax": (-70.0, 70.0), "tmin": (-70.0, 70.0), "tmean": (-70.0, 70.0)}
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


def _key(name) -> str:
    return re.sub(r"[\s_\-\.]+", "", str(name).lstrip("\ufeff").strip().lower())


def _variable_or_none(name):
    try:
        return normalise_variable(name)
    except ValueError:
        return None


def _resolve_columns(df: pd.DataFrame, column_map: dict | None, variable: str | None):
    """Find the columns of each role.  Returns (roles dict, variable inferred from the value column name or None)."""
    norm = {_key(c): c for c in df.columns}
    out, inferred = {}, None
    for role, aliases in COLUMN_ALIASES.items():
        if column_map and role in column_map:
            col = column_map[role]
            if col not in df.columns:
                raise KeyError(f"Mapped column '{col}' for '{role}' not in file; columns are {list(df.columns)}")
            out[role] = col
            continue
        hit = next((norm[_key(a)] for a in aliases if _key(a) in norm), None)
        if hit is not None:
            out[role] = hit
    if "value" not in out:                           # simple format whose value column is named after the variable
        for c in df.columns:
            v = _variable_or_none(_key(c))
            if v is not None and c not in out.values():
                out["value"], inferred = c, v
                break
    missing = [r for r in ("station", "date", "lat", "lon", "value") if r not in out]
    if missing:
        raise KeyError(f"Cannot find columns for {missing}. File columns: {list(df.columns)}. "
                       f"Pass a column mapping (CLI: --station-columns station=ID,date=DAY,lat=Y,lon=X,value=RR).")
    if "variable" not in out and variable is None and inferred is None:
        pass                                          # simple format: the caller's assumption is recorded later
    return out, inferred


def _detect_delimiter(path: Path, sep):
    if sep:
        return sep
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        head = next((ln for ln in fh if ln.strip()), "")
    counts = {d: head.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=lambda d: (counts[d], d == ","))
    return best if counts[best] > 0 else ","


def _read_raw(path: Path, sep):
    delim = _detect_delimiter(path, sep)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(path, sep=delim, dtype=str, encoding=enc, skipinitialspace=True, na_filter=False)
            return raw, delim, enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} (tried UTF-8 and Latin-1)")


def _parse_dates(s: pd.Series, date_format, dayfirst):
    """Dates -> (datetimes, note).  Day-first vs month-first is never guessed: an ambiguous file raises an error."""
    t = s.astype(str).str.strip()
    filled = t != ""
    if date_format:
        return pd.to_datetime(t.where(filled), format=date_format, errors="coerce"), f"user format {date_format}"
    iso = pd.to_datetime(t.where(filled), format="ISO8601", errors="coerce")
    if filled.any() and iso[filled].notna().mean() >= 0.99:
        return iso, "ISO 8601 (year-month-day)"
    m = t.str.extract(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})$")
    if not filled.any() or m[0][filled].notna().mean() < 0.99:
        raise ValueError("Cannot recognise the date format. Use year-month-day (2000-01-31), day/month/year with a 4-digit year, "
                         "or pass the format explicitly (CLI: --station-date-format '%d.%m.%Y').")
    a, b = m[0].astype(float), m[2].astype(float)
    sep_char = m[1].dropna().iloc[0]
    first_big, second_big = bool((a > 12).any()), bool((b > 12).any())
    if first_big and second_big:
        raise ValueError("Dates mix day-first and month-first values (both fields exceed 12 in different rows); fix the file or pass --station-date-format.")
    if first_big:
        order, how = "dmy", "detected: first field exceeds 12"
    elif second_big:
        order, how = "mdy", "detected: second field exceeds 12"
    elif dayfirst is not None:
        order, how = ("dmy" if dayfirst else "mdy"), "chosen by you"
    else:
        raise ValueError("Dates like 01/02/2000 are ambiguous: every first and second field is 12 or less, so day-first and month-first "
                         "both parse. Say which one you mean: CLI --station-dayfirst or --station-monthfirst "
                         "(Python: dayfirst=True/False), or pass --station-date-format.")
    fmt = f"%d{sep_char}%m{sep_char}%Y" if order == "dmy" else f"%m{sep_char}%d{sep_char}%Y"
    return pd.to_datetime(t.where(filled), format=fmt, errors="coerce"), f"{fmt} ({how})"


def _numeric(s: pd.Series, trace_ok=None):
    """Strings -> floats.  Handles decimal commas, missing / trace tokens.  Returns (values, n_trace, unparseable examples)."""
    t = s.astype(str).str.strip(); low = t.str.lower()
    missing = low.isin(MISSING_TOKENS)
    trace = low.isin(TRACE_TOKENS) & (np.zeros(len(t), bool) if trace_ok is None else np.asarray(trace_ok, bool))
    num = pd.to_numeric(t.where(~(missing | trace)), errors="coerce")
    comma = num.isna() & ~(missing | trace) & t.str.fullmatch(r"[+-]?\d+,\d+")
    if comma.any():
        num[comma] = pd.to_numeric(t[comma].str.replace(",", ".", regex=False))
    num[trace] = 0.0
    bad = num.isna() & ~(missing | trace)
    return num.values.astype(float), int(trace.sum()), t[bad]


def load_station_csv(path, column_map: dict | None = None, variable: str | None = None,
                     missing_values=None, date_format: str | None = None, dayfirst: bool | None = None,
                     sep: str | None = None) -> StationData:
    """Read one station CSV (long or simple format) into the tidy observation table.

    Tolerates a byte-order mark, ',' ';' tab or '|' separators, decimal commas, missing tokens (NA, -, ...) and 'T'/trace
    rainfall (0 mm).  Day-first / month-first dates are detected when a field exceeds 12, otherwise you must say which
    (``dayfirst=True/False``); nothing is guessed silently.  Every repair or drop is counted in ``StationData.report``.
    ``variable`` is needed for the simple format unless the value column is named after the variable (Rain, Tmax, ...);
    otherwise 'precipitation' is assumed and the assumption is recorded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Station file not found: {path}")
    if path.is_dir():
        return load_station_directory(path, column_map=column_map, variable=variable, missing_values=missing_values,
                                      date_format=date_format, dayfirst=dayfirst, sep=sep)
    missing_values = settings.STATION_MISSING_VALUES if missing_values is None else tuple(missing_values)
    raw, delim, enc = _read_raw(path, sep)
    raw.columns = [str(c).lstrip("\ufeff").strip() for c in raw.columns]
    cols, inferred = _resolve_columns(raw, column_map, variable)
    rep = {"file": str(path), "rows_read": int(len(raw)), "assumptions": [], "delimiter": repr(delim), "encoding": enc, "warnings": []}

    dates, note = _parse_dates(raw[cols["date"]], date_format, dayfirst)
    rep["date_format"] = note
    df = pd.DataFrame({"station": raw[cols["station"]].astype(str).str.strip(), "date": dates})
    df["lat"], _, bad_lat = _numeric(raw[cols["lat"]]); df["lon"], _, bad_lon = _numeric(raw[cols["lon"]])
    if "variable" in cols and variable is None:
        df["variable"] = raw[cols["variable"]].map(normalise_variable)
    else:
        if variable is None and inferred is not None:
            variable = inferred
            rep["assumptions"].append(f"variable '{inferred}' taken from the name of the value column '{cols['value']}'")
        if variable is None:
            variable = "precipitation"
            rep["assumptions"].append("no variable column or argument: assumed 'precipitation'")
        df["variable"] = normalise_variable(variable)
    df["value"], n_trace, bad_val = _numeric(raw[cols["value"]], (df["variable"] == "precipitation").values)
    rep["trace_rain_set_to_zero"] = n_trace
    rep["unparseable_values_set_to_nan"] = int(len(bad_val))
    if len(bad_val):
        rep["unparseable_value_examples"] = sorted(set(bad_val.tolist()))[:8]
        log.warning("%d value(s) could not be read as numbers (e.g. %s): set to missing", len(bad_val), rep["unparseable_value_examples"][:4])

    bad_date = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    rep["dropped_unparseable_dates"] = bad_date
    df["date"] = df["date"].dt.normalize()
    rep["dates_outside_1800_2100_dropped"] = int(((df["date"].dt.year < 1800) | (df["date"].dt.year > 2100)).sum())
    df = df[(df["date"].dt.year >= 1800) & (df["date"].dt.year <= 2100)]

    sentinel = df["value"].isin(missing_values)
    rep["sentinel_values_set_to_nan"] = int(sentinel.sum())
    df.loc[sentinel, "value"] = np.nan

    neg = (df["variable"] == "precipitation") & (df["value"] < 0)
    rep["negative_precipitation_set_to_nan"] = int(neg.sum())
    df.loc[neg, "value"] = np.nan

    for var in ("tmax", "tmin", "tmean"):                     # unit problems are reported BEFORE the impossible values are blanked
        v = df.loc[df["variable"] == var, "value"].dropna()
        if len(v) > 30 and (v.abs() > HARD_LIMITS[var][1]).mean() > 0.01:
            rep["warnings"].append(f"{var}: {100 * (v.abs() > HARD_LIMITS[var][1]).mean():.0f}% of values exceed {HARD_LIMITS[var][1]:.0f} "
                                   f"(max {v.max():.0f}): values may be in tenths of degrees (divide by 10)")
    v = df.loc[df["variable"] == "precipitation", "value"].dropna()
    if len(v) > 30 and (v > 500).mean() > 0.001:
        rep["warnings"].append(f"precipitation: {100 * (v > 500).mean():.1f}% of daily values exceed 500 mm (max {v.max():.0f}): "
                               f"values may be in tenths of mm (divide by 10)")
    impossible = pd.Series(False, index=df.index)
    for var, (lo, hi) in HARD_LIMITS.items():
        impossible |= (df["variable"] == var) & ((df["value"] > hi) | (df["value"] < lo))
    rep["physically_impossible_values_set_to_nan"] = int(impossible.sum())
    df.loc[impossible, "value"] = np.nan
    for w in rep["warnings"]:
        log.warning("%s", w)

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
    log.info("%s: %d rows kept, %d stations, variables %s (dates: %s)", path.name, len(df), st.shape[0], rep["variables"], note)
    return StationData(df, st, rep)


def load_station_directory(folder, **kwargs) -> StationData:
    """Combine every ``*.csv`` of a folder (e.g. one file per station) as if they were one file.

    All files are read with the same options.  After combining, duplicate (station, date, variable) records across files
    are dropped and each station gets one coordinate pair (median); disagreements between files are reported.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.CSV"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .csv files in {folder}")
    parts, reports = [], {}
    for f in files:
        try:
            d = load_station_csv(f, **kwargs)
        except (ValueError, KeyError) as e:
            raise type(e)(f"{f.name}: {e}") from e
        parts.append(d.observations); reports[f.name] = d.report
    obs = pd.concat(parts, ignore_index=True)
    g = obs.groupby("station")[["lat", "lon"]]                                   # coordinates compared BEFORE duplicates are dropped
    spread = (g.max() - g.min()).max(axis=1)
    conflicts = spread[spread > 1e-3].index.tolist()
    if conflicts:
        log.warning("%d station(s) have different coordinates in different files; median used: %s", len(conflicts), conflicts[:5])
    med = g.median()
    obs["lat"] = obs["station"].map(med["lat"]); obs["lon"] = obs["station"].map(med["lon"])
    dup = obs.duplicated(["station", "date", "variable"], keep="first")
    n_dup = int(dup.sum()); obs = obs[~dup]
    obs = obs.sort_values(["variable", "station", "date"]).reset_index(drop=True)[OBS_COLUMNS]
    st = (obs.dropna(subset=["value"]).groupby("station")
          .agg(lat=("lat", "first"), lon=("lon", "first"), n_obs=("value", "size"),
               start=("date", "min"), end=("date", "max")).reset_index())
    rep = {"folder": str(folder), "n_files": len(files), "duplicate_records_across_files_dropped": n_dup,
           "stations_with_conflicting_coordinates": conflicts, "n_stations": int(st.shape[0]),
           "variables": sorted(obs["variable"].unique().tolist()), "files": reports,
           "warnings": [w for r in reports.values() for w in r.get("warnings", [])]}
    log.info("%s: %d files, %d stations, %d rows", folder, len(files), st.shape[0], len(obs))
    return StationData(obs, st, rep)


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
