"""Generic phase classification and event detection for ocean-index time series.

Origin: indices_calculation.py."""
from __future__ import annotations

import pandas as pd


def classify_phase(series: pd.Series, threshold: float, smooth: int = 1) -> pd.Series:
    """'positive' (>= +thr), 'negative' (<= -thr), else 'neutral'; optional centred running mean."""
    s = series.rolling(smooth, center=True, min_periods=smooth).mean() if smooth > 1 else series
    ph = pd.Series("neutral", index=series.index, dtype=object)
    ph[s >= threshold] = "positive"
    ph[s <= -threshold] = "negative"
    ph[s.isna()] = "missing"
    return ph


def detect_events(series: pd.Series, threshold: float, min_months: int, strong: float,
                  smooth: int = 1, kind: str = "ENSO") -> pd.DataFrame:
    """Contiguous runs of the same non-neutral phase lasting >= min_months."""
    s = series.rolling(smooth, center=True, min_periods=smooth).mean() if smooth > 1 else series
    ph = classify_phase(series, threshold, smooth)
    rows, i, n = [], 0, len(ph)
    labels = {"ENSO": {"positive": "El Nino", "negative": "La Nina"},
              "IOD": {"positive": "Positive IOD", "negative": "Negative IOD"}}[kind]
    while i < n:
        p = ph.iloc[i]
        if p in ("positive", "negative"):
            j = i
            while j + 1 < n and ph.iloc[j + 1] == p:
                j += 1
            dur = j - i + 1
            if dur >= min_months:
                seg = s.iloc[i:j + 1]
                pk = seg.abs().idxmax()
                peak = float(seg.loc[pk])
                cat = "strong" if abs(peak) >= strong else ("moderate" if abs(peak) >= (threshold + strong) / 2
                                                            else "weak")
                rows.append({"Event": labels[p], "Phase": p, "Start": ph.index[i].strftime("%Y-%m"),
                             "End": ph.index[j].strftime("%Y-%m"), "Duration_months": dur,
                             "Peak_value": round(peak, 3), "Peak_month": pk.strftime("%Y-%m"),
                             "Mean_intensity": round(float(seg.mean()), 3),
                             "Cumulative_intensity": round(float(seg.sum()), 3), "Category": cat})
            i = j + 1
        else:
            i += 1
    cols = ["Event", "Phase", "Start", "End", "Duration_months", "Peak_value", "Peak_month",
            "Mean_intensity", "Cumulative_intensity", "Category"]
    return pd.DataFrame(rows, columns=cols)


def event_summary(ev: pd.DataFrame, n_years: float, label: str) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame({"Event": [], "N_events": []})
    g = ev.groupby("Event").agg(N_events=("Event", "size"), Mean_duration=("Duration_months", "mean"),
                                Max_duration=("Duration_months", "max"),
                                Mean_peak=("Peak_value", "mean"), Strongest_peak=("Peak_value",
                                lambda x: x.loc[x.abs().idxmax()]),
                                Total_months=("Duration_months", "sum"))
    g["Events_per_decade"] = g["N_events"] / n_years * 10
    g["Pct_of_record_in_event"] = g["Total_months"] / (n_years * 12) * 100
    g.insert(0, "System", label)
    return g.round(3).reset_index()
