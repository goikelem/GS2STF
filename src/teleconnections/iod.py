"""Indian Ocean Dipole (DMI) event and phase identification.

Origin: indices_calculation.py."""
from __future__ import annotations

import pandas as pd

from src import settings
from src.teleconnections.events import classify_phase, detect_events


def iod_events(dmi: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    ev = detect_events(dmi, settings.IOD_THRESHOLD, settings.IOD_MIN_MONTHS, settings.IOD_STRONG, 1, "IOD")
    ph = classify_phase(dmi, settings.IOD_THRESHOLD, 1)
    ph = ph.replace({"positive": "Positive IOD", "negative": "Negative IOD", "neutral": "Neutral"})
    in_event = pd.Series(False, index=dmi.index)
    for _, r in ev.iterrows():
        in_event.loc[pd.Timestamp(r["Start"] + "-01"):pd.Timestamp(r["End"] + "-01")] = True
    ph[(ph.isin(["Positive IOD", "Negative IOD"])) & (~in_event)] = "Neutral"
    return ev, ph
