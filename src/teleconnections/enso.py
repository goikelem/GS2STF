"""ENSO (Nino 3.4) event and phase identification.

Origin: indices_calculation.py."""
from __future__ import annotations

import pandas as pd

from src import settings
from src.teleconnections.events import classify_phase, detect_events


def enso_events(r34: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    ev = detect_events(r34, settings.ENSO_THRESHOLD, settings.ENSO_MIN_MONTHS, settings.ENSO_STRONG,
                       settings.ENSO_USE_RUNNING_MEAN, "ENSO")
    ph = classify_phase(r34, settings.ENSO_THRESHOLD, settings.ENSO_USE_RUNNING_MEAN)
    ph = ph.replace({"positive": "El Nino", "negative": "La Nina"})
    # months that are over-threshold but not part of a qualifying (>= min_months) event are neutral
    in_event = pd.Series(False, index=r34.index)
    for _, r in ev.iterrows():
        in_event.loc[pd.Timestamp(r["Start"] + "-01"):pd.Timestamp(r["End"] + "-01")] = True
    ph[(ph.isin(["El Nino", "La Nina"])) & (~in_event)] = "Neutral"
    ph[ph == "neutral"] = "Neutral"
    return ev, ph
