"""Station forecast maps: stations coloured by anomaly or by most-likely tercile category."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_station_forecasts(df, outdir, variable: str | None = None, polygons=None, dpi: int = 200) -> list:
    """Two maps per variable from a forecast table (output of forecasting.seasonal.forecast): anomaly (diverging colours)
    and most-likely tercile category (marker size grows with its probability)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for var in ([variable] if variable else sorted(df["Variable"].dropna().unique())):
        d = df[df["Variable"] == var].dropna(subset=["lat", "lon"])
        if d.empty:
            continue
        period = d["Forecast_Period"].iloc[0]
        cmap = "BrBG" if var == "precipitation" else "RdBu_r"
        lim = max(np.nanmax(np.abs(d["Anomaly"])), 1e-9)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(d["lon"], d["lat"], c=d["Anomaly"], cmap=cmap, vmin=-lim, vmax=lim, s=80, edgecolor="k")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="forecast anomaly")
        ax.set_title(f"{var} anomaly forecast, {period}", fontsize=10)
        paths.append(_finish(fig, ax, polygons, outdir / f"stations_{var}_{period}_anomaly.png", dpi))
        P = d[["P_below", "P_normal", "P_above"]].values
        ok = ~np.isnan(P).any(1); P, dd = P[ok], d[ok]
        k = P.argmax(1); pm = P.max(1)
        colors = np.array(["#a6611a", "#9e9e9e", "#018571"] if var == "precipitation" else ["#2166ac", "#9e9e9e", "#b2182b"])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(dd["lon"], dd["lat"], c=colors[k], s=40 + 400 * (pm - 1 / 3), edgecolor="k")
        for c, lab in zip(colors, ("below normal", "near normal", "above normal")):
            ax.scatter([], [], c=c, edgecolor="k", label=lab)
        ax.legend(fontsize=8, title="most likely category\n(larger = higher probability)", title_fontsize=8)
        ax.set_title(f"{var} tercile forecast, {period}", fontsize=10)
        paths.append(_finish(fig, ax, polygons, outdir / f"stations_{var}_{period}_tercile.png", dpi))
    return paths


def _finish(fig, ax, polygons, path, dpi):
    for rings in polygons or []:
        for r in rings:
            ax.plot(r[:, 0], r[:, 1], color="k", lw=0.8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path
