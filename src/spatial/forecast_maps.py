"""Publication-style maps of gridded fields (cartopy when available, plain Matplotlib otherwise).

Origin: climate_maps.py."""
from __future__ import annotations

from pathlib import Path
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from src import settings
from src.drought_indices import index_kernels


log = logging.getLogger("clim.maps")


matplotlib.use("Agg")


CLASS_COLORS = ["#8c510a", "#d8a05a", "#f6e8c3", "#f5f5f5", "#c7eae5", "#5ab4ac", "#01665e"]


INDEX_LEVELS = [-3, -2, -1.5, -1, 1, 1.5, 2, 3]


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:  # pragma: no cover
    HAVE_CARTOPY = False


def _axes(fig, extent):
    wide = (extent[1] - extent[0]) > 200            # global field: Pacific-centred
    if HAVE_CARTOPY:
        proj = ccrs.PlateCarree(central_longitude=180) if wide else ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        if not wide:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        try:
            ax.coastlines(resolution="110m" if wide else "50m", linewidth=0.6)
            if not wide:
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="k")
                ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="k", linewidth=0.4)
        except Exception as e:  # offline: Natural Earth download failed
            log.warning("cartopy features unavailable (%s); drawing without coastlines", e)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.6, linestyle=":")
        gl.top_labels = gl.right_labels = False
        return ax, {"transform": ccrs.PlateCarree()}
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.grid(ls=":", lw=0.4)
    if not wide:
        ax.set_aspect("equal")
    return ax, {}


def _extent(lat, lon):
    return [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]


def map_field(field2d, lat, lon, title, outfile, kind="index", label="", cmap=None, vmin=None, vmax=None,
              formats=("png",), dpi=None, subtitle="", stipple=None):
    """Draw one 2-D field.  kind: 'index' (standardized, diverging bins), 'class' (codes 1..7),
    'sequential' (frequency/duration/severity), 'diverging' (symmetric continuous)."""
    dpi = dpi or settings.FIG_DPI
    ext = _extent(lat, lon)
    fig = plt.figure(figsize=(12, 5.2) if (ext[1] - ext[0]) > 200 else (7.2, 6.2))
    ax, kw = _axes(fig, ext)
    f = np.ma.masked_invalid(np.asarray(field2d, dtype=float))
    if kind == "class":
        cm = ListedColormap(CLASS_COLORS)
        norm = BoundaryNorm(np.arange(0.5, 8.5, 1.0), cm.N)
        f = np.ma.masked_less(f, 1)
        im = ax.pcolormesh(lon, lat, f, cmap=cm, norm=norm, shading="auto", **kw)
        cb = fig.colorbar(im, ax=ax, ticks=range(1, 8), shrink=0.8, pad=0.03)
        cb.ax.set_yticklabels([index_kernels.CLASS_NAMES[i] for i in range(1, 8)], fontsize=8)
    elif kind == "index":
        cm = plt.get_cmap(cmap or "BrBG")
        norm = BoundaryNorm(INDEX_LEVELS, cm.N, extend="both")
        im = ax.pcolormesh(lon, lat, f, cmap=cm, norm=norm, shading="auto", **kw)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, extend="both")
        cb.set_label(label or "Standardized index")
    elif kind == "diverging":
        m = vmax if vmax is not None else float(np.nanmax(np.abs(f))) if f.count() else 1.0
        im = ax.pcolormesh(lon, lat, f, cmap=cmap or "RdBu_r", vmin=-m, vmax=m, shading="auto", **kw)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03); cb.set_label(label)
    else:
        im = ax.pcolormesh(lon, lat, f, cmap=cmap or "YlOrRd", vmin=vmin, vmax=vmax, shading="auto", **kw)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03); cb.set_label(label)
    if stipple is not None and np.any(stipple):          # dots where the result is (FDR-)significant
        ii, jj = np.where(np.asarray(stipple, dtype=bool))
        step = max(1, len(ii) // 3000)
        lon_a, lat_a = np.asarray(lon), np.asarray(lat)
        ax.scatter(lon_a[jj[::step]], lat_a[ii[::step]], s=1.2, c="k", alpha=0.7, linewidths=0, **kw)
    if stipple is not None:                       # e.g. FDR-significant cells
        yy, xx = np.where(np.asarray(stipple, bool) & ~np.ma.getmaskarray(f))
        step = max(1, len(yy) // 4000)
        ax.plot(np.asarray(lon)[xx[::step]], np.asarray(lat)[yy[::step]], "k.", ms=1.2, **kw)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    fig.tight_layout()
    out = []
    for fmt in formats:
        p = Path(outfile).with_suffix("." + fmt)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def stacked_area_plot(df_pct, title, outfile, dpi=None):
    """Stacked time series of the area share (%) in each class."""
    dpi = dpi or settings.FIG_DPI
    fig, ax = plt.subplots(figsize=(11, 3.8))
    cols = [f"pct_{index_kernels.CLASS_NAMES[i]}" for i in range(1, 8)]
    ax.stackplot(df_pct.index, [df_pct[c].fillna(0).values for c in cols], colors=CLASS_COLORS,
                 labels=[index_kernels.CLASS_NAMES[i] for i in range(1, 8)], edgecolor="none")
    ax.set_ylim(0, 100); ax.set_ylabel("% of study area"); ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- seasonal forecast maps
def _category_colors(variable: str):
    """(below, normal, above) base colours: browns/greens for rainfall, blues/reds for temperature."""
    if variable == "precipitation":
        return ("#a6611a", "#9e9e9e", "#018571")
    return ("#2166ac", "#9e9e9e", "#b2182b")


def _blend(base: str, strength: np.ndarray) -> np.ndarray:
    rgb = np.array(matplotlib.colors.to_rgb(base))
    return 1.0 - strength[..., None] * (1.0 - rgb)


def _polygon_lines(ax, polygons):
    for rings in polygons or []:
        for r in rings:
            ax.plot(r[:, 0], r[:, 1], color="k", lw=0.8)


def plot_forecast_maps(result: dict, outdir, variable: str = "precipitation", units: str = "", polygons=None,
                       dpi: int = 200) -> list:
    """Anomaly, tercile-probability and hindcast-skill maps (plain Matplotlib).  Returns the written PNG paths."""
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    lat, lon = np.asarray(result["lat"], float), np.asarray(result["lon"], float)
    order = np.argsort(lat)                                              # draw with south at the bottom
    lat_s = lat[order]
    dy = abs(np.median(np.diff(lat_s))) if len(lat) > 1 else 1.0; dx = abs(np.median(np.diff(lon))) if len(lon) > 1 else 1.0
    extent = (lon.min() - dx / 2, lon.max() + dx / 2, lat_s.min() - dy / 2, lat_s.max() + dy / 2)
    meta = result["meta"]; tag = f"{meta['forecast_period']}_init{meta['init_date'][:7]}"
    sub = f"initialized {meta['init_date']}, lead {meta['lead_months']} month(s), valid {meta['target_start']} to {meta['target_end']}"
    paths = []

    def prep(a):
        return np.asarray(a, float)[order] if np.ndim(a) == 2 else np.asarray(a)[..., order, :]

    def finish(fig, ax, name):
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
        _polygon_lines(ax, polygons)
        p = outdir / name; fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 1. anomaly
    an = prep(result["anomaly"]); lim = np.nanmax(np.abs(an)) if np.isfinite(an).any() else 1.0
    cmap = "BrBG" if variable == "precipitation" else "RdBu_r"
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(an, origin="lower", extent=extent, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"anomaly {units}".strip())
    ax.set_title(f"{variable} anomaly forecast, {meta['forecast_period']}\n{sub}", fontsize=9)
    finish(fig, ax, f"{variable}_{tag}_anomaly.png")

    # 2. most-probable tercile category, shaded by probability
    dom, dp = prep(result["dominant_category"]), prep(result["dominant_probability"])
    cols = _category_colors(variable); img = np.ones(dom.shape + (4,))
    strength = np.clip((dp - 1 / 3) / (0.7 - 1 / 3), 0, 1) * 0.85 + 0.15
    for k in range(3):
        sel = dom == k
        img[sel, :3] = _blend(cols[k], strength[sel])
    img[dom < 0, 3] = 0.0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(cols, ("below normal", "near normal", "above normal"))]
    ax.legend(handles=handles, loc="lower left", fontsize=8, title="most likely category\n(darker = higher probability)",
              title_fontsize=8)
    ax.set_title(f"{variable} tercile forecast, {meta['forecast_period']}\n{sub}", fontsize=9)
    finish(fig, ax, f"{variable}_{tag}_tercile_probability.png")

    # 3. hindcast skill
    if "rpss" in result.get("skill", {}):
        sk = prep(result["skill"]["rpss"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sk, origin="lower", extent=extent, cmap="RdYlBu", vmin=-0.5, vmax=0.5, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="RPSS vs climatology (hindcast)")
        ax.set_title(f"Out-of-sample hindcast skill, {meta['forecast_period']} at lead {meta['lead_months']}\n(positive = beats climatology)", fontsize=9)
        finish(fig, ax, f"{variable}_{tag}_hindcast_rpss.png")
    return paths
