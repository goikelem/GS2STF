"""Publication-quality figures for SST indices, hindcasts and forecasts (headless Agg backend).

Origin: visualization.py."""
from __future__ import annotations

import logging

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import settings


log = logging.getLogger("sst.plots")


matplotlib.use("Agg")


IDX = settings.INDICES


NAMES = {"R34": "Nino 3.4 SST anomaly (R34)", "WIOD": "Western Indian Ocean SST anomaly (WIOD)",
         "EIOD": "Eastern Indian Ocean SST anomaly (EIOD)", "DMI": "Dipole Mode Index (DMI = WIOD - EIOD)"}


WARM, COOL, NEUTRAL = "#c0392b", "#2c6fbb", "#8a8a8a"


MCOL = {"persistence": "#7f7f7f", "climatology": "#bcbd22", "AR": "#17becf", "ARIMA": "#2ca02c", "SARIMA": "#98df8a",
        "VAR": "#8c564b", "RandomForest": "#1f77b4", "XGBoost": "#d62728", "SVR": "#9467bd",
        "GradientBoosting": "#ff7f0e"}


def _style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
                         "figure.dpi": 100, "savefig.dpi": settings.FIG_DPI, "savefig.bbox": "tight"})


def _save(fig, name):
    p = settings.out("figures") / name
    fig.savefig(p)
    plt.close(fig)
    log.info("  figure: %s", p.name)
    return p


def _dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))


def _anom_fill(ax, s, lw=0.6):
    ax.fill_between(s.index, 0, s.values, where=s.values >= 0, color=WARM, alpha=0.75, lw=lw, interpolate=True)
    ax.fill_between(s.index, 0, s.values, where=s.values < 0, color=COOL, alpha=0.75, lw=lw, interpolate=True)
    ax.axhline(0, color="k", lw=0.6)


def fig_timeseries(df):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, k in zip(axes, IDX):
        s = df[k]
        _anom_fill(ax, s)
        ax.plot(s.rolling(12, center=True, min_periods=12).mean(), color="k", lw=1.2, label="12-month running mean")
        ax.set_title(NAMES[k], loc="left")
        ax.set_ylabel("Anomaly (°C)")
        _dates(ax)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Monthly SST-based climate indices", y=0.995, fontsize=13)
    fig.tight_layout()
    return _save(fig, "01_timeseries_all_indices.png")


def fig_climatology(res, clim_sst, baseline):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    mc = res["monthly_clim"]
    for ax, k in zip(axes.ravel(), IDX):
        m, sd = mc[("mean", k)], mc[("std", k)]
        ax.bar(m.index, m.values, color=np.where(m.values >= 0, WARM, COOL), alpha=0.8, label="Mean anomaly")
        ax.errorbar(m.index, m.values, yerr=sd.values, fmt="none", ecolor="k", capsize=3, lw=1,
                    label="±1 s.d. (interannual)")
        ax.set_title(NAMES[k], loc="left")
        ax.set_ylabel("°C")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(list("JFMAMJJASOND"))
    axes[0, 0].legend(loc="upper right")
    fig.supxlabel("Month")
    fig.suptitle("Monthly climatology of anomaly indices: mean and interannual spread", y=1.0)
    fig.tight_layout()
    _save(fig, "02a_monthly_climatology_indices.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in ("R34", "WIOD", "EIOD"):
        ax.plot(clim_sst.index, clim_sst[k], marker="o", label=k)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Regional mean SST (°C)")
    ax.set_title(f"Regional SST climatology ({baseline[0]}–{baseline[1]}), cos(lat)-weighted")
    ax.legend()
    return _save(fig, "02b_monthly_climatology_SST.png")


def fig_annual(res):
    a = res["annual"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, k in zip(axes, IDX):
        m = a[("mean", k)]
        ax.bar(m.index, m.values, color=np.where(m.values >= 0, WARM, COOL), alpha=0.85)
        z = np.polyfit(m.index, m.values, 1)
        ax.plot(m.index, np.polyval(z, m.index), "k--", lw=1.2, label=f"OLS trend {z[0] * 10:+.2f} °C/decade")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(NAMES[k], loc="left")
        ax.set_ylabel("Annual mean (°C)")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Annual-mean anomalies and linear trends", y=0.995)
    fig.tight_layout()
    return _save(fig, "03_annual_variability.png")


def _event_plot(s, ev, thr, ttl, fname, ylabel, smooth=1, strong=None):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ss = s.rolling(smooth, center=True, min_periods=smooth).mean() if smooth > 1 else s
    ax.fill_between(ss.index, 0, ss.values, where=ss.values >= thr, color=WARM, alpha=0.85, interpolate=True, lw=0)
    ax.fill_between(ss.index, 0, ss.values, where=ss.values <= -thr, color=COOL, alpha=0.85, interpolate=True, lw=0)
    mid = (ss.values < thr) & (ss.values > -thr)
    ax.fill_between(ss.index, 0, ss.values, where=mid, color=NEUTRAL, alpha=0.5, interpolate=True, lw=0)
    for sgn in (1, -1):
        ax.axhline(sgn * thr, color="k", ls="--", lw=0.8)
    if strong:
        for sgn in (1, -1):
            ax.axhline(sgn * strong, color="k", ls=":", lw=0.8)
    for _, r in ev.iterrows():
        a, b = pd.Timestamp(r["Start"] + "-01"), pd.Timestamp(r["End"] + "-01") + pd.DateOffset(months=1)
        ax.axvspan(a, b, color=WARM if r["Phase"] == "positive" else COOL, alpha=0.12, lw=0)
        if r["Category"] == "strong":
            pk = pd.Timestamp(r["Peak_month"] + "-01")
            ax.annotate(pk.strftime("%Y"), (pk, r["Peak_value"]), xytext=(0, 6 if r["Peak_value"] > 0 else -12),
                        textcoords="offset points", ha="center", fontsize=7)
    ax.set_title(ttl, loc="left")
    ax.set_ylabel(ylabel)
    _dates(ax)
    return _save(fig, fname)


def fig_events(df, events):
    _event_plot(df["R34"], events["enso_events"], settings.ENSO_THRESHOLD,
                f"ENSO classification: Nino 3.4 anomaly ({settings.ENSO_USE_RUNNING_MEAN}-month running mean), "
                f"±{settings.ENSO_THRESHOLD} °C, ≥{settings.ENSO_MIN_MONTHS} consecutive months; shading = events",
                "04_ENSO_event_classification.png", "Nino 3.4 anomaly (°C)", settings.ENSO_USE_RUNNING_MEAN,
                settings.ENSO_STRONG)
    _event_plot(df["DMI"], events["iod_events"], settings.IOD_THRESHOLD,
                f"Indian Ocean Dipole: DMI, ±{settings.IOD_THRESHOLD} °C, ≥{settings.IOD_MIN_MONTHS} "
                "consecutive months; shading = events (red = positive IOD, blue = negative IOD)",
                "05a_IOD_event_classification.png", "DMI (°C)", 1, settings.IOD_STRONG)


def fig_iod_seasonal(res):
    t = res["iod_seasonal"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(t.index, t["Mean_DMI"], color=np.where(t["Mean_DMI"] >= 0, WARM, COOL), alpha=0.8)
    axes[0].errorbar(t.index, t["Mean_DMI"], yerr=t["Std_DMI"], fmt="none", ecolor="k", capsize=3)
    axes[0].set_title("DMI: monthly mean and s.d.", loc="left")
    axes[0].set_ylabel("°C")
    w = 0.4
    axes[1].bar(t.index - w / 2, t["Pct_positive_IOD"], w, color=WARM, label="Positive IOD")
    axes[1].bar(t.index + w / 2, t["Pct_negative_IOD"], w, color=COOL, label="Negative IOD")
    axes[1].set_title("Share of years in an IOD event phase, by calendar month", loc="left")
    axes[1].set_ylabel("% of years")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Month")
    fig.tight_layout()
    return _save(fig, "05b_IOD_seasonal_behaviour.png")


def fig_corr(res):
    mats = {"All months": res["corr"], **{k: v for k, v in res["seasonal_mats"].items()}}
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4.2))
    for ax, (t, m) in zip(axes, mats.items()):
        sns.heatmap(m.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax,
                    cbar=False, linewidths=0.5)
        ax.set_title(t)
    fig.suptitle("Pearson correlation matrices (monthly; seasonal means)", y=1.03)
    fig.tight_layout()
    return _save(fig, "06_correlation_heatmaps.png")


def fig_leadlag(res, df):
    ll = res["lead_lag"]
    pairs = [("R34", "DMI"), ("R34", "WIOD"), ("R34", "EIOD"), ("WIOD", "EIOD"), ("R34", "R34"), ("DMI", "DMI")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, (x, y) in zip(axes.ravel(), pairs):
        if x == y:
            ac = res["acf_pacf"]
            lags = ac.index.values
            ax.bar(lags - 0.2, ac[(x, "ACF")], 0.4, label="ACF", color=COOL)
            ax.bar(lags + 0.2, ac[(x, "PACF")], 0.4, label="PACF", color=WARM)
            ax.axhline(ac.attrs.get("conf95", 0.09), color="k", ls="--", lw=0.7)
            ax.axhline(-ac.attrs.get("conf95", 0.09), color="k", ls="--", lw=0.7)
            ax.set_title(f"{x}: ACF / PACF", loc="left")
            ax.set_xlabel("Lag (months)")
            ax.legend()
            continue
        g = ll[(ll["X"] == x) & (ll["Y"] == y)].sort_values("Lag_months")
        ax.plot(g["Lag_months"], g["Correlation"], color="k")
        sig = g["p_value"] < 0.05
        ax.scatter(g["Lag_months"][sig], g["Correlation"][sig], s=14, color=WARM, zorder=3, label="p<0.05 (N_eff)")
        ax.axvline(0, color="grey", lw=0.8)
        ax.axhline(0, color="grey", lw=0.8)
        b = g.loc[g["Correlation"].abs().idxmax()]
        ax.annotate(f"max |r|={b['Correlation']:.2f}\nat lag {int(b['Lag_months'])}", (b["Lag_months"], b["Correlation"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=8)
        ax.set_title(f"corr({x}(t), {y}(t+lag))", loc="left")
        ax.set_xlabel(f"Lag (months); lag>0: {x} leads")
        ax.legend(loc="lower left")
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Correlation")
    fig.suptitle("Cross-correlation, lead–lag relationships and autocorrelation", y=1.0)
    fig.tight_layout()
    return _save(fig, "07_leadlag_crosscorrelation.png")


def fig_hindcast(hc, sel):
    m = hc.merge(sel[["Index", "Lead", "Selected_Model"]], left_on=["index", "lead"], right_on=["Index", "Lead"])
    m = m[m["model"] == m["Selected_Model"]]
    leads = [1, 3, 6, 12]
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for i, k in enumerate(IDX):
        for j, h in enumerate(leads):
            ax = axes[i, j]
            g = m[(m["index"] == k) & (m["lead"] == h)].sort_values("valid")
            ax.plot(g["valid"], g["obs"], color="k", lw=1.2, label="Observed")
            ax.plot(g["valid"], g["pred"], color=WARM, lw=1.1, label="Hindcast")
            r = np.corrcoef(g["obs"], g["pred"])[0, 1] if len(g) > 3 else np.nan
            mdl = g["model"].iloc[0] if len(g) else "-"
            ax.set_title(f"{k} | lead +{h} | {mdl} | r={r:.2f}", fontsize=9, loc="left")
            ax.axhline(0, color="grey", lw=0.5)
            if j == 0:
                ax.set_ylabel("°C")
            ax.xaxis.set_major_locator(mdates.YearLocator(4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Observations vs. independent hindcasts of the selected models (by valid date)", y=1.0)
    fig.tight_layout()
    return _save(fig, "08_observed_vs_hindcast.png")


def fig_performance(perf, sel):
    out = []
    for per in ("selection", "test"):
        fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True)
        for j, k in enumerate(IDX):
            for m, g in perf[(perf["Index"] == k) & (perf["Period"] == per)].groupby("Model"):
                g = g.sort_values("Lead")
                axes[0, j].plot(g["Lead"], g["RMSE"], marker="o", ms=3, color=MCOL.get(m), label=m,
                                lw=2.2 if m in ("persistence", "climatology") else 1.2,
                                ls="--" if m in ("persistence", "climatology") else "-")
                axes[1, j].plot(g["Lead"], g["Corr"], marker="o", ms=3, color=MCOL.get(m), label=m,
                                lw=2.2 if m in ("persistence", "climatology") else 1.2,
                                ls="--" if m in ("persistence", "climatology") else "-")
            axes[0, j].set_title(k, loc="left")
            axes[0, j].set_ylabel("RMSE (°C)" if j == 0 else "")
            axes[1, j].set_ylabel("Correlation" if j == 0 else "")
            axes[1, j].set_xlabel("Lead (months)")
            axes[1, j].axhline(0, color="grey", lw=0.6)
            axes[1, j].set_xticks(range(1, 13))
        axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        ttl = "model-selection period" if per == "selection" else "independent test period"
        fig.suptitle(f"Forecast skill by model and lead time – {ttl}", y=1.0)
        fig.tight_layout()
        out.append(_save(fig, f"09_model_performance_{per}.png"))
    # best-model map
    piv = sel.pivot(index="Index", columns="Lead", values="Selected_Model").reindex(IDX)
    codes = {m: i for i, m in enumerate(sorted(set(piv.values.ravel())))}
    fig, ax = plt.subplots(figsize=(11, 3.2))
    sns.heatmap(piv.replace(codes).astype(float), annot=piv.values, fmt="", cmap="tab20", cbar=False, ax=ax,
                annot_kws={"fontsize": 7}, linewidths=0.5)
    ax.set_title("Selected model by index and lead time (lowest selection-period RMSE, parsimony rule)")
    out.append(_save(fig, "09b_selected_models.png"))
    return out


def fig_forecast(df, fcdf, years=6):
    init = pd.Timestamp(fcdf["Initialization_Date"].iloc[0])
    fig_all, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax_all, k in zip(axes.ravel(), IDX):
        g = fcdf[fcdf["Index"] == k].sort_values("Lead_Month")
        d = pd.to_datetime(g["Date"])
        for fig_, ax in ((None, ax_all), ("single", None)):
            if fig_ == "single":
                fig1, ax = plt.subplots(figsize=(10, 4.5))
            h = df[k].loc[init - pd.DateOffset(years=years):]
            ax.plot(h.index, h.values, color="k", lw=1.4, label="Observed")
            ax.plot(pd.concat([pd.Series([h.iloc[-1]], index=[h.index[-1]]), pd.Series(g["Forecast_Value"].values, index=d)]),
                    color=WARM, lw=1.8, marker="o", ms=3.5, label="Forecast")
            ax.fill_between(d, g["Lower_Bound"], g["Upper_Bound"], color=WARM, alpha=0.2,
                            label=f"{int(g['Interval_Level'].iloc[0] * 100)}% prediction interval")
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(init, color="grey", ls=":", lw=1)
            ax.set_ylabel("Anomaly (°C)")
            ax.set_title(f"{NAMES[k]}", loc="left", fontsize=10)
            ax.xaxis.set_major_locator(mdates.YearLocator(1, month=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
            ax.legend(loc="upper left", fontsize=8)
            if fig_ == "single":
                ns = (g["Validated_Skill"] == "yes").sum()
                ax.text(0.99, 0.02, f"Initialised {init:%Y-%m}; validated skill at {ns}/12 leads",
                        transform=ax.transAxes, ha="right", fontsize=8, color="dimgray")
                _save(fig1, f"10_forecast_{k}.png")
    fig_all.suptitle(f"12-month forecasts (initialised {init:%B %Y}) with empirical prediction intervals "
                     "– forecasts, not observations", y=1.0)
    fig_all.tight_layout()
    return _save(fig_all, "10_forecast_12_months_all_indices.png")


def make_all(df, res, events, clim, monthly_sst, baseline, hc=None, perf=None, sel=None, fcdf=None):
    _style()
    log.info("Creating figures ...")
    fig_timeseries(df)
    fig_climatology(res, clim, baseline)
    fig_annual(res)
    fig_events(df, events)
    fig_iod_seasonal(res)
    fig_corr(res)
    fig_leadlag(res, df)
    if hc is not None:
        fig_hindcast(hc, sel)
        fig_performance(perf, sel)
    if fcdf is not None:
        fig_forecast(df, fcdf)
