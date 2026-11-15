"""Descriptive statistics and variability analysis of the SST indices.

Origin: statistical_analysis.py."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src import settings
from src.teleconnections.lag_analysis import acf, best_lags, correlation_matrix, lead_lag, pacf, season_series, seasonal_correlations


log = logging.getLogger("sst.stats")


IDX = settings.INDICES


def monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.month)
    out = pd.concat({"mean": g.mean(), "std": g.std(), "min": g.min(), "max": g.max()}, axis=1)
    out.index.name = "Month"
    return out


def annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    a = df.groupby(df.index.year)
    out = pd.concat({"mean": a.mean(), "std": a.std(), "max": a.max(), "min": a.min()}, axis=1)
    out.index.name = "Year"
    return out


def seasonal_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sname, months in settings.SEASONS.items():
        for k in df.columns:
            s = season_series(df[k], months)
            rows.append({"Season": sname, "Index": k, "Mean": s.mean(), "Std": s.std(),
                         "Min": s.min(), "Min_year": s.idxmin(), "Max": s.max(), "Max_year": s.idxmax(),
                         "Pct_positive": 100 * (s > 0).mean(), "N_years": len(s)})
    return pd.DataFrame(rows).round(3)


def trend_analysis(df: pd.DataFrame, monthly_sst: pd.DataFrame) -> pd.DataFrame:
    """OLS trends (per decade) with autocorrelation-adjusted significance, for indices and absolute SST."""
    rows = []
    sources = [(k, df[k], "anomaly index") for k in df.columns]
    sources += [(f"{k}_SST", monthly_sst[k], "absolute SST") for k in monthly_sst.columns]
    for name, s, kind in sources:
        s = s.dropna()
        t = (s.index.year + (s.index.month - 0.5) / 12).values
        lr = stats.linregress(t, s.values)
        resid = s.values - (lr.intercept + lr.slope * t)
        r1 = acf(resid, 1)[1]
        ne = float(np.clip(len(s) * (1 - r1) / (1 + r1), 3, len(s)))
        tstat = lr.slope / lr.stderr
        p_adj = float(2 * stats.t.sf(abs(tstat) * np.sqrt((ne - 2) / (len(s) - 2)), ne - 2))
        # Mann-Kendall-type robust slope on annual means
        ann = s.groupby(s.index.year).mean()
        ts = stats.theilslopes(ann.values, ann.index.values)
        tau, p_mk = stats.kendalltau(ann.index.values, ann.values)
        rows.append({"Series": name, "Type": kind, "Slope_per_decade": lr.slope * 10, "Std_err_per_decade": lr.stderr * 10,
                     "R2": lr.rvalue ** 2, "p_value_naive": lr.pvalue, "Lag1_resid_autocorr": r1,
                     "N_eff": ne, "p_value_AR1_adjusted": p_adj,
                     "Significant_5pct_adjusted": bool(p_adj < 0.05),
                     "TheilSen_slope_per_decade_annual": ts[0] * 10, "MannKendall_tau_annual": tau,
                     "MannKendall_p_annual": p_mk, "N_months": len(s)})
    return pd.DataFrame(rows).round(5)


def basic_variability(df: pd.DataFrame, monthly_sst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in df.columns:
        s = df[k].dropna()
        row = {"Index": k, "Mean": s.mean(), "Std": s.std(), "Skewness": stats.skew(s), "Kurtosis": stats.kurtosis(s),
               "Min": s.min(), "Min_date": s.idxmin().strftime("%Y-%m"), "Max": s.max(),
               "Max_date": s.idxmax().strftime("%Y-%m"),
               "Pct_months_positive": 100 * (s > 0).mean(), "Pct_months_negative": 100 * (s < 0).mean(),
               "Lag1_autocorr": acf(s.values, 1)[1]}
        if k in monthly_sst:   # CV is only meaningful for absolute SST (anomalies have ~zero mean)
            a = monthly_sst[k].dropna()
            row["SST_mean_degC"], row["SST_std_degC"] = a.mean(), a.std()
            row["SST_CV_percent"] = 100 * a.std() / a.mean()
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def extremes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    rows = []
    for k in df.columns:
        s = df[k].dropna()
        z = (s - s.mean()) / s.std()
        for kind, sel in (("positive", s.nlargest(n)), ("negative", s.nsmallest(n))):
            for d, v in sel.items():
                rows.append({"Index": k, "Type": kind, "Date": d.strftime("%Y-%m"), "Value": round(v, 3),
                             "Z_score": round(z.loc[d], 2)})
    return pd.DataFrame(rows)


def timescale_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Share of variance on interannual (1-7 yr), decadal-plus (>7 yr) and sub-annual scales via 12-/84-month smoothing."""
    rows = []
    for k in df.columns:
        s = df[k].dropna()
        lp12 = s.rolling(12, center=True, min_periods=12).mean()
        lp84 = s.rolling(84, center=True, min_periods=84).mean()
        v = s.var()
        rows.append({"Index": k, "Total_var": v,
                     "Sub_annual_pct": 100 * (s - lp12).var() / v,
                     "Interannual_pct": 100 * (lp12 - lp84).var() / v if lp84.notna().any() else np.nan,
                     "Decadal_plus_pct": 100 * lp84.var() / v if lp84.notna().any() else np.nan})
    return pd.DataFrame(rows).round(3)


def decadal_means(df: pd.DataFrame) -> pd.DataFrame:
    d = df.groupby((df.index.year // 10) * 10).mean()
    d.index.name = "Decade_start"
    return d.round(3)


def acf_pacf_table(df: pd.DataFrame, nlags: int = 36) -> pd.DataFrame:
    rows = {}
    for k in df.columns:
        x = df[k].dropna().values
        rows[(k, "ACF")] = acf(x, nlags)
        rows[(k, "PACF")] = pacf(x, nlags)
    out = pd.DataFrame(rows)
    out.index.name = "Lag"
    out.attrs["conf95"] = 1.96 / np.sqrt(len(df))
    return out


def iod_seasonal_behaviour(dmi: pd.Series, iod_phase: pd.Series, iod_events: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"DMI": dmi, "phase": iod_phase}).dropna()
    g = d.groupby(d.index.month)
    out = pd.DataFrame({"Mean_DMI": g["DMI"].mean(), "Std_DMI": g["DMI"].std(),
                        "Pct_positive_IOD": g["phase"].apply(lambda x: 100 * (x == "Positive IOD").mean()),
                        "Pct_negative_IOD": g["phase"].apply(lambda x: 100 * (x == "Negative IOD").mean()),
                        "Pct_neutral": g["phase"].apply(lambda x: 100 * (x == "Neutral").mean())})
    if not iod_events.empty:
        pk = pd.to_datetime(iod_events["Peak_month"] + "-01").dt.month
        st = pd.to_datetime(iod_events["Start"] + "-01").dt.month
        out["N_event_peaks"] = pk.value_counts().reindex(range(1, 13), fill_value=0).values
        out["N_event_onsets"] = st.value_counts().reindex(range(1, 13), fill_value=0).values
    out.index.name = "Month"
    return out.round(3)


def strong_event_years(enso_ev: pd.DataFrame, iod_ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ev, sys in ((enso_ev, "ENSO"), (iod_ev, "IOD")):
        if ev.empty:
            continue
        for _, r in ev[ev["Category"] == "strong"].iterrows():
            rows.append({"System": sys, "Event": r["Event"], "Peak_year": int(r["Peak_month"][:4]),
                         "Peak_month": r["Peak_month"], "Peak_value": r["Peak_value"],
                         "Start": r["Start"], "End": r["End"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # co-occurrence of strong ENSO and IOD events within +-6 months of peak
    out["Co_occurring_with"] = ""
    for i, r in out.iterrows():
        pk = pd.Timestamp(r["Peak_month"] + "-01")
        oth = out[(out["System"] != r["System"])]
        hits = [f"{o['Event']} ({o['Peak_month']})" for _, o in oth.iterrows()
                if abs((pd.Timestamp(o["Peak_month"] + "-01") - pk).days) <= 183]
        out.at[i, "Co_occurring_with"] = "; ".join(hits)
    return out.sort_values("Peak_month").reset_index(drop=True)


def enso_iod_cooccurrence(en_ph: pd.Series, io_ph: pd.Series) -> pd.DataFrame:
    d = pd.concat([en_ph.rename("ENSO"), io_ph.rename("IOD")], axis=1).dropna()
    return pd.crosstab(d["ENSO"], d["IOD"], margins=True)


def run_analysis(idx_df: pd.DataFrame, monthly_sst: pd.DataFrame, events: dict) -> dict:
    d = settings.out("analysis")
    df = idx_df[IDX].dropna(how="all")
    res = {}
    res["monthly_clim"] = monthly_climatology(df)
    res["annual"] = annual_stats(df)
    res["seasonal"] = seasonal_stats(df)
    res["trends"] = trend_analysis(df, monthly_sst)
    res["variability"] = basic_variability(df, monthly_sst)
    res["extremes"] = extremes(df)
    res["timescales"] = timescale_variance(df)
    res["decadal"] = decadal_means(df)
    res["acf_pacf"] = acf_pacf_table(df)
    res["corr"], res["corr_p"] = correlation_matrix(df)
    res["lead_lag"] = lead_lag(df.dropna())
    res["best_lags"] = best_lags(res["lead_lag"])
    res["seasonal_corr"], res["seasonal_mats"] = seasonal_correlations(df)
    res["iod_seasonal"] = iod_seasonal_behaviour(df["DMI"], events["iod_phase"], events["iod_events"])
    res["strong_years"] = strong_event_years(events["enso_events"], events["iod_events"])
    res["cooccurrence"] = enso_iod_cooccurrence(events["enso_phase"], events["iod_phase"])

    res["monthly_clim"].to_csv(d / "monthly_climatology_indices.csv")
    res["annual"].to_csv(d / "annual_statistics.csv")
    res["seasonal"].to_csv(d / "seasonal_statistics.csv", index=False)
    res["trends"].to_csv(d / "trend_analysis.csv", index=False)
    res["variability"].to_csv(d / "basic_variability_statistics.csv", index=False)
    res["extremes"].to_csv(d / "extreme_months.csv", index=False)
    res["timescales"].to_csv(d / "variance_by_timescale.csv", index=False)
    res["decadal"].to_csv(d / "decadal_means.csv")
    res["acf_pacf"].to_csv(d / "acf_pacf.csv")
    res["corr"].to_csv(d / "correlation_matrix_pearson.csv")
    res["corr_p"].to_csv(d / "correlation_matrix_pvalues_effN.csv")
    res["lead_lag"].to_csv(d / "lead_lag_correlations.csv", index=False)
    res["best_lags"].to_csv(d / "lead_lag_best_lags.csv", index=False)
    res["seasonal_corr"].to_csv(d / "seasonal_correlations.csv", index=False)
    for s, m in res["seasonal_mats"].items():
        m.to_csv(d / f"correlation_matrix_{s}.csv")
    res["iod_seasonal"].to_csv(d / "IOD_seasonal_behaviour.csv")
    res["strong_years"].to_csv(d / "strong_ENSO_IOD_years.csv", index=False)
    res["cooccurrence"].to_csv(d / "ENSO_IOD_phase_cooccurrence.csv")
    log.info("Statistical analysis saved to %s", d)
    return res
