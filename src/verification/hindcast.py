"""Rolling-origin hindcast execution and validation outputs.

Origin: model_validation.py."""
from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from src import settings
from src.forecasting import machine_learning, sst_forecast
from src.verification.cross_validation import _cache_id, _setup, leakage_check, split_dates, tune_all


log = logging.getLogger("sst.validation")


IDX, H = settings.INDICES, settings.HORIZONS


def _window(S, i):
    r = S["refits"][i]
    nxt = S["refits"][i + 1] if i + 1 < len(S["refits"]) else machine_learning.add_months(S["last"], 1)
    return S["origins"][(S["origins"] >= r) & (S["origins"] < nxt)]


def _keep(S, origin, h):
    v = machine_learning.add_months(origin, h)
    return (S["vs"] <= v <= S["ve"]) and v <= S["last"], v


def hindcast_ml(df, name, tuned, S) -> list[dict]:
    rows = []
    for k in IDX:
        X = machine_learning.build_features(df, k)
        for h in H:
            y = machine_learning.target_for_horizon(df, k, h)
            params = tuned[(name, k, h)]
            for i, r in enumerate(S["refits"]):
                win = [t for t in _window(S, i) if _keep(S, t, h)[0]]
                if not win:
                    continue
                try:
                    est, cols, _ = machine_learning.fit_ml(name, params, X, y, h, r)
                except ValueError as e:
                    log.warning("%s %s h=%d refit %s skipped: %s", name, k, h, r.strftime("%Y-%m"), e)
                    continue
                Xw = X.loc[win, cols]
                ok = ~Xw.isna().any(axis=1).values
                if not ok.any():
                    continue
                pr = est.predict(Xw[ok])
                for t, p in zip([w for w, o in zip(win, ok) if o], pr):
                    v = machine_learning.add_months(t, h)
                    rows.append(dict(model=name, index=k, lead=h, origin=t, valid=v, obs=df.at[v, k], pred=float(p)))
    return rows


def hindcast_stat(df, name, S, state) -> list[dict]:
    rows = []
    for i, r in enumerate(S["refits"]):
        model = sst_forecast.make_stat_model(name, state).fit(df.loc[:r])
        if name == "ARIMA":
            state["ARIMA_orders"] = model.orders
        for t in _window(S, i):
            f = model.forecast(df.loc[:t], 12)
            for k in IDX:
                for h in H:
                    ok, v = _keep(S, t, h)
                    if ok:
                        rows.append(dict(model=name, index=k, lead=h, origin=t, valid=v, obs=df.at[v, k],
                                         pred=float(f.at[h, k])))
        log.info("  %s refit %s done", name, r.strftime("%Y-%m"))
    return rows


def run_hindcast(df: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    df = df[IDX].dropna()
    S = _setup(df)
    log.info("Hindcast: validation %s -> %s | refits every %d months (%d refits) | models: %s",
             S["vs"].strftime("%Y-%m"), S["ve"].strftime("%Y-%m"), settings.REFIT_EVERY, len(S["refits"]), names)
    leakage_check(df)
    tuned = tune_all(df, names, S)
    cdir = settings.out("cache")
    state, parts = {}, []
    for name in names:
        f = cdir / f"hindcast_{name}.csv"
        kf = cdir / f"hindcast_{name}.key"
        cid = _cache_id(df, name)
        if f.exists() and kf.exists() and kf.read_text() == cid and not settings.FORCE_RECOMPUTE:
            log.info("Hindcast %-16s: loaded from cache", name)
            part = pd.read_csv(f, parse_dates=["origin", "valid"])
            if name == "ARIMA":
                st = cdir / "arima_orders.joblib"
                if st.exists():
                    state["ARIMA_orders"] = joblib.load(st)
        else:
            log.info("Hindcast %-16s: running ...", name)
            rows = hindcast_ml(df, name, tuned, S) if name in machine_learning.ML_MODELS else hindcast_stat(df, name, S, state)
            part = pd.DataFrame(rows)
            part.to_csv(f, index=False)
            kf.write_text(cid)
            if name == "ARIMA" and state.get("ARIMA_orders"):
                joblib.dump(state["ARIMA_orders"], cdir / "arima_orders.joblib")
        parts.append(part)
    hc = pd.concat(parts, ignore_index=True)
    tuned["ARIMA_orders"] = state.get("ARIMA_orders")
    return hc, tuned, S


def save_validation(hc, perf, sel, S):
    d = settings.out("validation")
    hc.sort_values(["model", "index", "lead", "valid"]).to_csv(d / "hindcast_all_models.csv", index=False)
    perf.round(4).to_csv(d / "model_performance_metrics.csv", index=False)
    sel.round(4).to_csv(d / "model_selection.csv", index=False)
    m = hc.merge(sel[["Index", "Lead", "Selected_Model"]], left_on=["index", "lead"], right_on=["Index", "Lead"])
    m = m[m["model"] == m["Selected_Model"]].drop(columns=["Index", "Lead", "Selected_Model"])
    m.sort_values(["index", "lead", "valid"]).to_csv(d / "hindcast_selected_models.csv", index=False)
    cut, end = split_dates(hc)
    (d / "validation_protocol.json").write_text(json.dumps({
        "validation_valid_dates": [S["vs"].strftime("%Y-%m"), S["ve"].strftime("%Y-%m")],
        "selection_period_ends": cut.strftime("%Y-%m"), "independent_test_period": [
            (cut + pd.DateOffset(months=1)).strftime("%Y-%m"), end.strftime("%Y-%m")],
        "refit_every_months": settings.REFIT_EVERY, "cv_folds": settings.CV_FOLDS,
        "parsimony_tolerance": settings.PARSIMONY_TOL, "mbe_definition": "mean(forecast - observation)",
        "mape_note": "computed only where |observation| >= 0.5 degC (anomalies near 0 make MAPE unstable)"},
        indent=2))
    log.info("Validation outputs saved to %s", d)
