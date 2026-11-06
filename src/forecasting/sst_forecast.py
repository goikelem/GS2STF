"""SST-index forecasting: statistical models (persistence, climatology, AR, ARIMA, SARIMA, VAR), model registry, final forecasts.

Origin: forecasting.py."""
from __future__ import annotations

from datetime import datetime
import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

from src import settings
from src.forecasting.machine_learning import HAS_XGB, ML_MODELS, add_months, build_features, fit_ml, ml_grid, target_for_horizon


log = logging.getLogger("sst.forecast")


IDX, H = settings.INDICES, settings.HORIZONS


try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SM = True
except Exception:  # pragma: no cover
    HAS_SM = False


STAT_MODELS = ["persistence", "climatology", "AR", "ARIMA", "SARIMA", "VAR"]


NEEDS_SM = {"ARIMA", "SARIMA", "VAR"}


COMPLEXITY = {"persistence": 0, "climatology": 0, "AR": 1, "ARIMA": 2, "SARIMA": 3, "VAR": 3,
              "SVR": 4, "RandomForest": 4, "GradientBoosting": 4, "XGBoost": 4}


def available_models() -> list[str]:
    out = []
    for m in settings.MODELS:
        if m == "XGBoost" and not HAS_XGB:
            log.warning("xgboost not installed -> XGBoost skipped")
        elif m in NEEDS_SM and not HAS_SM:
            log.warning("statsmodels not installed -> %s skipped", m)
        elif m in ML_MODELS + STAT_MODELS:
            out.append(m)
        else:
            log.warning("Unknown model '%s' ignored", m)
    return out


class StatForecaster:
    """Base: fit(hist) then forecast(hist_to_origin, steps) -> DataFrame (index 1..steps, columns IDX)."""
    name = "base"

    def fit(self, hist: pd.DataFrame):
        raise NotImplementedError

    def forecast(self, hist: pd.DataFrame, steps: int = 12) -> pd.DataFrame:
        raise NotImplementedError


class Persistence(StatForecaster):
    name = "persistence"

    def fit(self, hist):
        return self

    def forecast(self, hist, steps=12):
        last = hist[IDX].iloc[-1]
        return pd.DataFrame([last.values] * steps, columns=IDX, index=range(1, steps + 1))


class Climatology(StatForecaster):
    """Mean anomaly of each calendar month over the fitting window (the 'seasonal climatology' benchmark)."""
    name = "climatology"

    def fit(self, hist):
        self.clim = hist[IDX].groupby(hist.index.month).mean()
        return self

    def forecast(self, hist, steps=12):
        origin = hist.index[-1]
        rows = [self.clim.loc[add_months(origin, h).month].values for h in range(1, steps + 1)]
        return pd.DataFrame(rows, columns=IDX, index=range(1, steps + 1))


def _fit_ar(x: np.ndarray, maxlag: int = 12):
    """OLS AR(p) with intercept, p chosen by BIC on a common sample."""
    x = np.asarray(x, float)
    maxlag = min(maxlag, len(x) // 5)
    best = None
    Y = x[maxlag:]
    for p in range(1, maxlag + 1):
        A = np.column_stack([np.ones(len(Y))] + [x[maxlag - j: len(x) - j] for j in range(1, p + 1)])
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        rss = float(np.sum((Y - A @ beta) ** 2))
        bic = len(Y) * np.log(rss / len(Y)) + (p + 1) * np.log(len(Y))
        if best is None or bic < best[0]:
            best = (bic, p)
    p = best[1]
    Y = x[p:]
    A = np.column_stack([np.ones(len(Y))] + [x[p - j: len(x) - j] for j in range(1, p + 1)])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return beta[0], beta[1:], p


class AR(StatForecaster):
    name = "AR"

    def fit(self, hist):
        self.par = {k: _fit_ar(hist[k].dropna().values) for k in IDX}
        return self

    def forecast(self, hist, steps=12):
        out = {}
        for k in IDX:
            c, phi, p = self.par[k]
            buf = list(hist[k].dropna().values[-p:])
            fc = []
            for _ in range(steps):
                v = c + sum(phi[j] * buf[-1 - j] for j in range(p))
                fc.append(v)
                buf.append(v)
            out[k] = fc
        return pd.DataFrame(out, index=range(1, steps + 1))[IDX]


class _SMBase(StatForecaster):
    """Shared roll-forward logic for statsmodels state-space models (append without re-estimation)."""

    def _series(self, hist, k):
        s = hist[k].dropna()
        s.index = pd.DatetimeIndex(s.index, freq="MS")
        return s

    def fit(self, hist):
        self.res, self.end, self.info = {}, hist.index[-1], {}
        self.cur_end = {k: self.end for k in IDX}
        for k in IDX:
            s = self._series(hist, k)
            try:
                self.res[k], self.info[k] = self._fit_one(k, s)
            except Exception as e:     # noqa: BLE001
                log.warning("%s fit failed for %s (%s); falling back to AR", self.name, k, e)
                self.res[k], self.info[k] = None, {"fallback": "AR"}
        self._ar = AR().fit(hist)
        return self

    def forecast(self, hist, steps=12):
        out = {}
        ar_fc = None
        for k in IDX:
            res = self.res.get(k)
            if res is not None:
                try:
                    s = self._series(hist, k)
                    new = s[s.index > self.cur_end[k]]
                    if len(new):
                        res = res.append(new, refit=False)
                        self.res[k] = res
                        self.cur_end[k] = s.index[-1]
                    out[k] = np.asarray(res.forecast(steps))
                    continue
                except Exception as e:  # noqa: BLE001
                    log.warning("%s forecast failed for %s (%s); AR fallback", self.name, k, e)
            if ar_fc is None:
                ar_fc = self._ar.forecast(hist, steps)
            out[k] = ar_fc[k].values
        return pd.DataFrame(out, index=range(1, steps + 1))[IDX]


class ARIMAModel(_SMBase):
    name = "ARIMA"
    ORDERS = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (3, 0, 0)]

    def __init__(self, orders: dict | None = None):
        self.orders = orders          # chosen once on the initial training window, then reused

    def fit(self, hist):
        if self.orders is None:
            self.orders = {}
        return super().fit(hist)

    def _fit_one(self, k, s):
        if k not in self.orders:
            best = None
            for o in self.ORDERS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = ARIMA(s, order=o, trend="c").fit()
                    if best is None or r.aic < best[0]:
                        best = (r.aic, o)
                except Exception:  # noqa: BLE001
                    continue
            if best is None:
                raise RuntimeError("no ARIMA order converged")
            self.orders[k] = best[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ARIMA(s, order=self.orders[k], trend="c").fit()
        return r, {"order": self.orders[k], "aic": float(r.aic)}


class SARIMAModel(_SMBase):
    name = "SARIMA"
    ORDER, SEAS = (1, 0, 1), (1, 0, 0, 12)

    def _fit_one(self, k, s):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = SARIMAX(s, order=self.ORDER, seasonal_order=self.SEAS, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        return r, {"order": self.ORDER, "seasonal_order": self.SEAS, "aic": float(r.aic)}


class VARModel(StatForecaster):
    """VAR on the three independent series R34, WIOD, EIOD; DMI is derived as WIOD - EIOD
    (DMI is a linear combination, so including it would make the system singular)."""
    name = "VAR"
    COLS = ["R34", "WIOD", "EIOD"]

    def fit(self, hist):
        d = hist[self.COLS].dropna()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = VAR(d).fit(maxlags=6, ic="aic", trend="c")
                if res.k_ar < 1:
                    res = VAR(d).fit(1, trend="c")
            self.res, self.k_ar = res, res.k_ar
        except Exception as e:     # noqa: BLE001
            log.warning("VAR failed (%s); AR fallback", e)
            self.res = None
        self._ar = AR().fit(hist)
        return self

    def forecast(self, hist, steps=12):
        if self.res is None:
            return self._ar.forecast(hist, steps)
        d = hist[self.COLS].dropna()
        fc = self.res.forecast(d.values[-self.k_ar:], steps)
        out = pd.DataFrame(fc, columns=self.COLS, index=range(1, steps + 1))
        out["DMI"] = out["WIOD"] - out["EIOD"]
        return out[IDX]


def make_stat_model(name: str, state: dict | None = None) -> StatForecaster:
    state = state if state is not None else {}
    if name == "persistence":
        return Persistence()
    if name == "climatology":
        return Climatology()
    if name == "AR":
        return AR()
    if name == "ARIMA":
        return ARIMAModel(state.setdefault("ARIMA_orders", None))
    if name == "SARIMA":
        return SARIMAModel()
    if name == "VAR":
        return VARModel()
    raise ValueError(name)


def interval_offsets(hc: pd.DataFrame, model: str, index: str, h: int) -> tuple[float, float, int]:
    """Empirical residual quantiles (obs - pred) of the selected model's hindcast -> additive offsets."""
    e = hc[(hc["model"] == model) & (hc["index"] == index) & (hc["lead"] == h)]
    err = (e["obs"] - e["pred"]).dropna().values
    a = (1 - settings.PI_LEVEL) / 2
    if len(err) < 10:
        sd = np.nanstd(hc[(hc["index"] == index)]["obs"]) if len(hc) else 0.5
        z = 1.645
        return -z * sd, z * sd, len(err)
    return float(np.quantile(err, a)), float(np.quantile(err, 1 - a)), len(err)


def final_forecast(df: pd.DataFrame, selection: pd.DataFrame, tuned: dict, hindcast: pd.DataFrame,
                   skill_flags: dict) -> pd.DataFrame:
    """Retrain every selected model on ALL available data and forecast 12 months from the last complete month."""
    df = df[IDX].dropna()
    origin = df.index[-1]
    log.info("Final forecast: initialization (last complete month) = %s", origin.strftime("%Y-%m"))
    feats = {k: build_features(df, k) for k in IDX}
    stat_cache, rows = {}, []
    state = {"ARIMA_orders": tuned.get("ARIMA_orders")}   # reuse orders chosen on the initial training window
    models_out = {k: {} for k in IDX}

    for _, sel in selection.iterrows():
        k, h, name = sel["Index"], int(sel["Lead"]), sel["Selected_Model"]
        if name in ML_MODELS:
            params = tuned.get((name, k, h)) or ml_grid(name)[0]
            y = target_for_horizon(df, k, h)
            est, cols, nrow = fit_ml(name, params, feats[k], y, h, origin)
            x_now = feats[k].loc[[origin], cols]
            if x_now.isna().values.any():
                raise ValueError(f"NaN in predictors at origin for {k}/{h}")
            fc = float(est.predict(x_now)[0])
            models_out[k][h] = {"model_name": name, "estimator": est, "features": cols, "params": params,
                                "n_train_rows": nrow, "trained_through": origin.strftime("%Y-%m")}
        else:
            if name not in stat_cache:
                stat_cache[name] = make_stat_model(name, state).fit(df)
                stat_cache[name + "_fc"] = stat_cache[name].forecast(df, 12)
            fc = float(stat_cache[name + "_fc"].loc[h, k])
            models_out[k][h] = {"model_name": name, "estimator": stat_cache[name], "features": None,
                                "params": getattr(stat_cache[name], "orders", None) or {},
                                "n_train_rows": len(df), "trained_through": origin.strftime("%Y-%m")}
        lo, hi, ne = interval_offsets(hindcast, name, k, h)
        rows.append({"Date": add_months(origin, h).strftime("%Y-%m-%d"), "Lead_Month": h, "Index": k,
                     "Forecast_Value": round(fc, 4), "Lower_Bound": round(fc + lo, 4),
                     "Upper_Bound": round(fc + hi, 4), "Selected_Model": name,
                     "Initialization_Date": origin.strftime("%Y-%m-%d"),
                     "Interval_Level": settings.PI_LEVEL, "N_residuals_for_interval": ne,
                     "Validated_Skill": skill_flags.get((k, h), "unknown")})
    fc_df = pd.DataFrame(rows).sort_values(["Index", "Lead_Month"]).reset_index(drop=True)
    save_models(models_out, df, origin)
    return fc_df


def save_models(models_out: dict, df: pd.DataFrame, origin: pd.Timestamp):
    d = settings.out("models")
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "initialization": origin.strftime("%Y-%m"),
            "random_seed": settings.RANDOM_SEED, "n_lags": settings.N_LAGS, "roll_windows": settings.ROLL_WINDOWS,
            "train_period": [df.index[0].strftime("%Y-%m"), df.index[-1].strftime("%Y-%m")],
            "pred_interval_level": settings.PI_LEVEL, "models": {}}
    for k, per_h in models_out.items():
        # statistical forecasters hold statsmodels results; strip heavy data before pickling when possible
        bundle = {}
        for h, m in per_h.items():
            m = dict(m)
            if hasattr(m["estimator"], "res") and isinstance(getattr(m["estimator"], "res"), dict):
                for r in m["estimator"].res.values():
                    try:
                        if r is not None:
                            r.remove_data()
                    except Exception:  # noqa: BLE001
                        pass
            bundle[h] = m
        try:
            joblib.dump(bundle, d / f"{k}_selected_models.joblib", compress=3)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not serialise some models for %s (%s); saving ML models only", k, e)
            joblib.dump({h: m for h, m in bundle.items() if m["features"] is not None},
                        d / f"{k}_selected_models.joblib", compress=3)
        meta["models"][k] = {str(h): {"model": m["model_name"], "params": _jsonable(m["params"]),
                                      "n_train_rows": m["n_train_rows"], "trained_through": m["trained_through"]}
                             for h, m in per_h.items()}
    joblib.dump({"indices": settings.INDICES, "regions": settings.REGIONS, "baseline": [settings.BASELINE_START,
                 settings.BASELINE_END], "feature_builder": "forecasting.build_features"},
                d / "preprocessing_config.joblib")
    (d / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("Saved trained models and metadata to %s", d)


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return json.loads(json.dumps(o, default=str))
