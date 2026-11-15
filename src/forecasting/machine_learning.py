"""Leak-free feature engineering and machine-learning estimators (RF, XGBoost, SVR, GBM) with rolling-origin tuning.

Origin: forecasting.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src import settings


IDX, H = settings.INDICES, settings.HORIZONS


try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


ML_MODELS = ["RandomForest", "XGBoost", "SVR", "GradientBoosting"]


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=n)).replace(day=1)


def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Predictors for every origin month t.  Every term is a trailing function of data <= t."""
    X = {}
    for k in IDX:
        for lag in range(settings.N_LAGS):
            X[f"{k}_lag{lag}"] = df[k].shift(lag)
        for w in settings.ROLL_WINDOWS:
            X[f"{k}_rm{w}"] = df[k].rolling(w).mean()
    for w in settings.ROLL_WINDOWS:
        X[f"{target}_rs{w}"] = df[target].rolling(w).std()
    # ENSO-IOD interaction terms and tendencies
    X["R34xDMI_lag0"] = df["R34"] * df["DMI"]
    X["R34xDMI_rm3"] = X["R34_rm3"] * X["DMI_rm3"]
    X["R34xDMI_rm6"] = X["R34_rm6"] * X["DMI_rm6"]
    X["R34_tend3"] = df["R34"] - df["R34"].shift(3)
    X["DMI_tend3"] = df["DMI"] - df["DMI"].shift(3)
    m = df.index.month
    X["sin_month"], X["cos_month"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return pd.DataFrame(X, index=df.index)


def target_for_horizon(df: pd.DataFrame, target: str, h: int) -> pd.Series:
    return df[target].shift(-h)          # value at t+h, aligned to origin t


def ml_grid(name: str) -> list[dict]:
    if name == "RandomForest":
        return [dict(max_depth=4, min_samples_leaf=5, max_features=0.33),
                dict(max_depth=8, min_samples_leaf=5, max_features=0.33),
                dict(max_depth=None, min_samples_leaf=10, max_features=0.5)]
    if name == "GradientBoosting":
        return [dict(n_estimators=150, learning_rate=0.03, max_depth=2, subsample=0.8),
                dict(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8),
                dict(n_estimators=300, learning_rate=0.02, max_depth=2, subsample=0.8)]
    if name == "XGBoost":
        return [dict(n_estimators=200, learning_rate=0.03, max_depth=2, subsample=0.8, colsample_bytree=0.6,
                     reg_lambda=5.0),
                dict(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.6,
                     reg_lambda=5.0),
                dict(n_estimators=400, learning_rate=0.02, max_depth=2, subsample=0.8, colsample_bytree=0.5,
                     reg_lambda=10.0)]
    if name == "SVR":
        return [dict(C=0.3, epsilon=0.1, gamma="scale"), dict(C=1.0, epsilon=0.1, gamma="scale"),
                dict(C=3.0, epsilon=0.1, gamma="scale"), dict(C=1.0, epsilon=0.1, gamma=0.002)]
    raise ValueError(name)


def make_estimator(name: str, params: dict):
    seed = settings.RANDOM_SEED
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=settings.RF_TREES, random_state=seed, n_jobs=settings.N_JOBS, **params)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=seed, **params)
    if name == "XGBoost":
        return XGBRegressor(random_state=seed, n_jobs=settings.N_JOBS if settings.N_JOBS > 0 else 0,
                            tree_method="hist", verbosity=0, **params)
    if name == "SVR":
        return Pipeline([("scaler", StandardScaler()), ("svr", SVR(**params))])
    raise ValueError(name)


def tune_ml(name: str, X: pd.DataFrame, y: pd.Series, h: int, last_origin: pd.Timestamp) -> dict:
    """Rolling-origin (expanding window) CV using only rows with origin <= last_origin - h.

    An embargo of h months separates each training block from its test block so no training target
    overlaps the test period.  Returns the parameter set with the lowest mean CV RMSE.
    """
    cutoff = add_months(last_origin, -h)
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    d = d[d.index <= cutoff]
    n = len(d)
    grid = ml_grid(name)
    if n < 60 or len(grid) == 1:
        return grid[0]
    folds = settings.CV_FOLDS
    bs = max(12, n // (folds + 2))
    scores = np.zeros(len(grid))
    for gi, params in enumerate(grid):
        errs = []
        for f in range(folds):
            te_start = n - (folds - f) * bs
            tr_end = te_start - h                       # embargo
            if tr_end < 36:
                continue
            tr, te = d.iloc[:tr_end], d.iloc[te_start: te_start + bs]
            est = make_estimator(name, params).fit(tr.drop(columns="y"), tr["y"])
            p = est.predict(te.drop(columns="y"))
            errs.append(np.sqrt(np.mean((te["y"].values - p) ** 2)))
        scores[gi] = np.mean(errs) if errs else np.inf
    best = grid[int(np.argmin(scores))]
    return best


def fit_ml(name: str, params: dict, X: pd.DataFrame, y: pd.Series, h: int, origin: pd.Timestamp):
    """Fit on rows t <= origin - h (targets observed by `origin`)."""
    cutoff = add_months(origin, -h)
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    d = d[d.index <= cutoff]
    if len(d) < 36:
        raise ValueError(f"Too few training rows ({len(d)}) for {name} h={h}")
    est = make_estimator(name, params).fit(d.drop(columns="y"), d["y"])
    return est, list(X.columns), len(d)
