"""Tests for the binomial GLM (rainfall occurrence), wet-day targets and event verification (python tests/test_glm.py)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, os.path.dirname(__file__))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_seasonal import SPEC, YEARS, make_indices  # noqa: E402

from src.forecasting import glm as G  # noqa: E402
from src.forecasting import seasonal as S  # noqa: E402
from src.preprocessing import station_processing as P  # noqa: E402
from src.verification import probabilistic_scores as V  # noqa: E402


def planted_fraction(signal=0.6, n_st=3, seed=1, trials=122):
    ind = make_indices(); X = S.build_predictors(ind, SPEC, YEARS + [2026])
    z = (X["R34"] / X["R34"].std()).loc[YEARS].values
    rng = np.random.default_rng(seed)
    Y = pd.DataFrame({f"S{j}": rng.binomial(trials, expit(logit(0.45) + signal * z + 0.1 * rng.normal(size=len(z)))) / trials
                      for j in range(n_st)}, index=YEARS)
    return X, Y, trials


def test_irls_recovers_coefficients_and_penalty_shrinks():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(6000, 2)); trials = 100
    y = rng.binomial(trials, expit(0.2 + 0.7 * Z[:, 0] - 0.4 * Z[:, 1])) / trials
    b0, b = G.fit_binomial_glm(Z, y, trials, alpha=1e-8)
    assert abs(b0 - 0.2) < 0.03 and abs(b[0] - 0.7) < 0.03 and abs(b[1] + 0.4) < 0.03
    _, bs = G.fit_binomial_glm(Z, y, trials, alpha=50.0)
    assert np.abs(bs).sum() < np.abs(b).sum()
    _, bsep = G.fit_binomial_glm(Z[:40], np.ones(40), trials, alpha=1.0)                   # always wet: must not diverge
    assert np.isfinite(bsep).all()
    b0e, be = G.fit_binomial_glm(np.empty((50, 0)), y[:50], trials)
    assert abs(expit(b0e) - y[:50].mean()) < 1e-6 and be.size == 0                         # intercept-only = the mean fraction


def test_glm_finds_occurrence_signal_not_noise_and_stays_in_unit_interval():
    X, Y, N = planted_fraction()
    h = S.hindcast(Y, X, SPEC, method="glm", trials=N)
    s = V.summarize(h)
    assert s["pooled"]["rpss"] > 0.1 and s["per_target"]["correlation"].mean() > 0.5
    assert ((h.forecast.values > 0) & (h.forecast.values < 1)).all() and np.allclose(h.probs.sum(-1), 1, atol=1e-9)
    assert np.mean([("R34" in x) for x in h.selected.values.ravel()]) > 0.6
    Xn, Yn, _ = planted_fraction(signal=0.0, seed=4)
    hn = S.hindcast(Yn, Xn, SPEC, method="glm", trials=N)
    assert V.summarize(hn)["pooled"]["rpss"] < 0.06 and (hn.selected.values == "climatology").mean() > 0.5


def test_glm_input_checks_and_no_future_leakage():
    X, Y, N = planted_fraction(n_st=2)
    for bad in (lambda: S.hindcast(Y, X, SPEC, method="glm"), lambda: S.hindcast(Y * 100, X, SPEC, method="glm", trials=N)):
        try:
            bad(); raise AssertionError
        except ValueError:
            pass
    t = 2010
    Y2 = Y.copy(); Y2.loc[t:] = 0.99
    a, b = S.hindcast(Y, X, SPEC, method="glm", trials=N), S.hindcast(Y2, X, SPEC, method="glm", trials=N)
    assert np.allclose(a.forecast.loc[t], b.forecast.loc[t]) and np.allclose(a.p_above.loc[t], b.p_above.loc[t])
    f = S.forecast(Y, X, SPEC, 2026, method="glm", trials=N)
    assert ((f["Forecast"] > 0) & (f["Forecast"] < 1)).all() and (f["Method"] == "glm").all()


def test_season_days_and_wet_day_fraction_table():
    assert P.season_days("JJAS", 2020) == 122 and P.season_days("ONDJ", 2020, "start") == 123
    assert P.season_days("DJF", 2021, "end") == 90 and P.season_days("DJF", 2024, "end") == 91
    d = pd.date_range("2000-01-01", "2002-12-31")
    rain = np.zeros(len(d)); rain[(d.month == 6) & (d.day <= 10)] = 1.0               # exactly 1.0 mm counts as wet
    rain[(d.month == 6) & (d.day == 11)] = 0.99                                       # just below the threshold
    rain[(d.month == 7) & (d.day % 2 == 0)] = 5.0
    obs = pd.DataFrame({"station": "A", "date": d, "lat": 1.0, "lon": 2.0, "variable": "precipitation", "value": rain})
    obs.loc[(obs.date == "2001-08-15"), "value"] = np.nan                              # a missing day -> season 2001 dropped
    tab, n = P.seasonal_wet_day_fraction(obs, "JJAS", 1.0)
    assert list(tab.season_year) == [2000, 2002] and abs(n - 122) < 1e-9 and (tab.variable == "wet_day_fraction").all()
    assert abs(tab.value.iloc[0] - (10 + 15) / 122) < 1e-12                           # 10 June days + 15 even July days (2000)


def test_contingency_scores_by_hand_and_event_scores():
    sc = V.contingency_scores([1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0])
    assert (sc["hits"], sc["misses"], sc["false_alarms"], sc["correct_negatives"]) == (2, 1, 1, 2)
    assert abs(sc["accuracy"] - 4 / 6) < 1e-12 and abs(sc["pod"] - 2 / 3) < 1e-12 and abs(sc["far"] - 1 / 3) < 1e-12
    assert abs(sc["csi"] - 0.5) < 1e-12 and abs(sc["bias_score"] - 1.0) < 1e-12 and abs(sc["pofd"] - 1 / 3) < 1e-12
    assert np.isnan(V.contingency_scores([0, 0], [0, 0])["pod"])                         # no events observed
    X, Y, N = planted_fraction()
    h = S.hindcast(Y, X, SPEC, method="glm", trials=N)
    ev = V.event_scores(h, 0.5)
    allr = ev[ev.target == "ALL"].set_index("event")
    assert set(allr.index) == {"below_normal", "above_normal"} and (ev.target != "ALL").sum() == 2 * Y.shape[1]
    assert allr.loc["above_normal", "pod"] >= 0 and allr.loc["above_normal", "accuracy"] > 0.5
    assert allr.loc["above_normal", "brier_skill_score"] > 0                              # the planted signal beats 1/3 climatology
    assert (allr[["hits", "misses", "false_alarms", "correct_negatives"]].sum(axis=1) == allr["n"]).all()


def test_cli_wet_day_workflow():
    d = Path(tempfile.mkdtemp()); ind = make_indices(); ind.reset_index(names="Date").to_csv(d / "idx.csv", index=False)
    X = S.build_predictors(ind, SPEC, YEARS)
    z = (X["R34"] / X["R34"].std()).loc[YEARS]
    rng = np.random.default_rng(8)
    days = pd.date_range("1981-01-01", "2025-12-31"); rows = []
    for k, stn in enumerate(["W1", "W2"]):
        pday = np.where(np.isin(days.month, [6, 7, 8, 9]), expit(logit(0.45) + 0.6 * z.reindex(days.year).values), 0.1)
        rain = np.where(rng.random(len(days)) < pday, rng.gamma(2, 6, len(days)) + 1.0, 0.0)
        rows.append(pd.DataFrame({"Station": stn, "Date": days.strftime("%Y-%m-%d"), "Lat": 10 + k, "Lon": 35 + k, "Variable": "precipitation", "Value": rain.round(2)}))
    pd.concat(rows).to_csv(d / "obs.csv", index=False)
    out = d / "out"
    r = subprocess.run([sys.executable, "main.py", "--stations", str(d / "obs.csv"), "--station-season", "JJAS", "--station-forecast",
                        "--init-month", "5", "--target-year", "2026", "--sst-indices", str(d / "idx.csv"), "--predictors", "R34,WIOD",
                        "--methods", "ridge", "--wet-day-threshold", "1.0", "--station-output", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=290)
    assert r.returncode == 0, r.stderr[-1200:]
    f = pd.read_csv(out / "station_forecast_JJAS_2026_init05.csv")
    w = f[f["Variable"] == "wet_day_fraction"]
    assert len(w) == 2 and ((w["Forecast"] > 0) & (w["Forecast"] < 1)).all() and "Weight_glm" in w.columns
    assert np.allclose(w["Forecast_wet_days"], w["Forecast"] * 122, atol=1e-2) and (w["wet_day_threshold_mm"] == 1.0).all()
    ev = pd.read_csv(out / "hindcast_event_scores_wet_day_fraction_JJAS_lead1.csv")
    assert {"accuracy", "pod", "far", "csi", "brier_skill_score"} <= set(ev.columns) and (ev.target == "ALL").sum() == 2
    cmp_ = pd.read_csv(out / "method_comparison_wet_day_fraction_JJAS_lead1.csv")
    assert {"glm", "ridge", "climatology", "ensemble"} <= set(cmp_["method"])


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
