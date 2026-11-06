"""Tests for the method registry, ensemble and method comparison (python tests/test_ensemble.py  or  pytest tests)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_seasonal import SPEC, YEARS, planted  # noqa: E402

from src.forecasting import ensemble as E  # noqa: E402
from src.forecasting import seasonal as S  # noqa: E402
from src.verification import probabilistic_scores as V  # noqa: E402

KW = dict(rf_trees=60)


def _noise(Y, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(400, 60, size=Y.shape), index=Y.index, columns=Y.columns)


def test_every_method_finds_signal_and_not_noise():
    ind, X, Y = planted(n_targets=4)
    Yn = _noise(Y)
    for m in ("ridge", "pcr", "cca", "rf"):
        h = S.hindcast(Y, X, SPEC, method=m, **KW)
        s = V.summarize(h)
        assert s["pooled"]["rpss"] > 0.08 and s["per_target"]["correlation"].mean() > 0.35, (m, s["pooled"]["rpss"])
        assert np.allclose(h.probs.sum(-1), 1, atol=1e-9)
        hn = S.hindcast(Yn, X, SPEC, method=m, **KW)
        assert V.summarize(hn)["pooled"]["rpss"] < 0.08, (m, V.summarize(hn)["pooled"]["rpss"])   # falls back to climatology
    try:
        S.hindcast(Y, X, SPEC, method="nope"); raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_methods_never_use_future_years():
    ind, X, Y = planted(n_targets=3)
    t = 2010
    Y2 = Y.copy(); Y2.loc[t:] = np.where(np.arange(len(Y2.loc[t:]))[:, None] % 2, 1e5, -1e5)
    for m in ("pcr", "cca", "rf"):
        a, b = S.hindcast(Y, X, SPEC, method=m, **KW), S.hindcast(Y2, X, SPEC, method=m, **KW)
        assert np.allclose(a.forecast.loc[t], b.forecast.loc[t]) and np.allclose(a.p_above.loc[t], b.p_above.loc[t]), m


def _members(Y, X, methods=("climatology", "ridge", "pcr", "cca"), scheme="rolling"):
    return {m: S.hindcast(Y, X, SPEC, method=m, scheme=scheme, **KW) for m in methods}


def test_ensemble_weights_use_only_past_and_favour_skill():
    ind, X, Y = planted(n_targets=4)
    mem = _members(Y, X, ("climatology", "ridge"))
    ens, w = E.combine_hindcasts(mem, "skill")
    assert np.allclose(w.sum(1), 1) and w["ridge"].iloc[-1] > w["climatology"].iloc[-1]
    assert V.summarize(ens)["pooled"]["rpss"] > 0.1 and np.allclose(ens.probs.sum(-1), 1, atol=1e-9)
    t = 2012                                                           # corrupt the test year and everything after it
    Y2 = Y.copy(); Y2.loc[t:] = 1e5
    ens2, _ = E.combine_hindcasts(_members(Y2, X, ("climatology", "ridge")), "skill")
    assert np.allclose(ens.forecast.loc[t], ens2.forecast.loc[t]) and np.allclose(ens.p_below.loc[t], ens2.p_below.loc[t])
    sel, _ = E.combine_hindcasts(mem, "select")
    assert (sel.selected.iloc[8:].values == "ridge").mean() > 0.7
    eq, _ = E.combine_hindcasts(mem, "equal")
    assert np.allclose(eq.p_above.values, 0.5 * (mem["ridge"].p_above.values + mem["climatology"].p_above.values))


def test_ensemble_shrinks_to_climatology_on_noise():
    ind, X, Y = planted(n_targets=4)
    Yn = _noise(Y)
    mem = _members(Yn, X)
    ens, w = E.combine_hindcasts(mem, "skill")
    assert V.summarize(ens)["pooled"]["rpss"] > -0.05
    cmp_ = V.compare({**mem, "ensemble": ens})
    assert cmp_["rpss"].max() < 0.1 and abs(cmp_.set_index("method").loc["climatology", "rpss"]) < 1e-9


def test_blocked_scheme_rules():
    ind, X, Y = planted(n_targets=2)
    mem = _members(Y, X, ("climatology", "ridge"), scheme="blocked")
    try:
        E.combine_hindcasts(mem, "skill"); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    ens, _ = E.combine_hindcasts(mem, "equal")
    assert V.summarize(ens)["pooled"]["rpss"] > 0.05


def test_operational_combination_and_comparison_table():
    ind, X, Y = planted(n_targets=3)
    meta = pd.DataFrame({"station": list(Y.columns), "lat": [10, 11, 12], "lon": [35, 36, 37]})
    fc, hc, allh = E.run_multimodel(Y, X, SPEC, 2026, ("ridge", "pcr", "cca"), meta=meta, variable="precipitation", **KW)
    assert {"ridge", "pcr", "cca", "climatology", "ensemble"} <= set(allh)
    assert np.allclose(fc[["P_below", "P_normal", "P_above"]].sum(1), 1)
    wc = [c for c in fc.columns if c.startswith("Weight_")]
    assert len(wc) == 4 and np.allclose(fc[wc].sum(1), 1) and (fc["Ensemble_spread"] >= 0).all()
    assert (fc["Method"].str.startswith("ensemble")).all() and (fc["Forecast_Period"] == "JJAS_2026").all()
    Yfut = Y.copy(); Yfut.loc[2026] = 1e9                              # target-year observation never used
    fc2, _, _ = E.run_multimodel(Yfut, X, SPEC, 2026, ("ridge", "pcr", "cca"), meta=meta, variable="precipitation", **KW)
    assert np.allclose(fc["Forecast"], fc2["Forecast"])
    cmp_ = V.compare(allh, n_boot=100)
    r = cmp_.set_index("method")
    assert cmp_["rpss"].iloc[0] == cmp_["rpss"].max() and r.loc["ridge", "diff_lo_90"] > 0
    assert {"rpss_lo_90", "rpss_hi_90", "hss", "mean_correlation"} <= set(cmp_.columns)


if __name__ == "__main__":
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("PASS", n)
            except Exception as e:
                import traceback; traceback.print_exc(); fails += 1; print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
