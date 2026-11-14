"""Step 7.1 calibration fits (src/calibrate.py)."""
import numpy as np
import pandas as pd
import pytest

from src import calibrate as cal
from src import eval as ev


def synthetic(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(-4, 2.5, n)
    return m, rng


def test_temperature_recovers_known_T():
    m, rng = synthetic()
    y = (rng.random(len(m)) < cal.sigmoid(m / 2.0)).astype(int)
    fit = cal.fit_temperature(m, y, (0.05, 20))
    assert fit["T"] == pytest.approx(2.0, rel=0.03) and not fit["at_bound"]


def test_temperature_respects_bounds():
    m, rng = synthetic(20_000)
    y = (rng.random(len(m)) < cal.sigmoid(m / 50.0)).astype(int)          # true T beyond 20
    fit = cal.fit_temperature(m, y, (0.05, 20))
    assert 0.05 <= fit["T"] <= 20 and fit["at_bound"]


def test_platt_recovers_known_slope_and_intercept():
    m, rng = synthetic()
    y = (rng.random(len(m)) < cal.sigmoid(0.6 * m - 1.5)).astype(int)
    fit = cal.fit_platt(m, y)
    assert fit["a"] == pytest.approx(0.6, rel=0.05) and fit["b"] == pytest.approx(-1.5, abs=0.15)


def test_platt_refuses_a_decreasing_map():
    m, rng = synthetic(50_000)
    y = (rng.random(len(m)) < cal.sigmoid(-0.8 * m - 4)).astype(int)     # inverted relation
    with pytest.raises(RuntimeError, match="would not preserve the ranking"):
        cal.fit_platt(m, y)


@pytest.mark.parametrize("method, params", [("temperature", {"T": 0.7}),
                                            ("platt", {"a": 0.5, "b": -3.0})])
def test_maps_are_increasing_and_commute_with_session_max(method, params):
    m = np.array([-12.0, -3.0, -2.5, 0.0, 4.0, 9.0])
    p = cal.apply(method, params, m)
    assert np.all(np.diff(p) > 0) and np.all((p > 0) & (p < 1))
    meta = pd.DataFrame({"row_id": list("abcdef"), "session_id": [1, 1, 2, 2, 3, 3], "user": "u",
                         "heldout_user": False, "score_time": pd.date_range("2010-11-01", periods=6, freq="h"),
                         "scenario": np.nan, "y": [0, 0, 0, 1, 0, 0]})
    _, per_row = ev.aggregate_sessions(meta, [p])                          # eval.py's max over rows
    session_max = np.array([m[:2].max(), m[2:4].max(), m[4:].max()])
    assert np.allclose(per_row[0], cal.apply(method, params, session_max))


def test_fit_data_session_unit_uses_max_margin_and_any_positive():
    preds = pd.DataFrame({"row_id": ["a", "b", "c"], "margin": [-5.0, -1.0, -3.0], "y_true": [0, 1, 0]})
    sessions = pd.DataFrame({"row_id": ["a", "b", "c"], "session_id": [1, 1, 2]})
    m, y = cal.fit_data(preds, "session", sessions)
    assert m.tolist() == [-1.0, -3.0] and y.tolist() == [1, 0]
    m, y = cal.fit_data(preds, "row", sessions)
    assert m.tolist() == [-5.0, -1.0, -3.0]


def test_nll_is_stable_for_extreme_margins():
    z = np.array([-800.0, 800.0])
    assert np.isfinite(cal.nll(z, np.array([0, 1]))) and cal.nll(z, np.array([0, 1])) == pytest.approx(0)
