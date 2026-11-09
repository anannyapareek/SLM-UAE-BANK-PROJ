"""Evaluation metrics: weighting, agreement with sklearn, edge cases, CLI guard."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_curve

from src import eval as ev

CFG = {"fpr_levels": [0.01, 0.05], "alerts_per_day": 3, "ece_bins": 15, "scenario_fpr": 0.05,
       "tie_break_seed": 0, "bootstrap": {"resamples": 200, "seed": 0, "ci": 0.95}}


def scored(score, y, day=None, key=None):
    n = len(score)
    day = np.zeros(n, int) if day is None else np.asarray(day)
    key = np.arange(n) if key is None else np.asarray(key)
    return ev.Scored(score, y, day, key)


def all_metrics(s, w):
    out = ev.core_metrics(s, w, CFG)
    out["recall_threshold"] = ev.recall_at_fpr(s, w, 0.05)[1]
    out["brier"] = ev.brier(s, w)
    out["ece_mass"] = ev.ece(s, w, 15, True)
    out["ece_width"] = ev.ece(s, w, 15, False)
    return out


@pytest.fixture
def small():
    rng = np.random.default_rng(3)
    n = 60
    score = np.round(rng.random(n), 1)          # rounding forces many ties
    y = (rng.random(n) < 0.3).astype(int)
    day = rng.integers(0, 4, n)
    return score, y, day, np.arange(n)


def test_unit_weights_match_plain_metrics(small):
    score, y, day, key = small
    s = scored(score, y, day, key)
    assert ev.average_precision(s, np.ones(len(y))) == pytest.approx(average_precision_score(y, score))


def test_weights_equal_explicit_duplication(small):
    score, y, day, key = small
    w = np.random.default_rng(5).integers(0, 4, len(y)).astype(float)
    weighted = all_metrics(scored(score, y, day, key), w)
    copies = np.repeat(np.arange(len(y)), w.astype(int))
    duplicated = all_metrics(scored(score[copies], y[copies], day[copies], key[copies]),
                             np.ones(len(copies)))
    for name, value in weighted.items():
        assert value == pytest.approx(duplicated[name], abs=1e-12), name


def test_pr_auc_and_roc_match_sklearn_with_ties(small):
    score, y, _, _ = small
    s = scored(score, y)
    w = np.ones(len(y))
    assert ev.average_precision(s, w) == pytest.approx(average_precision_score(y, score))
    fpr, tpr, thresholds = roc_curve(y, score)
    for level in (0.01, 0.05, 0.3):
        ok = fpr <= level
        recall, threshold = ev.recall_at_fpr(s, w, level)
        assert recall == pytest.approx(tpr[ok].max())
        if recall > 0:
            assert threshold == pytest.approx(thresholds[ok][np.argmax(tpr[ok])])


def test_perfect_and_random_scorers():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.1).astype(int)
    perfect = scored(y + rng.random(len(y)) * 0.5, y)
    w = np.ones(len(y))
    assert ev.average_precision(perfect, w) == pytest.approx(1.0)
    assert ev.recall_at_fpr(perfect, w, 0.01)[0] == pytest.approx(1.0)
    noise = scored(rng.random(len(y)), y)
    assert ev.average_precision(noise, w) == pytest.approx(y.mean(), abs=0.02)


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(1)
    score = rng.random(200000)
    y = (rng.random(len(score)) < score).astype(int)
    s = scored(score, y)
    w = np.ones(len(y))
    assert ev.ece(s, w, 15, True) < 0.01
    assert ev.ece(s, w, 15, False) < 0.01


def test_top_k_per_day_counts():
    s = scored(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.array([1, 0, 1, 0, 1]), day=[0, 0, 0, 1, 1])
    precision, recall = ev.top_k_per_day(s, np.ones(5), 2)
    # day 0 alerts on 0.9 and 0.8; day 1 alerts on both units
    assert precision == pytest.approx(2 / 4)
    assert recall == pytest.approx(2 / 3)


def unit_set(seed):
    rng = np.random.default_rng(seed)
    n = 400
    users = rng.integers(0, 40, n)
    y = (rng.random(n) < 0.1).astype(int)
    meta = pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], "session_id": np.arange(n),
                         "user": [f"u{x}" for x in users], "heldout_user": False,
                         "score_time": pd.Timestamp("2010-09-01") + pd.to_timedelta(
                             rng.integers(0, 10, n), unit="D"),
                         "scenario": np.where(y == 1, 2, np.nan), "y": y})
    scores = [np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1) for _ in range(2)]
    u = ev.build_units(meta, scores, "row", {**CFG, "tie_break_seed": 0})
    u["strata"], u["unit_key"] = ev.user_strata(u), "row"
    return u


def test_paired_model_against_itself_is_zero():
    u = unit_set(2)
    resamples = ev.bootstrap({"a": u, "b": u}, CFG, n_jobs=1)
    for metric in ev.CI_METRICS:
        diffs = [r["a"][metric] - r["b"][metric] for r in resamples]
        ci, skipped = ev.interval(diffs, 0.95)
        assert ci == [0.0, 0.0] and skipped == 0
        assert not (ci[0] > 0 or ci[1] < 0)


def test_session_aggregation():
    meta = pd.DataFrame({
        "row_id": ["a", "b", "c", "d"], "session_id": [1, 1, 2, 2], "user": ["u", "u", "v", "v"],
        "heldout_user": False, "score_time": pd.to_datetime(
            ["2010-09-01 09:00", "2010-09-01 10:00", "2010-09-02 09:00", "2010-09-02 10:00"]),
        "scenario": [np.nan, 1, np.nan, np.nan], "y": [0, 1, 0, 0]})
    units, scores = ev.aggregate_sessions(meta, [np.array([0.2, 0.7, 0.9, 0.1])])
    assert units["y"].tolist() == [1, 0]
    assert scores[0].tolist() == [0.7, 0.9]
    assert units["scenario"].tolist()[0] == 1


def test_test_split_needs_allow_test(monkeypatch):
    monkeypatch.setattr(ev, "verify_test_checksum", lambda: "x")
    with pytest.raises(RuntimeError, match="allow-test"):
        ev.main(["single", "--name", "m", "--split", "test", "--preds", "p.parquet",
                 "--out", "o.json"])
