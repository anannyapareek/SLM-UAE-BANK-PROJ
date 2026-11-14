"""Step 7.2 hybrid band rule (src/hybrid.py)."""
import numpy as np

from src import hybrid


def test_band_is_inclusive_and_swaps_only_inside():
    x = np.array([0.01, 0.05, 0.10, 0.20, 0.30])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert hybrid.band_mask(x, 0.05, 0.20).tolist() == [False, True, True, True, False]
    assert hybrid.hybrid_score(x, s, 0.05, 0.20).tolist() == [0.01, 0.8, 0.7, 0.6, 0.30]


def test_empty_band_is_xgboost_alone():
    x = np.array([0.1, 0.2, 0.3])
    s = np.array([0.9, 0.8, 0.7])
    assert hybrid.hybrid_score(x, s, 0.5, 0.4).tolist() == x.tolist()


def test_band_selection_ignores_rows_outside_and_respects_the_cap():
    rng = np.random.default_rng(0)
    x = [rng.random(2000) for _ in hybrid.CFG["seeds"]]
    s = [rng.random(2000) for _ in hybrid.CFG["seeds"]]
    y = (rng.random(2000) < 0.05).astype(int)
    frame, _ = hybrid.search(x, s, y, 0.10)
    assert len(frame) and frame["share_max"].max() <= 0.10
    assert frame["row_pr_auc_mean"].is_monotonic_decreasing
