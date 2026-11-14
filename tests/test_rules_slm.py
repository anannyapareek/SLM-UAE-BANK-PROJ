"""Step 7.3 rules + SLM ordering (src/rules_slm.py)."""
import numpy as np

from src import rules_slm


def test_rule_count_decides_first_then_slm_within_a_count():
    count = np.array([0, 0, 1, 1, 2])
    p = np.array([0.99, 0.10, 0.01, 0.95, 0.001])
    order = np.argsort(-rules_slm.combine(count, p))
    assert order.tolist() == [4, 3, 2, 0, 1]


def test_slm_score_of_one_cannot_reach_the_next_count():
    s = rules_slm.combine(np.array([0, 1]), np.array([1.0, 0.0]))
    assert s[0] < s[1]


def test_rule_count_uses_config_weights_and_ignores_tie_break():
    ids = [r["id"] for r in rules_slm.RULES]
    w = {r["id"]: r["weight"] for r in rules_slm.RULES}
    fired = np.array(["", ids[0], f"{ids[0]},{ids[1]}"])
    assert rules_slm.rule_count(fired).tolist() == [0.0, w[ids[0]], w[ids[0]] + w[ids[1]]]
