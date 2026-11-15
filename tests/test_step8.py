import numpy as np
import pytest

from src.slm import latency, rationales, xgb_latency


def test_parse_label_and_reason():
    assert rationales.parse(" yes\nReason: highest 30-day z: USB 3.1.") == ("yes", "highest 30-day z: USB 3.1.")
    assert rationales.parse(" no\nReason: no 30-day z is 2 or more.") == ("no", "no 30-day z is 2 or more.")
    assert rationales.parse(" no") == ("no", "")
    assert rationales.parse(" maybe\nsomething else") == ("maybe", "")


def test_latency_summaries_agree():
    s = [0.010, 0.020, 0.030, 0.040, 1.0]
    for f in (latency.summary, xgb_latency.summary):
        out = f(s)
        assert out["n"] == 5
        assert out["median_ms"] == pytest.approx(30.0)
        assert out["p95_ms"] == pytest.approx(np.percentile(s, 95) * 1e3)
