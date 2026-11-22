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


def test_diag_rows_are_the_logged_mismatch_set():
    from src.slm.rationale_diag import DIAG_ROWS
    assert len(set(DIAG_ROWS)) == 11
    assert sum(m == "c1" for m, _, _ in DIAG_ROWS) == 1 and sum(m == "c2" for m, _, _ in DIAG_ROWS) == 10


def test_diag_last_position():
    from src.slm.rationale_diag import last_position
    mask = np.array([[0, 0, 1, 1], [1, 1, 1, 1]])
    pos = np.array([[0, 0, 0, 1], [0, 1, 2, 3]])
    seen = {"input_ids": np.zeros((2, 4)), "attention_mask": mask, "position_ids": pos, "cache_position": np.arange(4)}
    assert last_position(seen, 0) == {"prompt_tokens_padded": 4, "pad_tokens": 2, "attention_mask_last": 1,
                                      "position_id_last": 1, "cache_position_last": 3}
    assert last_position(seen, 1)["position_id_last"] == 3
    one = {"input_ids": np.zeros((1, 3)), "attention_mask": None, "position_ids": np.array([[0, 1, 2]]),
           "cache_position": None}
    assert last_position(one, 0)["position_id_last"] == 2 and last_position(one, 0)["pad_tokens"] is None


def test_step8_protocol_check_rejects_absolute_inputs():
    import copy, json, pathlib
    from src.slm import step8_report
    d = pathlib.Path("results/step8")
    if not (d / "latency_gpu.json").exists():
        pytest.skip("lab outputs not present")
    gpu, xgb = (json.loads((d / f).read_text()) for f in ("latency_gpu.json", "latency_xgb_lab.json"))
    sha = gpu["test_ids_sha256"]
    step8_report.check_protocol(gpu, xgb, sha)
    bad = copy.deepcopy(gpu)
    bad["inputs"] = "D:\\x\\data\\step8\\latency_sample.parquet"
    with pytest.raises(SystemExit):
        step8_report.check_protocol(bad, xgb, sha)
    bad = copy.deepcopy(gpu)
    bad["partial"] = True
    with pytest.raises(SystemExit):
        step8_report.check_protocol(bad, xgb, sha)


def test_compute_line_arithmetic():
    from scripts.step7_table3 import line
    c = line({"rows": 1000, "row_share": (0.2, 0.0)}, 100.0, 10_000.0)
    assert c["slm_seconds"] == pytest.approx(2.0) and c["xgb_seconds"] == pytest.approx(0.1)
    assert c["implied_rows_per_second"] == pytest.approx(1000 / 2.1)
    assert c["speedup_vs_slm_on_all_rows"] == pytest.approx(1000 / 2.1 / 100)


def test_diag_controls_disjoint_from_mismatches():
    from src.slm.rationale_diag import CONTROL_ROWS, DIAG_ROWS
    assert len(set(CONTROL_ROWS)) == 7 and not set(CONTROL_ROWS) & set(DIAG_ROWS)


def test_generate_kwargs():
    from src.slm.rationales import GEN_KWARGS, GEN_KWARGS_17NOV
    assert GEN_KWARGS_17NOV == {"do_sample": False, "num_beams": 1, "temperature": None, "top_p": None, "top_k": None}
    assert "repetition_penalty" not in GEN_KWARGS_17NOV   # inherited 1.1 on 17 Nov (decisions log 2026-11-23)
    assert GEN_KWARGS == {**GEN_KWARGS_17NOV, "repetition_penalty": 1.0}


def test_rationales_refuses_17nov_file(monkeypatch):
    import sys
    from src.slm import rationales
    monkeypatch.setattr(sys, "argv", ["rationales", "--device", "cpu", "--out", str(rationales.V1)])
    with pytest.raises(SystemExit, match="refusing"):
        rationales.main()


def test_poisson_interval():
    pytest.importorskip("scipy")
    from src.slm.rationale_penalty_check import poisson_ci
    lo, hi = poisson_ci(10)
    assert lo == pytest.approx(4.795, abs=1e-3) and hi == pytest.approx(18.390, abs=1e-3)
    assert poisson_ci(0)[0] == 0.0
