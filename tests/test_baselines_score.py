"""Comparison 1 scoring of the saved baselines: CLI guards and the never-overwrite rule."""
import sys

import pandas as pd
import pytest

from src import run


@pytest.mark.parametrize("argv, message", [
    (["xgb", "--splits", "valid"], "need --score"),
    (["rules", "--allow-test"], "need --score"),
    (["sessions", "--score", "--splits", "valid"], "only applies to xgb and rules"),
    (["xgb", "--score"], "--score needs --splits"),
    (["xgb", "--score", "--fit", "--splits", "valid"], "not allowed with"),
])
def test_score_cli_guards(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit):
        run.main()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("module", ["xgb", "rules"])
def test_test_split_needs_allow_test(module):
    if module == "xgb":
        from src.baselines import xgb
        call = lambda: xgb.score_saved(["calib", "test"])
    else:
        from src.baselines import rules
        call = lambda: rules.score_splits(["test"])
    with pytest.raises(SystemExit, match="needs --allow-test"):
        call()


def test_existing_predictions_are_matched_never_overwritten(tmp_path):
    from src.baselines.xgb import write_or_match
    preds = pd.DataFrame({"row_id": ["a", "b"], "score": [0.1, 0.9], "margin": [-2.0, 2.0]})
    path = tmp_path / "preds_valid.parquet"
    assert write_or_match(preds, path) == "written"
    assert write_or_match(preds.copy(), path).startswith("identical")
    changed = preds.assign(score=[0.1, 0.8])
    with pytest.raises(RuntimeError, match="differs"):
        write_or_match(changed, path)
    assert pd.read_parquet(path).equals(preds)          # untouched


def test_eval_split_is_passed_through(monkeypatch):
    """run.py eval must forward --split to eval.py, not read it as an abbreviation of --splits."""
    from src import eval as evaluation
    seen = {}
    monkeypatch.setattr(evaluation, "main", lambda argv: seen.setdefault("argv", argv))
    argv = ["single", "--name", "m", "--split", "test", "--allow-test", "--session-agg", "final",
            "--preds", "p.parquet", "--out", "o.json"]
    monkeypatch.setattr(sys, "argv", ["run.py", "eval", *argv])
    run.main()
    assert seen["argv"] == argv          # every eval option reaches eval.py, incl. --allow-test
