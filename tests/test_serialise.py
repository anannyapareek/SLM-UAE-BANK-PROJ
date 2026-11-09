"""SLM serialisation (src/slm/serialise.py, docs/serialisation.md)."""
import ast
import math
import random

import pytest
import yaml

from conftest import ROOT
from src.slm import serialise

# Label-side columns of labels.parquet and splits.parquet; none may reach the text.
LABEL_COLUMNS = {"y", "in_positive_session", "scenario", "incident_id", "actor_user",
                 "first_malicious_time", "split", "heldout_user", "incident_seen_before_test"}


@pytest.fixture(scope="module")
def feature_names():
    cfg = yaml.safe_load((ROOT / "configs/features.yaml").read_text())
    return [f["name"] for f in cfg["features"]]


def synthetic_row(seed, feature_names):
    rng = random.Random(seed)
    row = {"row_id": f"r{seed}", "session_id": seed, "user": "U", "pc": "PC",
           "score_time": "2010-03-01 10:00:00", "is_final": False}
    for name in feature_names:
        if name in ("role", "department", "functional_unit"):
            row[name] = rng.choice(["Engineer", "2 - Sales", "unknown"])
        elif name.endswith("_z"):
            row[name] = rng.choice([None, float("nan"), rng.uniform(-3, 6)])
        elif name in ("off_hours", "weekend", "new_pc"):
            row[name] = rng.randint(0, 1)
        elif name == "session_minutes_so_far":
            row[name] = 60 * rng.randint(1, 14)
        else:
            row[name] = rng.randint(0, 40) * (1000 if name == "email_bytes" else 1)
    return row


def test_every_feature_used_or_excluded(feature_names):
    used = set(serialise.USED)
    assert len(serialise.USED) == len(used), "a column is read by two lines"
    assert used == set(feature_names), (set(feature_names) - used, used - set(feature_names))
    assert not used & set(serialise.EXCLUDED)
    # every column of features.parquet: the 44 features plus the excluded metadata
    meta = ["row_id", "session_id", "user", "pc", "score_time", "is_final"]
    assert set(serialise.EXCLUDED) == set(meta)


def test_no_label_column_used(feature_names):
    assert not set(serialise.USED) & LABEL_COLUMNS
    assert not set(feature_names) & LABEL_COLUMNS
    # The module never names a label column as a string.
    tree = ast.parse((ROOT / "src/slm/serialise.py").read_text())
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
               and isinstance(n.value, str)}
    assert not strings & LABEL_COLUMNS


def test_label_columns_do_not_change_the_text(feature_names):
    row = synthetic_row(1, feature_names)
    with_labels = dict(row, y=1, in_positive_session=True, scenario=2, incident_id=5,
                       actor_user="X", split="train", heldout_user=False)
    assert serialise.prompt(with_labels) == serialise.prompt(row)
    assert serialise.reason(with_labels) == serialise.reason(row)


def test_reads_only_listed_columns(feature_names):
    """Every column outside USED can be removed without breaking or changing the prompt."""
    row = synthetic_row(2, feature_names)
    only_used = {k: row[k] for k in serialise.USED}
    assert serialise.prompt(only_used) == serialise.prompt(row)
    assert serialise.reason(only_used) == serialise.reason(row)


def test_deterministic(feature_names):
    rows = [synthetic_row(s, feature_names) for s in range(200)]
    first = [serialise.training_text(r, s % 2) for s, r in enumerate(rows)]
    second = [serialise.training_text(dict(reversed(list(r.items()))), s % 2)
              for s, r in enumerate(rows)]
    assert first == second


def test_fixed_layout(feature_names):
    text = serialise.scoring_prompt(synthetic_row(3, feature_names))
    lines = text.split("\n")
    assert lines[0] == serialise.HEADER
    assert [l.split(":")[0] for l in lines[1:-1]] == [label for label, _, _ in serialise.LINES]
    assert lines[-1] == "Label:"


def test_target_format(feature_names):
    row = synthetic_row(4, feature_names)
    for label, word in ((1, "yes"), (0, "no"), (True, "yes"), (False, "no")):
        full = serialise.training_text(row, label)
        tail = full.split("\n")[-2:]
        assert tail[0] == f"Label: {word}"
        assert tail[1].startswith("Reason: ") and tail[1].endswith(".")
        assert ". " not in tail[1]          # one sentence
    # the reason does not depend on the label
    assert (serialise.training_text(row, 1).split("Reason: ")[1]
            == serialise.training_text(row, 0).split("Reason: ")[1])


def test_reason_from_top_deviations():
    row = {c: 0.0 for c, _, _ in serialise.DEVIATIONS}
    assert serialise.reason(row) == "no 30-day z is 2 or more."
    row.update(usb_connects_z=4.14, file_events_z=2.5, emails_sent_z=9.0,
               http_job_search_z=float("nan"), external_recipients_z=None)
    assert serialise.reason(row) == "highest 30-day z: emails sent +9.0, USB connects +4.1."
    row = {c: -5.0 for c, _, _ in serialise.DEVIATIONS}
    row["file_events_z"] = 2.0
    assert serialise.reason(row) == "highest 30-day z: file copies +2.0."


def test_missing_z_is_written_as_na(feature_names):
    row = synthetic_row(5, feature_names)
    row["usb_connects_z"] = None
    row["file_events_z"] = math.nan
    assert "30-day z: USB n/a files n/a " in serialise.prompt(row)
