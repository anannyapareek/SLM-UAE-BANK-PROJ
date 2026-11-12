"""Comparison 2 text blocks (serialise.text_block) and the unchanged structured prompts."""
import json
import random

import pandas as pd
import pytest

from conftest import ROOT
from src.slm import serialise

FIXTURE = ROOT / "tests/fixtures/structured_prompts_03b1641.json"
T0 = pd.Timestamp("2010-09-01 09:00")


def words(text):
    """A deterministic stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


# ---------- structured prompts: byte for byte as before the text change ----------

def decode(row):
    return {k: (float("nan") if v == "nan" else v) for k, v in row.items()}


def test_structured_prompts_unchanged():
    """scoring_prompt and training_text equal the 03b1641 output on 300 fixed rows."""
    fixture = json.loads(FIXTURE.read_text())
    rows = [decode(r) for r in fixture["rows"]]
    assert [serialise.scoring_prompt(r) for r in rows] == fixture["scoring_prompt"]
    assert [serialise.training_text(r, 1) for r in rows] == fixture["training_text_yes"]
    assert [serialise.training_text(r, 0) for r in rows] == fixture["training_text_no"]


# ---------- text blocks ----------

def event(i, source, minutes, content, meta="m", flagged=False):
    return {"event_id": f"e{i:03d}", "source": source, "ts": T0 + pd.Timedelta(minutes=minutes),
            "meta": meta, "content": content, "flagged": flagged}


def random_events(seed, n=40):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        source = rng.choice(["http", "http", "email", "file"])
        content = " ".join(f"w{rng.randint(0, 99)}" for _ in range(rng.randint(0, 60)))
        out.append(event(i, source, rng.randint(0, 600), content,
                         flagged=source == "http" and rng.random() < 0.1))
    return out


@pytest.mark.parametrize("seed", range(20))
def test_budget_respected(seed):
    events = random_events(seed)
    for budget in (10, 50, 256):
        block, info = serialise.text_block(events, T0 + pd.Timedelta(hours=10), words, budget)
        assert words(block) <= budget and info["tokens"] == words(block)
        assert block.startswith(serialise.TEXT_HEADER)


def test_budget_respected_with_the_model_tokenizer():
    """The real budget is in Qwen2.5 tokens; checked if the pinned tokenizer is available."""
    pytest.importorskip("transformers")
    from src.slm import common
    try:
        tok = common.load_tokenizer(common.config()["text"]["tokenizer"])
    except Exception as e:                      # offline machine without the files
        pytest.skip(f"tokenizer not available: {e}")
    count = lambda t: len(tok.encode(t, add_special_tokens=False))
    for seed in range(10):
        block, info = serialise.text_block(random_events(seed, 80), T0 + pd.Timedelta(hours=10),
                                           count, 256)
        assert count(block) <= 256 and info["selected_events"] >= 1


def test_deterministic_regardless_of_input_order():
    events = random_events(3)
    first = serialise.text_block(events, T0 + pd.Timedelta(hours=10), words, 120)
    for seed in range(5):
        shuffled = events[:]
        random.Random(seed).shuffle(shuffled)
        assert serialise.text_block(shuffled, T0 + pd.Timedelta(hours=10), words, 120) == first


def test_leakage_only_events_up_to_score_time():
    events = [event(1, "email", 0, "before"), event(2, "email", 60, "at score time"),
              event(3, "email", 61, "after secret"), event(4, "http", 120, "later secret",
                                                          flagged=True)]
    block, info = serialise.text_block(events, T0 + pd.Timedelta(minutes=60), words, 256)
    assert "before" in block and "at score time" in block
    assert "secret" not in block
    assert info["eligible_events"] == 2


def test_prompt_ends_with_label_and_starts_with_structured_prompt():
    fixture = json.loads(FIXTURE.read_text())
    row = decode(fixture["rows"][0])
    block, _ = serialise.text_block(random_events(1), T0 + pd.Timedelta(hours=10), words, 100)
    text = serialise.text_scoring_prompt(row, block)
    assert text.endswith("\nLabel:")
    assert text.startswith(serialise.prompt(row))
    assert text == serialise.prompt(row) + block + serialise.SUFFIX
    # the completion after "Label:" is the same target as for structured prompts
    assert serialise.target(row, 1).startswith(" yes\nReason: ")


def test_whole_events_only_and_priority():
    events = [event(1, "http", 10, "a " * 30),                       # other http: 33 tokens
              event(2, "email", 20, "b " * 5),                       # email
              event(3, "file", 30, "c " * 5, meta=".doc"),           # file
              event(4, "http", 5, "d " * 5, meta="wikileaks.org", flagged=True),
              event(5, "email", 40, "e " * 5)]
    block, info = serialise.text_block(events, T0 + pd.Timedelta(hours=1), words, 30)
    lines = block.split("\n")[2:]
    full = {serialise.event_line(e).lstrip("\n") for e in events}
    assert all(line in full for line in lines)                       # whole events only
    assert "wikileaks.org" in block                                  # tier 0 first
    assert "a a" not in block                                        # 33 tokens: skipped
    shown = [line.split()[1] for line in lines]
    assert shown == sorted(shown)                                    # shown oldest first
    # budget 30 = header 1 + flagged http 8 + email 8 + email 8 (5 left): file (8) skipped
    assert ".doc" not in block
    assert info["truncated"] and info["selected_events"] == 3


def test_newest_first_within_a_tier():
    events = [event(i, "email", i, "x " * 10) for i in range(6)]
    block, _ = serialise.text_block(events, T0 + pd.Timedelta(hours=1), words, 30)
    assert "email 09:05" in block and "email 09:04" in block and "email 09:00" not in block


def test_no_events_gives_none():
    block, info = serialise.text_block([], T0, words, 256)
    assert block == serialise.TEXT_HEADER + serialise.TEXT_NONE
    assert info["selected_events"] == 0 and not info["truncated"]


def test_text_budget_config_is_consistent():
    from src.slm import common
    cfg = common.config()
    assert cfg["text"]["budget_tokens"] == cfg["prompt"]["text_budget_tokens"] == serialise.TEXT_BUDGET
