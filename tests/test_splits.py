"""Split checks (a), (b), (c), (h) and the test_ids.csv overwrite guard."""
import duckdb
import pytest
import yaml

from conftest import ROOT


@pytest.fixture(scope="module")
def boundaries():
    return yaml.safe_load((ROOT / "configs/splits.yaml").read_text())["boundaries"]


@pytest.fixture(scope="module")
def have_splits(processed):
    if not (processed / "splits.parquet").exists():
        pytest.skip("splits not built for this folder")


def test_heldout_users_only_in_test(have_splits, con, table):
    rows = con.execute(
        f"SELECT count(*) FROM {table('splits')} WHERE heldout_user AND split <> 'test'").fetchone()[0]
    assert rows == 0


def test_time_order(have_splits, con, table, boundaries):
    limits = {"train": f">= TIMESTAMP '{boundaries['valid'][0]}'",
              "valid": f">= TIMESTAMP '{boundaries['calib'][0]}'",
              "calib": f">= TIMESTAMP '{boundaries['test'][0]}'",
              "test": f"< TIMESTAMP '{boundaries['test'][0]}'"}
    for split, condition in limits.items():
        bad = con.execute(f"""
            SELECT count(*) FROM {table('splits')} x JOIN {table('labels')} l USING (row_id)
            WHERE x.split = '{split}' AND l.score_time {condition}""").fetchone()[0]
        assert bad == 0, split


def test_sessions_stay_in_one_split(have_splits, con, table, boundaries):
    multi = con.execute(f"""
        SELECT count(*) FROM (SELECT session_id FROM {table('splits')} GROUP BY 1
                              HAVING count(DISTINCT split) > 1)""").fetchone()[0]
    assert multi == 0
    spans = " ".join(f"WHEN t >= TIMESTAMP '{a}' AND t < TIMESTAMP '{b}' + INTERVAL 1 DAY THEN '{n}'"
                     for n, (a, b) in boundaries.items())
    crossing = con.execute(f"""
        WITH k AS (SELECT DISTINCT session_id, split FROM {table('splits')})
        SELECT count(*) FROM k JOIN {table('sessions')} s USING (session_id)
        WHERE (SELECT CASE {spans} END FROM (SELECT s.start AS t)) IS DISTINCT FROM k.split
           OR (SELECT CASE {spans} END FROM (SELECT s."end" AS t)) IS DISTINCT FROM k.split
    """).fetchone()[0]
    assert crossing == 0


def test_test_incidents_have_no_positives_elsewhere(have_splits, con, table):
    leaked = con.execute(f"""
        WITH t AS (SELECT DISTINCT l.incident_id FROM {table('splits')} x
                   JOIN {table('labels')} l USING (row_id) WHERE x.split = 'test' AND l.y = 1)
        SELECT count(*) FROM {table('splits')} x JOIN {table('labels')} l USING (row_id)
        WHERE x.split <> 'test' AND l.in_positive_session = 1
          AND l.incident_id IN (SELECT incident_id FROM t)""").fetchone()[0]
    assert leaked == 0


def test_freeze_refuses_to_overwrite(tmp_path, monkeypatch):
    from src import splits

    con = duckdb.connect(config={"threads": 1})
    con.execute(f"""COPY (SELECT * FROM (VALUES ('b', 'test'), ('a', 'test'), ('c', 'train'))
                    t(row_id, split)) TO '{tmp_path}/splits.parquet' (FORMAT parquet)""")
    log = tmp_path / "log.md"
    log.write_text("| date | decision | reason |\n")
    monkeypatch.setattr(splits, "P", tmp_path)
    monkeypatch.setattr(splits, "TEST_IDS", tmp_path / "test_ids.csv")
    monkeypatch.setattr(splits, "LOG", log)

    digest = splits.freeze(con)
    assert (tmp_path / "test_ids.csv").read_text() == "row_id\na\nb\n"
    assert splits.verify_test_checksum() == digest
    with pytest.raises(RuntimeError, match="already exists"):
        splits.freeze(con)
    (tmp_path / "test_ids.csv").write_text("row_id\na\n")
    with pytest.raises(RuntimeError, match="does not match"):
        splits.verify_test_checksum()
    assert splits.freeze(con, force=True) == digest
    assert log.read_text().count("Frozen test_ids.csv") == 1   # same hash: no second line


# Stages that run before the freeze can exist, the freeze stages themselves, and the driver.
# Every other stage in run.py, including any added later, counts as post-freeze.
PRE_FREEZE = {"data", "sessions", "features", "features_text", "labels", "explore_positives",
              "prep_report"}
FREEZE = {"splits", "leakage"}
DRIVER = {"all"}


class Verified(Exception):
    pass


class ReadData(Exception):
    pass


def test_post_freeze_stages_verify_checksum_first(monkeypatch):
    """Each post-freeze stage reaches verify_test_checksum() before it opens any data."""
    import sys

    import pandas as pd
    import pyarrow.parquet as pq

    from src import run, splits

    post_freeze = sorted(set(run.STAGES) - PRE_FREEZE - FREEZE - DRIVER)
    assert {"xgb", "rules", "eval", "eval-baselines"} <= set(post_freeze)

    def verified():
        raise Verified

    def read_data(*args, **kwargs):
        raise ReadData

    # verify_test_checksum() looks up logged_checksum() in src.splits at call time, so this
    # catches the call however a stage imported it.
    monkeypatch.setattr(splits, "logged_checksum", verified)
    monkeypatch.setattr(duckdb, "connect", read_data)
    monkeypatch.setattr(pd, "read_parquet", read_data)
    monkeypatch.setattr(pq, "read_table", read_data)
    for stage in post_freeze:
        monkeypatch.setattr(sys, "argv", ["run.py", stage])
        with pytest.raises(Verified):
            run.main()
