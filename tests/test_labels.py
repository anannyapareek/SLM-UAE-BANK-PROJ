"""Label rule: a row is positive only once something malicious has happened in its session.

Uses the processed label tables only, not the answer files.
"""
import pytest


@pytest.fixture(scope="module")
def have_labels(table, processed):
    if not (processed / "labels.parquet").exists():
        pytest.skip("labels not built for this folder")


def test_labels_cover_every_scoring_row(have_labels, con, table):
    rows, ids, unmatched = con.execute(f"""
        SELECT count(*), count(DISTINCT l.row_id), count(*) FILTER (f.row_id IS NULL)
        FROM {table('labels')} l LEFT JOIN {table('features')} f USING (row_id)
    """).fetchone()
    features = con.execute(f"SELECT count(*) FROM {table('features')}").fetchone()[0]
    assert rows == ids == features and unmatched == 0


def test_y_follows_the_first_malicious_event(have_labels, con, table):
    """Recomputed from malicious events placed by user, pc and time, not by id."""
    wrong = con.execute(f"""
        WITH first AS (
            SELECT s.session_id, min(m.ts) AS first_time
            FROM {table('malicious_events')} m
            JOIN {table('sessions')} s
              ON s."user" = m."user" AND s.pc = m.pc AND m.ts BETWEEN s.start AND s."end"
            GROUP BY 1)
        SELECT count(*) FILTER (l.y <> (f.first_time IS NOT NULL AND f.first_time <= l.score_time)::INTEGER),
               count(*) FILTER (l.in_positive_session <> (f.first_time IS NOT NULL)::INTEGER),
               count(*) FILTER (l.y = 1 AND l.in_positive_session = 0)
        FROM {table('labels')} l LEFT JOIN first f USING (session_id)
    """).fetchone()
    assert wrong == (0, 0, 0)


def test_every_malicious_event_has_a_session(have_labels, con, table):
    unplaced = con.execute(f"""
        SELECT count(*) FROM {table('malicious_events')} m
        WHERE NOT EXISTS (SELECT 1 FROM {table('sessions')} s
                          WHERE s."user" = m."user" AND s.pc = m.pc
                            AND m.ts BETWEEN s.start AND s."end")
    """).fetchone()[0]
    assert unplaced == 0
