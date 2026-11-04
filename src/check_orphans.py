"""Check whether dropped orphan events and Logoffs include malicious activity.

Output is counts only, with no user IDs, because it is published with the repo.

Run from the repo root: python src/check_orphans.py
"""
import csv
import pathlib
from datetime import datetime

import duckdb

ANSWERS = pathlib.Path("data/raw/answers")
P = "data/processed"
SCENARIOS = ["r4.2-1", "r4.2-2", "r4.2-3"]
EVENT_TABLES = ["device", "file", "email", "http"]
TS_FORMAT = "%m/%d/%Y %H:%M:%S"  # Python also accepts the unpadded dates in insiders.csv


def load_answers(con):
    """Table `answers`: one row per malicious event line in the r4.2 answer files."""
    rows = []
    for scenario in SCENARIOS:
        for path in sorted((ANSWERS / scenario).glob("*.csv")):
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                for event_type, event_id, ts, user, pc, *_ in csv.reader(f):
                    rows.append((event_id, event_type, datetime.strptime(ts, TS_FORMAT),
                                 user, pc, scenario, path.name))
    con.execute("""
        CREATE TABLE answers (id VARCHAR, event_type VARCHAR, ts TIMESTAMP, "user" VARCHAR,
                              pc VARCHAR, scenario VARCHAR, source_file VARCHAR)
    """)
    con.executemany("INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def load_insiders(con):
    """Table `insiders`: the r4.2 insiders and their malicious activity windows."""
    with (ANSWERS / "insiders.csv").open(newline="") as f:
        rows = [(r["user"], r["scenario"], datetime.strptime(r["start"], TS_FORMAT),
                 datetime.strptime(r["end"], TS_FORMAT))
                for r in csv.DictReader(f) if r["dataset"] == "4.2"]
    con.execute("""
        CREATE TABLE insiders ("user" VARCHAR, scenario VARCHAR, start TIMESTAMP, "end" TIMESTAMP)
    """)
    con.executemany("INSERT INTO insiders VALUES (?, ?, ?, ?)", rows)


def load_dropped(con):
    """Table `dropped`: orphan events, plus logon-table rows not used by any session."""
    events = " UNION ALL ".join(
        f"""SELECT id, '{t}' AS source, 'orphan_event' AS kind, "user", ts
            FROM '{P}/{t}_sessioned.parquet' WHERE session_id IS NULL"""
        for t in EVENT_TABLES)
    # Every Logon starts or continues a session, and a Logoff is used only if it directly
    # follows a Logon for that user and PC. Same ordering as src/sessions.py.
    logon = f"""
        SELECT id, 'logon' AS source, 'unused_logon_row' AS kind, "user", ts
        FROM (SELECT id, "user", ts, activity,
                     lag(activity) OVER (PARTITION BY "user", pc ORDER BY ts, activity, id) AS prev
              FROM '{P}/logon.parquet')
        WHERE activity = 'Logoff' AND (prev IS NULL OR prev = 'Logoff')
    """
    con.execute(f"CREATE TABLE dropped AS {events} UNION ALL {logon}")


def show(con, title, sql):
    rows = con.execute(sql).fetchall()
    print(f"\n{title}")
    if not rows:
        print("  (none)")
    for row in rows:
        print("  " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in row))
    return rows


def main():
    con = duckdb.connect(config={"threads": 1})
    load_answers(con)
    load_insiders(con)
    load_dropped(con)

    show(con, "Answer files: event lines, distinct ids, users (by scenario)", """
        SELECT scenario, count(*), count(DISTINCT id), count(DISTINCT "user")
        FROM answers GROUP BY 1 ORDER BY 1""")
    show(con, "Answer files: event lines by type", """
        SELECT event_type, count(*), count(DISTINCT id) FROM answers GROUP BY 1 ORDER BY 1""")
    show(con, "Insiders: rows, distinct users", """
        SELECT count(*), count(DISTINCT "user") FROM insiders""")
    show(con, "Dropped rows: kind, source, rows, distinct ids, users", """
        SELECT kind, source, count(*), count(DISTINCT id), count(DISTINCT "user")
        FROM dropped GROUP BY 1, 2 ORDER BY 1, 2""")

    # (d) malicious events among the dropped rows
    con.execute("""
        CREATE TABLE hits AS
        SELECT a.id, a.event_type, a.ts, a."user", a.scenario, d.kind, d.source
        FROM answers a JOIN dropped d USING (id)
    """)
    total = show(con, "Malicious events among dropped rows: kind, distinct ids", """
        SELECT kind, count(DISTINCT id) FROM hits GROUP BY 1 ORDER BY 1""")
    for label, column in [("table", "source"), ("scenario", "scenario")]:
        show(con, f"Malicious dropped events by {label}: kind, {label}, distinct ids", f"""
            SELECT kind, {column}, count(DISTINCT id) FROM hits GROUP BY 1, 2 ORDER BY 1, 2""")
    show(con, "Malicious dropped events by user: kind, distinct users", """
        SELECT kind, count(DISTINCT "user") FROM hits GROUP BY 1 ORDER BY 1""")

    # (e) insiders with any orphan events, and orphans inside their activity windows
    show(con, "Insiders with orphan events: insiders of total", """
        SELECT count(DISTINCT i."user") FILTER (d.id IS NOT NULL), count(DISTINCT i."user")
        FROM insiders i LEFT JOIN dropped d ON i."user" = d."user" AND d.kind = 'orphan_event'""")
    show(con, "Insider orphan events by scenario: scenario, insiders, events, inside start-end window", """
        SELECT i.scenario, count(DISTINCT i."user"), count(*),
               count(*) FILTER (d.ts BETWEEN i.start AND i."end")
        FROM insiders i JOIN dropped d ON i."user" = d."user" AND d.kind = 'orphan_event'
        GROUP BY 1 ORDER BY 1""")
    show(con, "Insider dropped Logoffs by scenario: scenario, insiders, logoffs, inside start-end window", """
        SELECT i.scenario, count(DISTINCT i."user"), count(*),
               count(*) FILTER (d.ts BETWEEN i.start AND i."end")
        FROM insiders i JOIN dropped d ON i."user" = d."user" AND d.kind = 'unused_logon_row'
        GROUP BY 1 ORDER BY 1""")

    # (f) every malicious event id must exist in its processed table
    tables = {t: f"{t}_sessioned" for t in EVENT_TABLES} | {"logon": "logon"}
    processed = " UNION ALL ".join(
        f"""SELECT id, '{t}' AS event_type, "user", ts FROM '{P}/{name}.parquet'"""
        for t, name in tables.items())
    con.execute(f"CREATE TABLE processed AS {processed}")
    missing = show(con, "Answer ids missing from processed tables: id, type", """
        SELECT DISTINCT a.id, a.event_type FROM answers a
        ANTI JOIN processed p ON a.id = p.id AND a.event_type = p.event_type ORDER BY 2, 1""")
    show(con, "Answer rows found but with a different user or time: id, type", """
        SELECT DISTINCT a.id, a.event_type FROM answers a
        JOIN processed p ON a.id = p.id AND a.event_type = p.event_type
        WHERE a."user" <> p."user" OR a.ts <> p.ts ORDER BY 2, 1""")

    n = sum(row[1] for row in total)
    print(f"\nRESULT: {n:,} malicious events among dropped rows; "
          f"{len(missing):,} answer ids missing from processed tables")


if __name__ == "__main__":
    main()
