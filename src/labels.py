"""Labels from the r4.2 answer files: malicious events, session labels, row labels.

This is the only pipeline stage that reads data/raw/answers (besides the coverage check
in src/check_orphans.py). No feature code reads its outputs.

A scoring row is positive only once something malicious has happened in its session:
y = 1 if the session is positive and first_malicious_time <= score_time.

The report (results/labels_report.md) holds counts and session ids only, no user IDs,
because it is published with the repo.
"""
import csv
import os
import pathlib
from datetime import datetime

import duckdb

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
ANSWERS = pathlib.Path("data/raw/answers")
RELEASE_README = pathlib.Path("data/raw/r4.2/readme.txt")
REPORT = (pathlib.Path("results/labels_report.md") if P == pathlib.Path("data/processed")
          else P / "labels_report.md")
RELEASE = "4.2"
EVENT_TABLES = ["logon", "device", "file", "email", "http"]
TS_FORMAT = "%m/%d/%Y %H:%M:%S"


def read_insiders():
    """{answer file name: (incident_id, insider, scenario)}, numbered in insiders.csv order."""
    with (ANSWERS / "insiders.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == RELEASE]
    return {r["details"]: (i, r["user"], int(r["scenario"])) for i, r in enumerate(rows, 1)}


def read_malicious_events(insiders):
    """One row per event line. Rows are variable-length, so only the first 5 fields are used."""
    rows = []
    for folder in sorted(ANSWERS.glob(f"r{RELEASE}-*")):
        scenario = int(folder.name.rsplit("-", 1)[1])
        for path in sorted(folder.glob("*.csv")):
            incident_id, insider, listed_scenario = insiders[path.name]
            if listed_scenario != scenario:
                raise RuntimeError(f"{path.name}: folder scenario {scenario}, "
                                   f"insiders.csv says {listed_scenario}")
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                for event_type, event_id, ts, user, pc, *_ in csv.reader(f):
                    rows.append((event_id, event_type, datetime.strptime(ts, TS_FORMAT),
                                 user, pc, scenario, incident_id, insider))
    return rows


def build(con, insiders):
    con.execute("""
        CREATE TABLE malicious (event_id VARCHAR, type VARCHAR, ts TIMESTAMP, "user" VARCHAR,
                                pc VARCHAR, scenario INTEGER, incident_id INTEGER,
                                insider VARCHAR)
    """)
    con.executemany("INSERT INTO malicious VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    read_malicious_events(insiders))
    con.execute(f"COPY malicious TO '{P}/malicious_events.parquet' (FORMAT parquet)")

    # Match on event id first; fall back to user, pc and time within a session window.
    events = " UNION ALL ".join(
        f"SELECT id, '{t}' AS type, session_id FROM '{P}/{t}_sessioned.parquet'"
        for t in EVENT_TABLES)
    con.execute(f"""
        CREATE TABLE matched AS
        SELECT m.*, coalesce(e.session_id, s.session_id) AS session_id,
               CASE WHEN e.session_id IS NOT NULL THEN 'id'
                    WHEN s.session_id IS NOT NULL THEN 'fallback'
                    ELSE 'unmatched' END AS matched_by
        FROM malicious m
        LEFT JOIN ({events}) e ON e.id = m.event_id AND e.type = m.type
        LEFT JOIN '{P}/sessions.parquet' s
          ON e.session_id IS NULL AND s."user" = m."user" AND s.pc = m.pc
         AND m.ts BETWEEN s.start AND s."end"
    """)
    # The actor is the incident's insider, even when the session belongs to someone else
    # (scenario 3: the insider logs in as the supervisor).
    con.execute("""
        CREATE TABLE positive AS
        SELECT session_id, min(ts) AS first_malicious_time,
               arg_min(scenario, ts) AS scenario, arg_min(incident_id, ts) AS incident_id,
               arg_min(insider, ts) AS actor_user, count(DISTINCT incident_id) AS incidents,
               bool_or("user" = insider) AS insider_session
        FROM matched WHERE session_id IS NOT NULL GROUP BY 1
    """)
    con.execute(f"""
        COPY (
            SELECT f.row_id, f.session_id, f."user", f.score_time,
                   (p.session_id IS NOT NULL AND p.first_malicious_time <= f.score_time)::INTEGER
                       AS y,
                   (p.session_id IS NOT NULL)::INTEGER AS in_positive_session,
                   p.scenario, p.incident_id, p.actor_user
            FROM '{P}/features.parquet' f LEFT JOIN positive p USING (session_id)
            ORDER BY f.session_id, f.score_time
        ) TO '{P}/labels.parquet' (FORMAT parquet)
    """)


def recheck(con):
    """Rebuild y from events placed by user, pc and time only, without event ids."""
    return con.execute(f"""
        WITH first AS (
            SELECT s.session_id, min(m.ts) AS first_time
            FROM malicious m JOIN '{P}/sessions.parquet' s
              ON s."user" = m."user" AND s.pc = m.pc AND m.ts BETWEEN s.start AND s."end"
            GROUP BY 1)
        SELECT count(*),
               count(*) FILTER (l.y <> (f.first_time IS NOT NULL
                                        AND f.first_time <= l.score_time)::INTEGER),
               count(*) FILTER (l.in_positive_session <> (f.first_time IS NOT NULL)::INTEGER)
        FROM '{P}/labels.parquet' l LEFT JOIN first f USING (session_id)
    """).fetchone()


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in r) + " |"
                    for r in rows]


def report(con, insiders):
    q = lambda sql: con.execute(sql).fetchall()
    labels = f"'{P}/labels.parquet'"
    out = ["# Labels report", "", f"Generated by `python run.py labels`. Release {RELEASE}.", ""]

    total, distinct = q("SELECT count(*), count(DISTINCT event_id) FROM malicious")[0]
    counts = dict(q("SELECT matched_by, count(*) FROM matched GROUP BY 1"))
    out += ["## Matching", ""] + table(["Measure", "Events"], [
        ("Malicious events", total), ("Distinct event ids", distinct),
        ("Matched by event id", counts.get("id", 0)),
        ("Matched by user, pc and time", counts.get("fallback", 0)),
        ("Unmatched", counts.get("unmatched", 0))])
    unmatched = q("""SELECT event_id, type, ts::VARCHAR FROM matched
                     WHERE matched_by = 'unmatched' ORDER BY ts""")
    if unmatched:
        out += ["", "Unmatched events:", ""] + table(["Event id", "Type", "Time"], unmatched)

    sessions, insider_sessions, multi = q("""
        SELECT count(*), count(*) FILTER (insider_session), count(*) FILTER (incidents > 1)
        FROM positive""")[0]
    rows, positive_rows, in_positive, kept_positive, kept_sessions = q(f"""
        SELECT count(*), sum(y), sum(in_positive_session),
               count(DISTINCT session_id) FILTER (in_positive_session = 1),
               count(DISTINCT session_id)
        FROM {labels}""")[0]
    out += ["", "## Counts", ""] + table(["Measure", "Count"], [
        ("Incidents in insiders.csv", len(insiders)),
        ("Distinct insiders", len({u for _, u, _ in insiders.values()})),
        ("Positive sessions", sessions),
        ("Positive sessions with scoring rows", kept_positive),
        ("Positive sessions spanning more than one incident", multi),
        ("Scoring rows", rows), ("Positive rows (y = 1)", positive_rows),
        ("Rows in positive sessions before the first malicious event",
         in_positive - positive_rows)])
    out += ["", f"Positive rate: {positive_rows / rows:.4%} of rows, "
            f"{kept_positive / kept_sessions:.4%} of sessions."]

    listed = {}
    for _, _, scenario in insiders.values():
        listed[scenario] = listed.get(scenario, 0) + 1
    found = {s: (i, n, y) for s, i, n, y in q(f"""
        SELECT scenario, count(DISTINCT incident_id), count(DISTINCT session_id), sum(y)
        FROM {labels} WHERE in_positive_session = 1 GROUP BY 1""")}
    out += ["", "## By scenario", ""] + table(
        ["Scenario", "Incidents listed", "Incidents with a positive session",
         "Positive sessions", "Positive rows"],
        [(s, listed[s], *found.get(s, (0, 0, 0))) for s in sorted(listed)])

    others = q(f"""
        WITH s AS (SELECT session_id, scenario, incident_id, count(*) AS events,
                          string_agg(DISTINCT type, ', ' ORDER BY type) AS types
                   FROM matched WHERE "user" <> insider GROUP BY 1, 2, 3)
        SELECT s.session_id::VARCHAR, s.scenario, s.incident_id::VARCHAR, s.types, s.events,
               sum(l.y), count(l.row_id)
        FROM s LEFT JOIN {labels} l USING (session_id)
        GROUP BY 1, 2, 3, 4, 5 ORDER BY 1""")
    out += ["", "## Sessions of a user other than the incident's insider", "",
            f"{len(others)} sessions, {sum(r[4] for r in others):,} events. Kept positive: the "
            "insider logged in as the supervisor, and in a zero-trust setting the session is "
            "what gets scored. In labels.parquet, `user` is the session's user and "
            "`actor_user` is the insider.", ""] + table(
        ["Session id", "Scenario", "Incident id", "Event types", "Events",
         "Positive rows", "Rows"], others)

    quoted = [line.strip("* \n") for line in RELEASE_README.read_text().splitlines()
              if "instances of insider threats" in line]
    out += ["", "## Release readme", ""]
    for line in quoted:
        out.append(f'The r4.2 readme says: "{line}" This does not match the {len(insiders)} '
                   "incidents in insiders.csv, which agree with the answer files, so those are used.")

    checked, wrong_y, wrong_positive = recheck(con)
    out += ["", "## Recheck", "",
            f"Labels rebuilt from events placed by user, pc and time only (no event ids): "
            f"{wrong_y:,} differences in y and {wrong_positive:,} in in_positive_session "
            f"across {checked:,} rows."]
    if wrong_y or wrong_positive:
        raise RuntimeError("labels do not match the recheck")
    return out


def run():
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    insiders = read_insiders()
    files = {f.name for f in ANSWERS.glob(f"r{RELEASE}-*/*.csv")}
    if files != set(insiders):
        raise RuntimeError("answer files and insiders.csv do not list the same incidents")
    build(con, insiders)
    lines = report(con, insiders)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
