"""Build logon sessions and assign device, file, email and http events to them.

LEAKAGE RULE: `end` and `end_type` use future information, because they are only
known once a session is over. They may define the session and the final scoring
point, but they must NEVER become features.
"""
import pathlib

import duckdb
import yaml

P = pathlib.Path("data/processed")
CONFIG = pathlib.Path("configs/features.yaml")
EVENT_TABLES = ["device", "file", "email", "http"]
END_TYPES = ["logoff", "cap"]
QUANTILES = [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]


def build_sessions(con, logon, cap_hours):
    """Create table `sessions` from relation `logon`; return the orphan Logoff count.

    The readme says screen unlocks are recorded as Logons. A Logon while a session is
    open, within cap_hours of its start, is an unlock and continues the session. Once
    the cap has passed, a Logon closes the old session at start + cap and starts a new
    one. A real Logoff always ends the session, even after cap_hours.
    """
    # At equal timestamps 'Logoff' sorts before 'Logon', so a session closes first.
    # `run` groups the Logons between two Logoffs for one user and PC.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE logon_ordered AS
        SELECT "user", pc, ts, activity, id,
               lag(activity)  OVER w AS prev_activity,
               lead(activity) OVER w AS next_activity,
               lead(ts)       OVER w AS next_ts,
               count(*) FILTER (activity = 'Logoff') OVER w AS run
        FROM {logon}
        WINDOW w AS (PARTITION BY "user", pc ORDER BY ts, activity, id)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE logons AS
        SELECT "user", pc, ts, run, next_ts,
               coalesce(next_activity, 'none') AS next_activity,
               row_number() OVER (PARTITION BY "user", pc, run ORDER BY ts, id) AS pos
        FROM logon_ordered
        WHERE activity = 'Logon'
    """)

    # Whether a Logon is an unlock depends on the session start, which depends on the
    # Logons before it, so each run is walked in order.
    # LEAKAGE: `end` and `end_type` must never become features (see module docstring).
    # Total unlocks per session are left out of this table for the same reason.
    con.execute(f"""
        CREATE OR REPLACE TABLE sessions AS
        WITH RECURSIVE walk AS (
            SELECT "user", pc, run, pos, ts, next_activity, next_ts, ts AS start
            FROM logons WHERE pos = 1
            UNION ALL
            SELECT l."user", l.pc, l.run, l.pos, l.ts, l.next_activity, l.next_ts,
                   CASE WHEN l.ts <= w.start + INTERVAL {int(cap_hours)} HOUR
                        THEN w.start ELSE l.ts END
            FROM walk w
            JOIN logons l ON l."user" = w."user" AND l.pc = w.pc
                         AND l.run = w.run AND l.pos = w.pos + 1
        ),
        grouped AS (
            SELECT "user", pc, start,
                   max_by(next_activity, ts) AS last_next_activity,
                   max_by(next_ts, ts) AS last_next_ts
            FROM walk
            GROUP BY "user", pc, run, start
        )
        SELECT row_number() OVER (ORDER BY "user", pc, start) AS session_id,
               "user", pc, start,
               CASE WHEN last_next_activity = 'Logoff' THEN last_next_ts
                    ELSE start + INTERVAL {int(cap_hours)} HOUR END AS "end",
               CASE WHEN last_next_activity = 'Logoff' THEN 'logoff'
                    ELSE 'cap' END AS end_type
        FROM grouped
        ORDER BY session_id
    """)

    # An orphan Logoff has no open session: nothing, or another Logoff, comes before it.
    return con.execute("""
        SELECT count(*) FROM logon_ordered
        WHERE activity = 'Logoff' AND (prev_activity IS NULL OR prev_activity = 'Logoff')
    """).fetchone()[0]


def assign_sql(events, extra=""):
    """SQL giving `events` a session_id, or NULL when no session contains the event.

    Sessions for one user and PC never overlap, so the latest start at or before ts
    is the only candidate.
    """
    return f"""
        SELECT e.*, CASE WHEN e.ts <= s."end" THEN s.session_id END AS session_id {extra}
        FROM {events} e
        ASOF LEFT JOIN sessions s
          ON e."user" = s."user" AND e.pc = s.pc AND e.ts >= s.start
    """


def assign_events(con, table, extra=""):
    """Write <table>_sessioned.parquet and return (rows, orphan events)."""
    source, out = f"'{P}/{table}.parquet'", f"'{P}/{table}_sessioned.parquet'"
    con.execute(f"COPY ({assign_sql(source, extra)}) TO {out} (FORMAT parquet)")

    expected = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    # count(DISTINCT id) decodes every id, so it also catches unreadable data.
    rows, ids, matched = con.execute(
        f"SELECT count(*), count(DISTINCT id), count(session_id) FROM {out}").fetchone()
    if not rows == ids == expected:
        raise RuntimeError(f"{table}: {expected:,} events in, {rows:,} rows and {ids:,} ids out")
    return rows, rows - matched


def print_summary(con, cap_hours, orphan_logoffs, unlocks, orphans):
    total = con.execute("SELECT count(*) FROM sessions").fetchone()[0]
    print(f"sessions: {total:,}")
    print(f"unlocks (Logons that continue a session): {unlocks:,}")
    print(f"orphan logoffs dropped: {orphan_logoffs:,}")

    counts = dict(con.execute("SELECT end_type, count(*) FROM sessions GROUP BY 1").fetchall())
    print("end_type:")
    for end_type in END_TYPES:
        n = counts.get(end_type, 0)
        print(f"  {end_type}: {n:,} ({n / total:.2%})")

    values = con.execute(f"""
        SELECT quantile_cont(date_diff('second', start, "end") / 3600.0, {QUANTILES})
        FROM sessions
    """).fetchone()[0]
    print("session length (hours):")
    for q, hours in zip(QUANTILES, values):
        print(f"  p{q * 100:g}: {hours:.2f}")

    over_cap = con.execute(f"""
        SELECT count(*) FROM sessions WHERE "end" > start + INTERVAL {int(cap_hours)} HOUR
    """).fetchone()[0]
    print(f"sessions longer than {cap_hours}h: {over_cap:,} ({over_cap / total:.2%})")

    print("orphan events:")
    for table, (rows, n) in orphans.items():
        print(f"  {table}: {n:,} of {rows:,} ({n / rows:.4%})")


def run():
    cap_hours = yaml.safe_load(CONFIG.read_text())["session_cap_hours"]
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})

    orphan_logoffs = build_sessions(con, f"'{P}/logon.parquet'", cap_hours)
    con.execute(f"COPY sessions TO '{P}/sessions.parquet' (FORMAT parquet)")
    # logon_sessioned.parquet marks unlocks, so features can count them up to a score time.
    assign_events(con, "logon", """, e.activity = 'Logon' AND e.ts > s.start
                                     AND e.ts <= s."end" AS is_unlock""")
    unlocks = con.execute(
        f"SELECT count(*) FILTER (is_unlock) FROM '{P}/logon_sessioned.parquet'").fetchone()[0]
    orphans = {table: assign_events(con, table) for table in EVENT_TABLES}
    print_summary(con, cap_hours, orphan_logoffs, unlocks, orphans)
