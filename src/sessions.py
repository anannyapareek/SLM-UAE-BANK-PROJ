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
END_TYPES = ["logoff", "next_logon", "cap"]
QUANTILES = [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]


def build_sessions(con, logon, cap_hours):
    """Create table `sessions` from relation `logon`; return the orphan Logoff count."""
    # At equal timestamps 'Logoff' sorts before 'Logon', so a session closes first.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE logon_ordered AS
        SELECT "user", pc, ts, activity,
               lag(activity)  OVER w AS prev_activity,
               lead(activity) OVER w AS next_activity,
               lead(ts)       OVER w AS next_ts,
               ts + INTERVAL {int(cap_hours)} HOUR AS cap_ts
        FROM {logon}
        WINDOW w AS (PARTITION BY "user", pc ORDER BY ts, activity, id)
    """)

    # The next event for this user and PC decides how a session ends. The cap only
    # applies when no Logoff follows, so a real Logoff is kept even after 12 hours.
    # LEAKAGE: `end` and `end_type` must never become features (see module docstring).
    con.execute("""
        CREATE OR REPLACE TABLE sessions AS
        SELECT row_number() OVER (ORDER BY "user", pc, ts) AS session_id,
               "user", pc, ts AS start,
               CASE WHEN next_activity = 'Logoff' THEN next_ts
                    WHEN next_activity = 'Logon' AND next_ts <= cap_ts THEN next_ts
                    ELSE cap_ts END AS "end",
               CASE WHEN next_activity = 'Logoff' THEN 'logoff'
                    WHEN next_activity = 'Logon' AND next_ts <= cap_ts THEN 'next_logon'
                    ELSE 'cap' END AS end_type
        FROM logon_ordered
        WHERE activity = 'Logon'
        ORDER BY session_id
    """)

    # An orphan Logoff has no open session: nothing, or another Logoff, comes before it.
    return con.execute("""
        SELECT count(*) FROM logon_ordered
        WHERE activity = 'Logoff' AND (prev_activity IS NULL OR prev_activity = 'Logoff')
    """).fetchone()[0]


def assign_sql(events):
    """SQL giving `events` a session_id, or NULL when no session contains the event.

    Sessions for one user and PC never overlap, so the latest start at or before ts
    is the only candidate. An event exactly on a next_logon boundary goes to the
    session that is starting.
    """
    return f"""
        SELECT e.*, CASE WHEN e.ts <= s."end" THEN s.session_id END AS session_id
        FROM {events} e
        ASOF LEFT JOIN sessions s
          ON e."user" = s."user" AND e.pc = s.pc AND e.ts >= s.start
    """


def assign_events(con, table):
    """Write <table>_sessioned.parquet and return (rows, orphan events)."""
    source, out = f"'{P}/{table}.parquet'", f"'{P}/{table}_sessioned.parquet'"
    con.execute(f"COPY ({assign_sql(source)}) TO {out} (FORMAT parquet)")

    expected = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    # count(DISTINCT id) decodes every id, so it also catches unreadable data.
    rows, ids, matched = con.execute(
        f"SELECT count(*), count(DISTINCT id), count(session_id) FROM {out}").fetchone()
    if not rows == ids == expected:
        raise RuntimeError(f"{table}: {expected:,} events in, {rows:,} rows and {ids:,} ids out")
    return rows, rows - matched


def print_summary(con, cap_hours, orphan_logoffs, orphans):
    total = con.execute("SELECT count(*) FROM sessions").fetchone()[0]
    print(f"sessions: {total:,}")
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
    orphans = {table: assign_events(con, table) for table in EVENT_TABLES}
    print_summary(con, cap_hours, orphan_logoffs, orphans)
