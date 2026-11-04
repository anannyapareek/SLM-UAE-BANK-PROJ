"""Scoring grid and part 1 features (session, logon, device and file).

Every feature uses only events with ts <= score_time.

LEAKAGE RULE: `end`, `end_type` and total session length must never become features.
`end` is used here only to place the scoring points. Nothing may reveal that a row is
the last one, so is_final is metadata and session_minutes_so_far is rounded down to the
scoring interval, with a minimum of one interval: exact elapsed time would be off the
hour only on final rows, and this way every value also occurs on hourly rows.
"""
import pathlib

import duckdb
import yaml

P = pathlib.Path("data/processed")
CONFIG = pathlib.Path("configs/features.yaml")
OUT = P / "features_part1.parquet"
META_COLUMNS = ["row_id", "session_id", "user", "pc", "score_time", "is_final"]
FORBIDDEN_FEATURES = {"end", "end_type", "is_final", "session_length"}
PART1_SOURCES = {"sessions", "logon", "logon_sessioned", "device_sessioned", "file_sessioned"}
QUANTILES = [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]


def build_grid(con, interval):
    """Table `grid`: hourly points strictly before end, plus one point at end."""
    # `end` stops here: `grid` keeps only what features are allowed to use.
    con.execute(f"""
        CREATE TEMP TABLE grid AS
        WITH points AS (
            SELECT session_id, "user", pc, start,
                   start + k * INTERVAL {interval} MINUTE AS score_time, false AS is_final
            FROM (SELECT *, unnest(range(1, ceil(
                      date_diff('second', start, "end") / ({interval} * 60.0))::BIGINT)) AS k
                  FROM '{P}/sessions.parquet')
            UNION ALL
            SELECT session_id, "user", pc, start, "end" AS score_time, true AS is_final
            FROM '{P}/sessions.parquet'
        )
        SELECT md5(concat_ws('|', session_id,
                             strftime(score_time, '%Y-%m-%d %H:%M:%S'))) AS row_id, *
        FROM points
    """)
    rows, sessions = con.execute(
        "SELECT count(*), count(DISTINCT session_id) FROM grid").fetchone()
    values = con.execute(f"""
        SELECT quantile_cont(n, {QUANTILES})
        FROM (SELECT count(*) AS n FROM grid GROUP BY session_id)
    """).fetchone()[0]
    print(f"grid: {rows:,} rows for {sessions:,} sessions")
    print("rows per session: " + ", ".join(
        f"p{q * 100:g} {v:g}" for q, v in zip(QUANTILES, values)))


def apply_burn_in(con, burn_in):
    """Remove rows whose score_time falls in the burn-in period."""
    first_event = con.execute("SELECT min(t) FROM (" + " UNION ALL ".join(
        f"SELECT min(ts) AS t FROM '{P}/{t}.parquet'" for t in burn_in["tables"]) + ")"
    ).fetchone()[0]
    cutoff = con.execute(
        f"SELECT TIMESTAMP '{first_event}' + INTERVAL {int(burn_in['days'])} DAY").fetchone()[0]

    before = con.execute(
        "SELECT count(*), count(DISTINCT session_id) FROM grid").fetchone()
    con.execute(f"DELETE FROM grid WHERE score_time < TIMESTAMP '{cutoff}'")
    after = con.execute(
        "SELECT count(*), count(DISTINCT session_id) FROM grid").fetchone()
    print(f"burn-in: {first_event} to {cutoff} ({burn_in['days']} days)")
    print(f"burn-in removed: {before[0] - after[0]:,} rows ({(before[0] - after[0]) / before[0]:.2%}), "
          f"{before[1] - after[1]:,} sessions ({(before[1] - after[1]) / before[1]:.2%})")
    print(f"scoring rows left: {after[0]:,} rows, {after[1]:,} sessions")


def static_sql(cfg):
    """Features that need only the grid row."""
    work_start, work_end = cfg["off_hours"]["work_start"], cfg["off_hours"]["work_end"]
    interval = cfg["scoring"]["interval_minutes"]
    weekend = ", ".join(f"'{d}'" for d in cfg["off_hours"]["weekend_days"])
    return {
        "session_minutes_so_far": f"""
            greatest(floor(date_diff('second', g.start, g.score_time)
                           / ({interval} * 60.0))::BIGINT, 1) * {interval}""",
        "off_hours": f"""
            (NOT (g.start::TIME >= TIME '{work_start}:00'
                  AND g.start::TIME < TIME '{work_end}:00'))::INTEGER""",
        "weekend": f"(strftime(g.start, '%a') IN ({weekend}))::INTEGER",
        "new_pc": "(g.start <= p.first_logon)::INTEGER",
    }


def count_table(con, cfg, features, source, relation, counters):
    """Count `source` events per grid row; return {feature: SQL column}."""
    minutes = cfg["windows"]["rolling_window_minutes"]
    recent = f"e.ts > g.score_time - INTERVAL {minutes} MINUTE"
    selects = {}
    for f in features:
        if f["source"] != source or f["window"] not in ("session_cumulative", "rolling_1h"):
            continue
        name = f["name"]
        if f["window"] == "rolling_1h":
            selects[name] = f"({counters[name.removesuffix('_1h')]}) AND {recent}"
        elif name in counters:
            selects[name] = counters[name]
    if not selects:
        return {}
    columns = ", ".join(f"count(*) FILTER ({cond}) AS {name}" for name, cond in selects.items())
    # The session join keeps both windows inside the session, so ts >= start holds.
    con.execute(f"""
        CREATE TEMP TABLE counts_{source} AS
        SELECT g.row_id, {columns}
        FROM grid g JOIN {relation} e ON e.session_id = g.session_id AND e.ts <= g.score_time
        GROUP BY g.row_id
    """)
    return {name: f"coalesce(counts_{source}.{name}, 0)" for name in selects}


def day_count(con, name, relation):
    """User's events that calendar day on all PCs, up to score_time."""
    con.execute(f"""
        CREATE TEMP TABLE daily_{name} AS
        SELECT "user", ts, ts::DATE AS day,
               count(*) OVER (PARTITION BY "user", ts::DATE ORDER BY ts) AS n
        FROM ({relation})
    """)
    con.execute(f"""
        CREATE TEMP TABLE day_{name} AS
        SELECT g.row_id, d.n
        FROM grid g ASOF LEFT JOIN daily_{name} d
          ON g."user" = d."user" AND g.score_time::DATE = d.day AND g.score_time >= d.ts
    """)
    return {name: f"coalesce(day_{name}.n, 0)"}


def build_features(con, cfg, features):
    """Write the feature table and return its feature names, in config order."""
    device = f"'{P}/device_sessioned.parquet'"
    file_events = f"""(
        SELECT *, lower(regexp_extract(filename, '\\.([A-Za-z0-9]+)$', 1)) AS extension
        FROM '{P}/file_sessioned.parquet'
    )"""
    file_counters = {"file_events": "true"}
    file_counters |= {f"file_ext_{x}": f"e.extension = '{x}'" for x in cfg["file_extensions"]}

    sql = static_sql(cfg)
    sql |= count_table(con, cfg, features, "logon_sessioned", f"'{P}/logon_sessioned.parquet'",
                       {"unlocks_so_far": "e.is_unlock"})
    sql |= count_table(con, cfg, features, "device_sessioned", device,
                       {"usb_connects": "e.activity = 'Connect'"})
    sql |= count_table(con, cfg, features, "file_sessioned", file_events, file_counters)
    # Logoffs are left out, otherwise the session's own Logoff would mark the final row.
    sql |= day_count(con, "logon_events_today_so_far",
                     f"SELECT \"user\", ts FROM '{P}/logon.parquet' WHERE activity = 'Logon'")
    sql |= day_count(con, "usb_connects_today_so_far", f"""
        SELECT "user", ts FROM {device} WHERE activity = 'Connect' AND session_id IS NOT NULL""")

    names = [f["name"] for f in features]
    missing = [n for n in names if n not in sql]
    if missing:
        raise RuntimeError(f"no implementation for features: {missing}")

    joins = "".join(
        f" LEFT JOIN {t} USING (row_id)" for t in sorted(
            {expr.split("(", 1)[1].split(".")[0] for n, expr in sql.items()
             if n in names and expr.startswith("coalesce")}))
    meta = ", ".join(f'g."{c}"' for c in META_COLUMNS)
    columns = ", ".join(f"{sql[n]} AS {n}" for n in names)
    con.execute(f"""
        COPY (
            SELECT {meta}, {columns}
            FROM grid g
            LEFT JOIN (SELECT "user", pc, min(ts) AS first_logon FROM '{P}/logon.parquet'
                       WHERE activity = 'Logon' GROUP BY 1, 2) p USING ("user", pc)
            {joins}
            ORDER BY g.session_id, g.score_time
        ) TO '{OUT}' (FORMAT parquet)
    """)
    return names


def print_summary(con, names):
    rows, ids = con.execute(f"SELECT count(*), count(DISTINCT row_id) FROM '{OUT}'").fetchone()
    grid_rows = con.execute("SELECT count(*) FROM grid").fetchone()[0]
    if not rows == ids == grid_rows:
        raise RuntimeError(f"{grid_rows:,} grid rows, {rows:,} output rows, {ids:,} unique row_ids")
    print(f"features_part1: {rows:,} rows, row_id unique")

    print("feature | null rate | min | mean | max")
    for name in names:
        nulls, lo, mean, hi = con.execute(f"""
            SELECT 1 - count({name}) / count(*), min({name}), avg({name}), max({name})
            FROM '{OUT}'
        """).fetchone()
        print(f"{name} | {nulls:.2%} | {lo:g} | {mean:.4f} | {hi:g}")


def run():
    cfg = yaml.safe_load(CONFIG.read_text())
    features = [f for f in cfg["features"]
                if f["source"] in PART1_SOURCES and f["window"] != "baseline"]
    forbidden = FORBIDDEN_FEATURES & {f["name"] for f in features}
    if forbidden:
        raise RuntimeError(f"forbidden features in config: {sorted(forbidden)}")

    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    build_grid(con, cfg["scoring"]["interval_minutes"])
    apply_burn_in(con, cfg["burn_in"])
    print_summary(con, build_features(con, cfg, features))
