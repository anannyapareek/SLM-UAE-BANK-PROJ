"""Scoring grid and features: session, logon, device, file, email, http, LDAP, baseline.

Every feature uses only events with ts <= score_time.

LEAKAGE RULE: `end`, `end_type` and total session length must never become features.
`end` is used here only to place the scoring points and to select earlier sessions for
baselines. Nothing may reveal that a row is the last one, so is_final is metadata and
session_minutes_so_far is rounded down to the scoring interval, with a minimum of one
interval: exact elapsed time would be off the hour only on final rows, and this way
every value also occurs on hourly rows.
"""
import os
import pathlib

import duckdb
import yaml

# The environment overrides let tests build from a sample or with another config.
P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
CONFIG = pathlib.Path(os.environ.get("FEATURES_CONFIG", "configs/features.yaml"))
OUT_DIR = pathlib.Path(os.environ.get("FEATURES_OUT_DIR", P))
OUT_PART1 = OUT_DIR / "features_part1.parquet"
OUT_PART2 = OUT_DIR / "features_part2.parquet"
OUT = OUT_DIR / "features.parquet"
META_COLUMNS = ["row_id", "session_id", "user", "pc", "score_time", "is_final"]
FORBIDDEN_FEATURES = {"end", "end_type", "is_final", "session_length"}
PART1_SOURCES = {"sessions", "logon", "logon_sessioned", "device_sessioned", "file_sessioned"}
LDAP_FEATURES = ["role", "department", "functional_unit"]
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


def outside_work_hours(cfg, column):
    """SQL for a timestamp falling outside off_hours.work_start to work_end."""
    work_start, work_end = cfg["off_hours"]["work_start"], cfg["off_hours"]["work_end"]
    return (f"(NOT ({column}::TIME >= TIME '{work_start}:00' "
            f"AND {column}::TIME < TIME '{work_end}:00'))::INTEGER")


def static_sql(cfg):
    """Features that need only the grid row."""
    interval = cfg["scoring"]["interval_minutes"]
    weekend = ", ".join(f"'{d}'" for d in cfg["off_hours"]["weekend_days"])
    return {
        "session_minutes_so_far": f"""
            greatest(floor(date_diff('second', g.start, g.score_time)
                           / ({interval} * 60.0))::BIGINT, 1) * {interval}""",
        "off_hours": outside_work_hours(cfg, "g.start"),
        "weekend": f"(strftime(g.start, '%a') IN ({weekend}))::INTEGER",
    }


def new_pc_sql(con, joins):
    """1 if this user has no logon on this PC strictly before session start."""
    con.execute(f"""
        CREATE TEMP TABLE new_pc_rows AS
        SELECT g.row_id, (g.start <= p.first_logon)::INTEGER AS new_pc
        FROM grid g
        JOIN (SELECT "user", pc, min(ts) AS first_logon FROM '{P}/logon.parquet'
              WHERE activity = 'Logon' GROUP BY 1, 2) p USING ("user", pc)
    """)
    joins.append("new_pc_rows")
    return {"new_pc": "new_pc_rows.new_pc"}


def window_sums(con, cfg, features, source, relation, metrics, joins, running):
    """Session-cumulative and rolling sums of per-event `metrics` for one event table.

    Running totals per session are matched to each score_time with an ASOF join, so
    both windows stay inside the session and only use ts <= score_time. A rolling value
    is the running total at score_time minus the running total at score_time - window.
    """
    wanted = [f for f in features if f["source"] == source
              and f["window"] in ("session_cumulative", "rolling_1h")
              and f["name"].removesuffix("_1h") in metrics]
    rolling = [f["name"] for f in wanted if f["window"] == "rolling_1h"]
    bases = list(metrics)
    columns = ", ".join(f"sum({metrics[b]}) OVER w AS {b}" for b in bases)
    con.execute(f"""
        CREATE TEMP TABLE running_{source} AS
        SELECT e.session_id, e.ts, {columns}
        FROM {relation} e
        WHERE e.session_id IS NOT NULL
        WINDOW w AS (PARTITION BY e.session_id ORDER BY e.ts)
    """)
    running.update({b: f"running_{source}" for b in bases})

    def totals_at(table, score_time):
        picks = ", ".join(f"r.{b}::BIGINT AS {b}" for b in bases)
        con.execute(f"""
            CREATE TEMP TABLE {table} AS
            SELECT g.row_id, {picks}
            FROM grid g ASOF LEFT JOIN running_{source} r
              ON g.session_id = r.session_id AND {score_time} >= r.ts
        """)
        joins.append(table)

    totals_at(f"now_{source}", "g.score_time")
    if rolling:
        minutes = cfg["windows"]["rolling_window_minutes"]
        totals_at(f"before_{source}", f"g.score_time - INTERVAL {minutes} MINUTE")

    sql = {}
    for f in wanted:
        name, base = f["name"], f["name"].removesuffix("_1h")
        sql[name] = f"coalesce(now_{source}.{base}, 0)"
        if name in rolling:
            sql[name] += f" - coalesce(before_{source}.{base}, 0)"
    return sql


def day_count(con, name, relation, joins):
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
    joins.append(f"day_{name}")
    return {name: f"coalesce(day_{name}.n, 0)"}


def write_features(con, path, names, sql, joins):
    missing = [n for n in names if n not in sql]
    if missing:
        raise RuntimeError(f"no implementation for features: {missing}")
    meta = ", ".join(f'g."{c}"' for c in META_COLUMNS)
    columns = ", ".join(f"{sql[n]} AS {n}" for n in names)
    tables = "".join(f" LEFT JOIN {t} USING (row_id)" for t in joins)
    con.execute(f"""
        COPY (SELECT {meta}, {columns} FROM grid g {tables} ORDER BY g.session_id, g.score_time)
        TO '{path}' (FORMAT parquet)
    """)


def build_part1(con, cfg, features, running):
    """Session, logon, device and file features."""
    device = f"'{P}/device_sessioned.parquet'"
    # The header is the first token of file_content, before any text_rules strip.
    header_rules = cfg["file_header_extensions"]
    con.execute("CREATE TEMP TABLE file_headers (header VARCHAR PRIMARY KEY, extension VARCHAR)")
    con.executemany("INSERT INTO file_headers VALUES (?, ?)", list(header_rules["headers"].items()))
    file_events = f"""(
        SELECT f.*, lower(regexp_extract(f.filename, '\\.([A-Za-z0-9]+)$', 1)) AS extension,
               coalesce(h.extension, '{header_rules['default']}') AS header_extension
        FROM '{P}/file_sessioned.parquet' f
        JOIN '{P}/file_content.parquet' c USING (id)
        LEFT JOIN file_headers h ON split_part(c.content, ' ', 1) = h.header
    )"""
    file_metrics = {"file_events": "1",
                    "file_header_mismatch": "(e.extension <> e.header_extension)::INTEGER"}
    file_metrics |= {f"file_ext_{x}": f"(e.extension = '{x}')::INTEGER"
                     for x in cfg["file_extensions"]}

    joins = []
    sql = static_sql(cfg) | new_pc_sql(con, joins)
    sql |= window_sums(con, cfg, features, "logon_sessioned", f"'{P}/logon_sessioned.parquet'",
                       {"unlocks_so_far": "e.is_unlock::INTEGER"}, joins, running)
    sql |= window_sums(con, cfg, features, "device_sessioned", device,
                       {"usb_connects": "(e.activity = 'Connect')::INTEGER"}, joins, running)
    sql |= window_sums(con, cfg, features, "file_sessioned", file_events, file_metrics,
                       joins, running)
    # Logoffs are left out, otherwise the session's own Logoff would mark the final row.
    sql |= day_count(con, "logon_events_today_so_far",
                     f"SELECT \"user\", ts FROM '{P}/logon.parquet' WHERE activity = 'Logon'",
                     joins)
    sql |= day_count(con, "usb_connects_today_so_far", f"""
        SELECT "user", ts FROM {device} WHERE activity = 'Connect' AND session_id IS NOT NULL""",
                     joins)
    write_features(con, OUT_PART1, [f["name"] for f in features], sql, joins)


def ldap_sql(con, cfg, joins):
    """Latest LDAP snapshot from a month strictly before the month of score_time.

    A monthly file may reflect the state at month end, so the current month is not used.
    Presence or absence in LDAP never becomes a feature: departure is future information.
    """
    columns = ", ".join(LDAP_FEATURES)
    con.execute(f"""
        CREATE TEMP TABLE ldap_snapshots AS
        SELECT user_id, {columns},
               strptime(regexp_extract(month_file, '(\\d{{4}}-\\d{{2}})\\.csv$', 1), '%Y-%m')
                   AS snapshot_month
        FROM '{P}/ldap.parquet'
    """)
    picks = ", ".join(f"l.{c}" for c in LDAP_FEATURES)
    con.execute(f"""
        CREATE TEMP TABLE ldap_rows AS
        SELECT g.row_id, {picks}
        FROM grid g ASOF LEFT JOIN ldap_snapshots l
          ON g."user" = l.user_id AND date_trunc('month', g.score_time) > l.snapshot_month
    """)
    joins.append("ldap_rows")
    missing = cfg["fairness"]["missing_value"]
    return {c: f"coalesce(ldap_rows.{c}, '{missing}')" for c in LDAP_FEATURES}


def build_part2(con, cfg, features, running):
    """Email, http and LDAP features."""
    domain = cfg["internal_email_domain"]
    # Every email row is sent mail: the user column is the sender (see decisions_log.md).
    email_events = f"""(
        SELECT *, len([x FOR x IN list_distinct(
                           string_split(lower(concat_ws(';', "to", cc, bcc)), ';'))
                       IF x <> '' AND split_part(x, '@', 2) <> '{domain}']) AS external
        FROM '{P}/email_sessioned.parquet'
    )"""
    email_metrics = {
        "emails_sent": "1",
        "external_recipients": "e.external",
        "attachments_count": "e.attachments::INTEGER",
        "email_bytes": "e.size::BIGINT",
        "after_hours_emails": outside_work_hours(cfg, "e.ts"),
    }

    lists = cfg["http_keyword_lists"][cfg["http_lists"]]
    con.execute("CREATE TEMP TABLE http_hosts (host VARCHAR PRIMARY KEY, category VARCHAR)")
    hosts = [(host, category) for category, members in lists.items() for host in members]
    if hosts:
        con.executemany("INSERT INTO http_hosts VALUES (?, ?)", hosts)
    http_events = f"""(
        SELECT h.session_id, h.ts, c.category
        FROM (SELECT session_id, ts,
                     lower(regexp_extract(url, '^https?://([^/]+)', 1)) AS host
              FROM '{P}/http_sessioned.parquet') h
        LEFT JOIN http_hosts c USING (host)
    )"""
    http_metrics = {"http_visits": "1"}
    http_metrics |= {f"http_{c}": f"coalesce(e.category = '{c}', false)::INTEGER" for c in lists}

    joins = []
    sql = window_sums(con, cfg, features, "email_sessioned", email_events, email_metrics,
                      joins, running)
    sql |= window_sums(con, cfg, features, "http_sessioned", http_events, http_metrics,
                       joins, running)
    sql |= ldap_sql(con, cfg, joins)
    write_features(con, OUT_PART2, [f["name"] for f in features], sql, joins)


def build_baseline(con, cfg, bases, running):
    """Table `baseline`: per session and elapsed hour, stats of the user's earlier sessions.

    A row at elapsed hour h is compared with each earlier session's cumulative value at
    hour h, so partial counts meet partial counts. An earlier session shorter than h
    contributes its final value, because its running total stops there. Earlier sessions
    have end < this session's start, so everything used was known by then.
    """
    interval = cfg["scoring"]["interval_minutes"]
    days = int(cfg["windows"]["baseline_days"])
    con.execute(f"""
        CREATE TEMP TABLE row_hours AS
        SELECT DISTINCT session_id, session_minutes_so_far // {interval} AS h FROM '{OUT_PART1}'
    """)
    hours = con.execute("SELECT max(h) FROM row_hours").fetchone()[0]
    con.execute(f"""
        CREATE TEMP TABLE session_hours AS
        SELECT session_id, h, start + h * INTERVAL {interval} MINUTE AS at
        FROM (SELECT session_id, start, unnest(range(1, {hours + 1})) AS h
              FROM '{P}/sessions.parquet')
    """)
    con.execute(f"""
        CREATE TEMP TABLE prior_sessions AS
        SELECT c.session_id, p.session_id AS prior_id
        FROM '{P}/sessions.parquet' c
        JOIN '{P}/sessions.parquet' p
          ON p."user" = c."user" AND p."end" < c.start
         AND p."end" >= c.start - INTERVAL {days} DAY
    """)

    tables = sorted({running[b] for b in bases})
    joins = ""
    for t in tables:
        picks = ", ".join(f"r.{b}" for b in bases if running[b] == t)
        con.execute(f"""
            CREATE TEMP TABLE profile_{t} AS
            SELECT sh.session_id, sh.h, {picks}
            FROM session_hours sh ASOF LEFT JOIN {t} r
              ON sh.session_id = r.session_id AND sh.at >= r.ts
        """)
        joins += f" LEFT JOIN profile_{t} USING (session_id, h)"
    values = ", ".join(f"coalesce(profile_{running[b]}.{b}, 0)::DOUBLE AS {b}" for b in bases)
    con.execute(f"""
        CREATE TEMP TABLE profile AS
        SELECT session_id, h, {values} FROM session_hours {joins}
    """)

    stats = ", ".join(f"avg(p.{b}) AS {b}_mean, stddev_samp(p.{b}) AS {b}_std" for b in bases)
    con.execute(f"""
        CREATE TEMP TABLE baseline AS
        SELECT r.session_id, r.h, count(*) AS prior_sessions, {stats}
        FROM row_hours r
        JOIN prior_sessions x USING (session_id)
        JOIN profile p ON p.session_id = x.prior_id AND p.h = r.h
        GROUP BY r.session_id, r.h
    """)


def check_has_baseline(con, cfg):
    """Every scoring row must have a baseline after burn-in; it is not a model feature."""
    minimum = cfg["windows"]["baseline_min_sessions"]
    short = con.execute(f"""
        SELECT count(*)
        FROM row_hours r LEFT JOIN baseline b USING (session_id, h)
        WHERE coalesce(b.prior_sessions, 0) < {minimum}
    """).fetchone()[0]
    if short:
        raise RuntimeError(f"{short:,} session hours have fewer than {minimum} earlier sessions")
    print(f"has_baseline: 1 on every row (at least {minimum} earlier sessions)")


def assemble(con, cfg, names, part1_names, bases):
    """Write features.parquet: metadata, then every feature in config order."""
    minimum = cfg["windows"]["baseline_min_sessions"]
    floor = cfg["windows"]["std_floor"]
    interval = cfg["scoring"]["interval_minutes"]
    sql = {n: f"p1.{n}" if n in part1_names else f"p2.{n}" for n in names}
    for b in bases:
        current = f"p1.{b}" if b in part1_names else f"p2.{b}"
        sql[f"{b}_z"] = f"""
            CASE WHEN b.prior_sessions >= {minimum}
                 THEN ({current} - b.{b}_mean) / greatest(coalesce(b.{b}_std, 0), {floor}) END"""

    meta = ", ".join(f'p1."{c}"' for c in META_COLUMNS)
    columns = ", ".join(f"{sql[n]} AS {n}" for n in names)
    con.execute(f"""
        COPY (
            SELECT {meta}, {columns}
            FROM '{OUT_PART1}' p1
            JOIN '{OUT_PART2}' p2 USING (row_id)
            LEFT JOIN baseline b ON p1.session_id = b.session_id
             AND p1.session_minutes_so_far // {interval} = b.h
            ORDER BY p1.session_id, p1.score_time
        ) TO '{OUT}' (FORMAT parquet)
    """)


def print_summary(con, names):
    rows, ids = con.execute(f"SELECT count(*), count(DISTINCT row_id) FROM '{OUT}'").fetchone()
    part1_rows = con.execute(f"SELECT count(*) FROM '{OUT_PART1}'").fetchone()[0]
    if not rows == ids == part1_rows:
        raise RuntimeError(f"{part1_rows:,} part 1 rows, {rows:,} rows, {ids:,} unique row_ids")
    print(f"features: {rows:,} rows, matches part 1, row_id unique")

    print("feature | null rate | min | mean | max")
    for name in names:
        if name in LDAP_FEATURES:
            nulls, values = con.execute(f"""
                SELECT 1 - count({name}) / count(*), count(DISTINCT {name}) FROM '{OUT}'
            """).fetchone()
            print(f"{name} | {nulls:.2%} | {values} categories")
            continue
        nulls, lo, mean, hi = con.execute(f"""
            SELECT 1 - count({name}) / count(*), min({name}), avg({name}), max({name})
            FROM '{OUT}'
        """).fetchone()
        print(f"{name} | {nulls:.2%} | {lo:g} | {mean:.4f} | {hi:g}")


def run():
    cfg = yaml.safe_load(CONFIG.read_text())
    features = cfg["features"]
    names = [f["name"] for f in features]
    forbidden = FORBIDDEN_FEATURES & set(names)
    if forbidden:
        raise RuntimeError(f"forbidden features in config: {sorted(forbidden)}")
    part1 = [f for f in features if f["source"] in PART1_SOURCES and f["window"] != "baseline"]
    part2 = [f for f in features if f["source"] not in PART1_SOURCES and f["window"] != "baseline"]
    bases = [f["name"].removesuffix("_z") for f in features
             if f["window"] == "baseline" and f["name"].endswith("_z")]

    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    con.execute(f"SET temp_directory = '{P}/.duckdb_tmp'")
    build_grid(con, cfg["scoring"]["interval_minutes"])
    apply_burn_in(con, cfg["burn_in"])

    running = {}
    build_part1(con, cfg, part1, running)
    build_part2(con, cfg, part2, running)
    build_baseline(con, cfg, bases, running)
    check_has_baseline(con, cfg)
    assemble(con, cfg, names, {f["name"] for f in part1}, bases)
    print_summary(con, names)
