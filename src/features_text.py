"""Event-level text table for Comparison 2.

Kept at event level so later steps can join on session_id and filter ts <= score_time
without leakage. Nothing here is aggregated per scoring row. Rows are sorted by session
and time, so a session_id filter can skip row groups.
"""
import pathlib

import duckdb
import yaml

P = pathlib.Path("data/processed")
CONFIG = pathlib.Path("configs/features.yaml")
OUT = P / "features_text.parquet"
UNSORTED = P / "features_text_unsorted.parquet"
SOURCES = ["email", "file", "http"]
FILE_HEADER = r"^[0-9A-Fa-f]{2}(-[0-9A-Fa-f]{2})*(\s+|$)"


def content_sql(source, rules):
    """File content starts with a hex-encoded header, which some rows contain alone."""
    if source == "file" and rules["strip_file_header"]:
        return f"regexp_replace(c.content, '{FILE_HEADER}', '')"
    return "c.content"


def run():
    cfg = yaml.safe_load(CONFIG.read_text())
    build = cfg["text_build"]
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    con.execute(f"SET temp_directory = '{P}/.duckdb_tmp'")
    con.execute(f"SET memory_limit = '{build['memory_limit']}'")

    # Inner joins drop events with no session. features.parquet holds only the sessions
    # that keep at least one scoring row after burn-in.
    events = " UNION ALL ".join(f"""
        SELECT e.session_id, e."user", e.pc, '{s}' AS source, e.id AS event_id, e.ts,
               {content_sql(s, cfg['text_rules'])} AS content
        FROM '{P}/{s}_sessioned.parquet' e
        JOIN kept USING (session_id)
        JOIN '{P}/{s}_content.parquet' c USING (id)""" for s in SOURCES)
    con.execute(f"CREATE TEMP TABLE kept AS SELECT DISTINCT session_id FROM '{P}/features.parquet'")

    # Two passes: joining and sorting in one query exceeds memory_limit.
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"COPY ({events}) TO '{UNSORTED}' (FORMAT parquet)")
    con.execute("SET preserve_insertion_order = true")
    con.execute(f"""
        COPY (SELECT * FROM '{UNSORTED}' ORDER BY session_id, ts, event_id)
        TO '{OUT}' (FORMAT parquet, ROW_GROUP_SIZE {int(build['row_group_size'])})
    """)
    UNSORTED.unlink()

    print("source | rows | median tokens | empty content")
    for source, rows, tokens, empty in con.execute(f"""
        SELECT source, count(*),
               median(CASE WHEN trim(content) = '' THEN 0
                           ELSE len(string_split_regex(trim(content), '\\s+')) END),
               count(*) FILTER (trim(content) = '')
        FROM '{OUT}' GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"{source} | {rows:,} | {tokens:g} | {empty:,}")
