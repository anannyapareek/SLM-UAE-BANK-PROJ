"""Summary of the processed tables, written to docs/prep_output.txt by make report-prep.

It reads the ids and content themselves, because count(*) alone comes from the Parquet
footer and cannot see damaged pages.
"""
import os
import pathlib

import duckdb

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))


def run():
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    for t in ["logon", "device", "file", "email", "http"]:
        n, lo, hi = con.execute(
            f"SELECT count(*), min(ts), max(ts) FROM '{P}/{t}.parquet'").fetchone()
        print(f"{t}: {n:,} rows, {lo} to {hi}")

    for t in ["file", "email", "http"]:
        n, n_id, n_content_chars = con.execute(
            f"SELECT count(*), count(id), sum(length(content)) FROM '{P}/{t}_content.parquet'"
        ).fetchone()
        missing = con.execute(
            f"SELECT count(*) FROM '{P}/{t}_content.parquet' c "
            f"ANTI JOIN '{P}/{t}.parquet' m USING (id)").fetchone()[0]
        print(f"{t}_content: {n:,} rows, {n_id:,} ids read, {n_content_chars:,} chars, "
              f"{missing:,} ids not in {t}")

    rows, users = con.execute(
        f"SELECT count(*), count(DISTINCT user_id) FROM '{P}/ldap.parquet'").fetchone()
    print(f"ldap: {rows:,} rows, {users:,} distinct users")
