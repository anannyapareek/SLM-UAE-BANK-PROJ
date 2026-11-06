"""Small copy of data/processed for a few users, used to test the pipeline quickly."""
import pathlib
import random

import duckdb
import yaml

SOURCE = pathlib.Path("data/processed")
GLOBAL_CONFIG = pathlib.Path("configs/global.yaml")
EVENT_TABLES = ["logon", "device", "file", "email", "http"]
USER_TABLES = (["sessions"] + EVENT_TABLES
               + [f"{t}_sessioned" for t in EVENT_TABLES])
CONTENT_TABLES = ["file", "email", "http"]


def source_files():
    names = USER_TABLES + ["ldap"] + [f"{t}_content" for t in CONTENT_TABLES]
    return [SOURCE / f"{name}.parquet" for name in names]


def build(users):
    """Return the sample folder, rebuilding it if the source tables have changed."""
    out = SOURCE / f"sample_{users}"
    marker = out / ".complete"
    if marker.exists() and all(
            f.stat().st_mtime < marker.stat().st_mtime for f in source_files()):
        return out

    out.mkdir(exist_ok=True)
    marker.unlink(missing_ok=True)
    seed = yaml.safe_load(GLOBAL_CONFIG.read_text())["seed"]
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    con.execute(f"SET temp_directory = '{SOURCE}/.duckdb_tmp'")
    everyone = [r[0] for r in con.execute(
        f"""SELECT DISTINCT "user" FROM '{SOURCE}/sessions.parquet' ORDER BY 1""").fetchall()]
    picked = random.Random(seed).sample(everyone, users)
    con.execute("CREATE TEMP TABLE picked (name VARCHAR)")
    con.executemany("INSERT INTO picked VALUES (?)", [(u,) for u in picked])

    for table in USER_TABLES:
        con.execute(f"""
            COPY (SELECT * FROM '{SOURCE}/{table}.parquet'
                  WHERE "user" IN (SELECT name FROM picked))
            TO '{out}/{table}.parquet' (FORMAT parquet)
        """)
    con.execute(f"""
        COPY (SELECT * FROM '{SOURCE}/ldap.parquet'
              WHERE user_id IN (SELECT name FROM picked))
        TO '{out}/ldap.parquet' (FORMAT parquet)
    """)
    for table in CONTENT_TABLES:
        con.execute(f"""
            COPY (SELECT * FROM '{SOURCE}/{table}_content.parquet'
                  WHERE id IN (SELECT id FROM '{out}/{table}.parquet'))
            TO '{out}/{table}_content.parquet' (FORMAT parquet)
        """)
    marker.touch()
    return out
