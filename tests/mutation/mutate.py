"""Check that the test suite catches deliberate faults.

Each fault is applied to a temporary copy of the sample build, and the fast tests must
then fail. Nothing under data/processed is written. Run with make test-mutation.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
SAMPLE = PROCESSED / "sample_20"
EXPECTED_FAULTS = 19
FEATURES, TEXT = "features", "features_text"


def faults(base):
    """Fault name: (table to replace, SQL for the broken table)."""
    f = f"read_parquet('{base}/features.parquet')"
    t = f"read_parquet('{base}/features_text.parquet')"
    sessions = f"read_parquet('{base}/sessions.parquet')"
    columns = [r[0] for r in duckdb.connect().execute(f"DESCRIBE SELECT * FROM {f}").fetchall()]
    first_event = f"(SELECT min(event_id) FROM {t})"

    def replace(column, expression):
        return f"SELECT * REPLACE ({expression} AS {column}) FROM {f}"

    def whole_session(column):
        return (f"SELECT * REPLACE (max({column}) OVER (PARTITION BY session_id) AS {column}) "
                f"FROM {f}")

    return {
        "future leak: usb_connects uses whole session": (FEATURES, whole_session("usb_connects")),
        "future leak: http_visits uses whole session": (FEATURES, whole_session("http_visits")),
        "exact minutes reveal final row": (FEATURES, f"""
            SELECT f.* REPLACE (date_diff('minute', s.start, f.score_time)
                                AS session_minutes_so_far)
            FROM {f} f JOIN {sessions} s USING (session_id)"""),
        "email_bytes off by one on 1% of rows": (
            FEATURES, replace("email_bytes", "email_bytes + (hash(row_id) % 100 = 0)::INT")),
        "z-score nudged by 0.001": (FEATURES, replace("emails_sent_z", "emails_sent_z + 0.001")),
        "wrong LDAP role": (FEATURES, replace("role", "'X'")),
        "file_header_mismatch zeroed": (FEATURES, replace("file_header_mismatch", "0")),
        "rolling equals cumulative": (FEATURES, replace("file_events_1h", "file_events")),
        "extra end column": (FEATURES, f"""
            SELECT f.*, s."end" FROM {f} f JOIN {sessions} s USING (session_id)"""),
        "column order swapped": (FEATURES, f"""
            SELECT {', '.join(f'"{c}"' for c in columns[:6] + columns[:5:-1])} FROM {f}"""),
        "duplicate row": (FEATURES, f"SELECT * FROM {f} UNION ALL (SELECT * FROM {f} LIMIT 1)"),
        "score_time one minute off the hour": (FEATURES, replace("score_time", """
            CASE WHEN NOT is_final THEN score_time + INTERVAL 1 MINUTE ELSE score_time END""")),
        "score_time past session end": (FEATURES, replace("score_time", """
            CASE WHEN is_final THEN score_time + INTERVAL 1 HOUR ELSE score_time END""")),
        "row inside burn-in": (FEATURES, replace("score_time", f"""
            CASE WHEN session_id = (SELECT min(session_id) FROM {f})
                 THEN TIMESTAMP '2010-01-05 09:00:00' ELSE score_time END""")),
        "text event outside session": (TEXT, f"""
            SELECT * REPLACE (CASE WHEN event_id = {first_event} THEN ts + INTERVAL 30 DAY
                                   ELSE ts END AS ts) FROM {t}"""),
        "text session not in features": (TEXT, f"""
            SELECT * REPLACE (CASE WHEN event_id = {first_event} THEN -1
                                   ELSE session_id END AS session_id) FROM {t}"""),
        "text unsorted": (TEXT, f"SELECT * FROM {t} ORDER BY event_id"),
        "text header left in": (TEXT, f"""
            SELECT * REPLACE (CASE WHEN source = 'file'
                                   THEN 'D0-CF-11-E0-A1-B1-1A-E1 ' || content
                                   ELSE content END AS content) FROM {t}"""),
    }


def failed_tests(repo, processed):
    env = dict(os.environ, PROCESSED_DIR=str(processed), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not slow", "-q", "--no-header", "--tb=no",
         "-p", "no:cacheprovider"], cwd=repo, env=env, capture_output=True, text=True)
    return sorted(set(re.findall(r"(?:FAILED|ERROR) tests/\S+::(\S+)", result.stdout)))


def report(name, failed):
    print(f"{'CAUGHT' if failed else 'MISSED'} {name}" + (f": {', '.join(failed)}" if failed else ""),
          flush=True)
    return bool(failed)


def snapshot():
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in PROCESSED.rglob("*") if p.is_file()}


def main():
    before = snapshot()
    caught = 0
    with tempfile.TemporaryDirectory() as temp:
        temp = pathlib.Path(temp)
        base = temp / "base"
        shutil.copytree(SAMPLE, base)

        if failed_tests(ROOT, base):
            sys.exit("the unbroken copy already fails, so faults would prove nothing")

        broken = temp / "broken"
        all_faults = faults(base)
        for name, (table, sql) in all_faults.items():
            broken.mkdir()
            for path in base.iterdir():
                if path.name != f"{table}.parquet":
                    os.symlink(path, broken / path.name)
            con = duckdb.connect(config={"threads": 1})
            con.execute(f"COPY ({sql}) TO '{broken}/{table}.parquet' (FORMAT parquet)")
            con.close()
            caught += report(name, failed_tests(ROOT, broken))
            shutil.rmtree(broken)

        # The isolation test reads the repo, so this fault goes in a copy of it.
        repo = temp / "repo"
        shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(
            "data", ".venv", "__pycache__", ".pytest_cache", "results"))
        (repo / "src/fault.py").write_text("# reads the insiders file\n")
        caught += report("src file mentions the answer files", failed_tests(repo, base))
        total = len(all_faults) + 1

    print(f"{caught} of {total} faults caught")
    if snapshot() != before:
        sys.exit("data/processed changed during the run")
    if caught != total or total != EXPECTED_FAULTS:
        sys.exit(f"expected {EXPECTED_FAULTS} faults caught")


if __name__ == "__main__":
    main()
