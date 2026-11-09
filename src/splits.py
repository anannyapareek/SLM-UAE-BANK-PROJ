"""Four-way split: train, valid, calib and test, with a user hold-out and a test purge.

Label-side: reads labels and malicious events. Feature code never reads its outputs.
Run with: python run.py splits, then python run.py leakage. The leakage stage writes
results/leakage_report.md and, only if every check passes, freezes test_ids.csv.
Every later stage must call verify_test_checksum() first.
"""
import csv
import datetime
import hashlib
import os
import pathlib
import re
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import yaml

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
CONFIG = pathlib.Path("configs/splits.yaml")
FEATURES_CONFIG = pathlib.Path("configs/features.yaml")
TEST_IDS = P / "test_ids.csv"
LOG = pathlib.Path("decisions_log.md")
REPORT = pathlib.Path("results/leakage_report.md")
CHECKSUM = re.compile(r"test_ids\.csv: .*SHA-256 ([0-9a-f]{64})")
LABEL_COLUMNS = {"y", "scenario", "in_positive_session", "first_malicious_time", "incident_id",
                 "actor_user", "heldout_user", "incident_seen_before_test", "split"}
LABEL_SOURCES = {"labels", "malicious_events", "splits", "answers"}
RECOMPUTE_USERS, RECOMPUTE_ROWS, BASELINE_ROWS = 50, 500, 50
SPLITS = ["train", "valid", "calib", "test"]
EARLIER = SPLITS[:3]
SCENARIOS = [1, 2, 3]


def load_sessions(con):
    """One row per scoring session, with its label summary."""
    return con.execute(f"""
        SELECT l.session_id, any_value(l."user") AS "user", any_value(s.start) AS start,
               any_value(s."end") AS "end", count(*) AS rows, sum(l.y) AS positive_rows,
               max(l.in_positive_session) AS positive, any_value(l.scenario) AS scenario,
               any_value(l.incident_id) AS incident_id, any_value(l.actor_user) AS actor_user
        FROM '{P}/labels.parquet' l JOIN '{P}/sessions.parquet' s USING (session_id)
        GROUP BY l.session_id
    """).df()


def load_involved(con):
    """Every user named in each incident's malicious events."""
    return con.execute(f"""
        SELECT DISTINCT incident_id, "user" FROM '{P}/malicious_events.parquet'
    """).df()


def held_out_users(sessions, involved, seed, fraction, active_from=None):
    """Stratified by scenario: some insiders from each scenario, plus other users.

    An incident is held out whole: if its insider is held out, so is every user
    involved in it (in scenario 3, the supervisor). With active_from, insiders are drawn
    only from incidents with a positive session starting on or after that day.
    """
    rng = np.random.default_rng(seed)
    positive = sessions[sessions["positive"] == 1]
    incidents = positive.groupby("incident_id")[["actor_user", "scenario"]].first()
    candidates = incidents
    if active_from is not None:
        days = positive["start"].dt.normalize()
        active = positive.loc[days >= pd.Timestamp(active_from), "incident_id"]
        candidates = incidents[incidents.index.isin(active)]
    insiders = set(incidents["actor_user"])
    groups = [sorted(candidates.loc[candidates["scenario"] == s, "actor_user"]) for s in SCENARIOS]
    groups.append(sorted(set(sessions["user"]) - insiders - set(involved["user"])))
    sizes = [max(1, round(fraction * (incidents["scenario"] == s).sum())) for s in SCENARIOS]
    sizes.append(round(fraction * len(groups[-1])))
    picked = set()
    for group, size in zip(groups, sizes):
        picked |= set(rng.choice(group, size=min(size, len(group)), replace=False))
    held_incidents = set(incidents.index[incidents["actor_user"].isin(picked)])
    picked |= set(involved.loc[involved["incident_id"].isin(held_incidents), "user"])
    return picked, held_incidents


def period(times, boundaries):
    """Split containing each timestamp, or None outside every split."""
    out = pd.Series(None, index=times.index, dtype=object)
    for name, (first, last) in boundaries.items():
        inside = (times >= pd.Timestamp(first)) & (times < pd.Timestamp(last) + pd.Timedelta(days=1))
        out[inside] = name
    return out


def assign(sessions, involved, cfg):
    """Add split, status and incident flags to each session; return a summary dict."""
    boundaries, holdout = cfg["boundaries"], cfg["holdout"]
    s = sessions.copy()
    s["period"] = period(s["start"], boundaries)
    s["crosses"] = s["period"].isna() | (s["period"] != period(s["end"], boundaries))

    held, held_incidents = held_out_users(s, involved, holdout["seed"], holdout["fraction"],
                                          active_from=boundaries["test"][0])
    s["heldout_user"] = s["user"].isin(held)
    s["status"] = np.select(
        [s["crosses"], s["heldout_user"] & (s["period"] != "test")],
        ["dropped: crosses a boundary", "dropped: held-out user"], "kept")

    kept = s["status"] == "kept"
    positive = s["positive"] == 1
    test_incidents = set(s.loc[kept & positive & (s["period"] == "test")
                               & (s["positive_rows"] > 0), "incident_id"])
    earlier_positive = kept & positive & s["period"].isin(EARLIER)
    seen_before_test = set(s.loc[earlier_positive, "incident_id"]) & test_incidents
    before = {n: s.loc[kept & positive & (s["period"] == n), "incident_id"].nunique() for n in SPLITS}
    purge = earlier_positive & s["incident_id"].isin(test_incidents)
    s.loc[purge, "status"] = "dropped: test purge"
    s["split"] = s["period"].where(s["status"] == "kept")
    s["incident_seen_before_test"] = np.where(
        (s["split"] == "test") & positive, s["incident_id"].isin(seen_before_test), None)
    after = {n: s.loc[(s["split"] == n) & positive, "incident_id"].nunique() for n in SPLITS}
    return s, {"held": held, "held_incidents": held_incidents, "test_incidents": test_incidents,
               "seen_before_test": seen_before_test, "incidents_before": before,
               "incidents_after": after,
               "purged": s.loc[purge].groupby("period")["session_id"].count().to_dict()}


def text_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: " | ".join(str(x).ljust(w) for x, w in zip(r, widths))
    return "\n".join([line(header), "-+-".join("-" * w for w in widths)] + [line(r) for r in rows])


def split_tables(s):
    """(split table rows, per-scenario rows) for kept splits and each drop reason."""
    groups = SPLITS + sorted(x for x in s["status"].unique() if x != "kept")
    main, by_scenario = [], []
    for name in groups:
        part = s[s["split"] == name] if name in SPLITS else s[s["status"] == name]
        positive = part[part["positive"] == 1]
        scenarios = ",".join(str(int(x)) for x in sorted(positive["scenario"].dropna().unique()))
        main.append([name, f"{part['start'].min():%Y-%m-%d} to {part['start'].max():%Y-%m-%d}",
                     part["user"].nunique(), f"{len(part):,}", f"{int(part['rows'].sum()):,}",
                     len(positive), f"{int(positive['positive_rows'].sum()):,}",
                     positive["incident_id"].nunique(), positive["actor_user"].nunique(),
                     scenarios or "-"])
        by_scenario.append([name] + [
            f"{int((positive['scenario'] == sc).sum())} / "
            f"{int(positive.loc[positive['scenario'] == sc, 'positive_rows'].sum()):,} / "
            f"{positive.loc[positive['scenario'] == sc, 'incident_id'].nunique()}"
            for sc in SCENARIOS])
    return main, by_scenario


SPLIT_HEADER = ["split", "date range", "users", "sessions", "rows", "positive sessions",
                "positive rows", "incidents", "insiders", "scenarios present"]
SCENARIO_HEADER = ["split"] + [f"scenario {s} (sessions / rows / incidents)" for s in SCENARIOS]


def write_outputs(con, s, summary):
    keep = s.loc[s["status"] == "kept",
                 ["session_id", "split", "heldout_user", "incident_seen_before_test"]]
    con.register("assigned", keep)
    con.execute(f"""
        COPY (
            SELECT l.row_id, l.session_id, l."user", a.split, l.incident_id,
                   a.heldout_user, a.incident_seen_before_test::BOOLEAN AS incident_seen_before_test
            FROM '{P}/labels.parquet' l JOIN assigned a USING (session_id)
            ORDER BY l.session_id, l.score_time
        ) TO '{P}/splits.parquet' (FORMAT parquet)
    """)
    con.unregister("assigned")
    involved = load_involved(con)
    with (P / "heldout_users.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "incident_id"])
        for user in sorted(summary["held"]):
            incidents = sorted(involved.loc[(involved["user"] == user) & involved["incident_id"]
                                            .isin(summary["held_incidents"]), "incident_id"])
            for incident in incidents or [""]:
                writer.writerow([user, incident])


def build(con, cfg):
    sessions = load_sessions(con)
    s, summary = assign(sessions, load_involved(con), cfg)
    main, by_scenario = split_tables(s)
    print(text_table(SPLIT_HEADER, main))
    print("\npositives by scenario (positive sessions / positive rows / incidents)")
    print(text_table(SCENARIO_HEADER, by_scenario))
    crossing = s[s["status"] == "dropped: crosses a boundary"]
    print(f"\nsessions dropped for crossing a boundary: {len(crossing):,} "
          f"({int(crossing['rows'].sum()):,} rows, {int(crossing['positive'].sum())} positive)")
    print(f"positive sessions purged: {summary['purged']}")
    print(f"incidents per split before purge: {summary['incidents_before']}; "
          f"after: {summary['incidents_after']}")
    print(f"test incidents seen before test: {len(summary['seen_before_test'])} of "
          f"{len(summary['test_incidents'])}")
    return s, summary


# Freeze

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def logged_checksum():
    found = CHECKSUM.findall(LOG.read_text()) if LOG.exists() else []
    return found[-1] if found else None


def verify_test_checksum():
    """Raise unless test_ids.csv matches the SHA-256 last logged in decisions_log.md."""
    expected = logged_checksum()
    if expected is None or not TEST_IDS.exists():
        raise RuntimeError("test set is not frozen: no test_ids.csv or no logged SHA-256")
    actual = sha256(TEST_IDS)
    if actual != expected:
        raise RuntimeError(f"test_ids.csv SHA-256 {actual} does not match the logged {expected}")
    return actual


def git_commit():
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    return head + (" plus uncommitted changes" if dirty.strip() else "")


def freeze(con, force=False):
    """Write test_ids.csv (row_id, sorted) and log its SHA-256 with the date and commit."""
    if TEST_IDS.exists() and not force:
        raise RuntimeError(f"{TEST_IDS} already exists; pass --force-i-know to overwrite it")
    ids = [r[0] for r in con.execute(
        f"SELECT row_id FROM '{P}/splits.parquet' WHERE split = 'test' ORDER BY row_id").fetchall()]
    TEST_IDS.write_text("row_id\n" + "\n".join(ids) + "\n")
    digest = sha256(TEST_IDS)
    if digest == logged_checksum():
        return digest
    text = LOG.read_text()
    LOG.write_text(text + ("" if text.endswith("\n") else "\n") +
                   f"| {datetime.date.today()} | Frozen test_ids.csv: {len(ids):,} rows, "
                   f"SHA-256 {digest} | Computed at git commit {git_commit()} |\n")
    return digest


# Leakage report

def boundary_sql(column, boundaries):
    return "CASE " + " ".join(
        f"WHEN {column} >= TIMESTAMP '{a}' AND {column} < TIMESTAMP '{b}' + INTERVAL 1 DAY "
        f"THEN '{name}'" for name, (a, b) in boundaries.items()) + " END"


def recompute(con, cfg, seed):
    """Step 2 recomputation on RECOMPUTE_ROWS rows per split (from RECOMPUTE_USERS users)."""
    sys.path.insert(0, str(pathlib.Path("tests").resolve()))
    from test_leakage import Recomputer, values_match

    table = lambda name: f"read_parquet('{P}/{name}.parquet')"
    rng = np.random.default_rng(seed)
    results = {}
    for split in SPLITS:
        users = sorted(r[0] for r in con.execute(
            f"SELECT DISTINCT \"user\" FROM {table('splits')} WHERE split = '{split}'").fetchall())
        users = rng.choice(users, size=min(RECOMPUTE_USERS, len(users)), replace=False).tolist()
        names = ", ".join(f"'{u}'" for u in users)
        rows = con.execute(f"""
            SELECT f.*, s.start FROM {table('features')} f
            JOIN {table('splits')} x USING (row_id) JOIN {table('sessions')} s
              ON s.session_id = f.session_id
            WHERE x.split = '{split}' AND f."user" IN ({names}) ORDER BY f.row_id
        """).df()
        rows = rows.iloc[np.sort(rng.choice(len(rows), size=min(RECOMPUTE_ROWS, len(rows)),
                                            replace=False))]
        recomputer = Recomputer(cfg, con, table, users)
        wrong = sum(not values_match(recomputer.expected(f["name"], row), getattr(row, f["name"]))
                    for row in rows.itertuples() for f in cfg["features"])
        results[split] = (len(rows), len(rows) * len(cfg["features"]), wrong)
    return results


def leakage_checks(con, cfg, features_cfg, summary):
    b = cfg["boundaries"]
    sp = f"'{P}/splits.parquet'"
    q = lambda sql: con.execute(sql).fetchall()
    checks, info = [], []

    def check(label, ok, detail):
        checks.append((label, "PASS" if ok else "FAIL", detail))

    n = q(f"SELECT count(*) FROM {sp} WHERE heldout_user AND split <> 'test'")[0][0]
    check("(a) No held-out user has rows in train, valid or calib", n == 0, f"{n:,} rows")

    limits = {"train": f"score_time >= TIMESTAMP '{b['valid'][0]}'",
              "valid": f"score_time >= TIMESTAMP '{b['calib'][0]}'",
              "calib": f"score_time >= TIMESTAMP '{b['test'][0]}'",
              "test": f"score_time < TIMESTAMP '{b['test'][0]}'"}
    bad = {name: q(f"""SELECT count(*) FROM {sp} x JOIN '{P}/labels.parquet' l USING (row_id)
                       WHERE x.split = '{name}' AND l.{cond}""")[0][0] for name, cond in limits.items()}
    check("(b) Time order", not any(bad.values()),
          "; ".join(f"{k}: {v:,} rows out of range" for k, v in bad.items()))

    multi = q(f"""SELECT count(*) FROM (SELECT session_id FROM {sp} GROUP BY 1
                  HAVING count(DISTINCT split) > 1)""")[0][0]
    crossing = q(f"""SELECT count(*) FROM (SELECT DISTINCT x.session_id, x.split, s.start, s."end"
                     FROM {sp} x JOIN '{P}/sessions.parquet' s USING (session_id))
                     WHERE {boundary_sql('start', b)} IS DISTINCT FROM {boundary_sql('"end"', b)}
                        OR {boundary_sql('start', b)} IS DISTINCT FROM split""")[0][0]
    check("(c) Each session in one split; no kept session crosses a boundary",
          multi == 0 and crossing == 0,
          f"{multi:,} sessions in more than one split; {crossing:,} kept sessions crossing")

    names = {f["name"] for f in features_cfg["features"]}
    sources = {f["source"] for f in features_cfg["features"]}
    isolation = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         "tests/test_leakage.py", "-k", "answer_files or never_read_labels"],
        capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    label_named = sorted(names & LABEL_COLUMNS)
    label_sourced = sorted(sources & LABEL_SOURCES)
    check("(d) No label-derived features; isolation tests pass",
          not label_named and not label_sourced and isolation.returncode == 0,
          f"label columns as features: {label_named or 'none'}; label sources: "
          f"{label_sourced or 'none'}; isolation tests: "
          f"{isolation.stdout.strip().splitlines()[-1] if isolation.stdout.strip() else 'no output'}")

    seed = cfg["holdout"]["seed"]
    results = recompute(con, features_cfg, seed)
    wrong = sum(r[2] for r in results.values())
    check("(e) Features recomputed from events with ts <= score_time", wrong == 0,
          "; ".join(f"{k}: {r[0]} rows, {r[1]:,} values, {r[2]} mismatches"
                    for k, r in results.items()) +
          f" (rows drawn from {RECOMPUTE_USERS} random users per split)")

    positives = q(f"""
        SELECT x.split, count(DISTINCT x.session_id) FILTER (l.in_positive_session = 1),
               sum(l.y), count(DISTINCT l.scenario) FILTER (l.in_positive_session = 1)
        FROM {sp} x JOIN '{P}/labels.parquet' l USING (row_id) GROUP BY 1""")
    check("(f) Positives in every split, from every scenario",
          len(positives) == 4 and all(r[1] > 0 and r[3] == len(SCENARIOS) for r in positives),
          "; ".join(f"{r[0]}: {r[1]} positive sessions, {int(r[2]):,} positive rows, "
                    f"{r[3]} scenarios" for r in sorted(positives, key=lambda r: SPLITS.index(r[0]))))

    days = features_cfg["windows"]["baseline_days"]
    rng = np.random.default_rng(seed)
    sampled = []
    for split in SPLITS:
        ids = [r[0] for r in q(f"SELECT row_id FROM {sp} WHERE split = '{split}' ORDER BY row_id")]
        sampled += list(rng.choice(ids, size=min(BASELINE_ROWS, len(ids)), replace=False))
    con.execute("CREATE OR REPLACE TEMP TABLE sampled (row_id VARCHAR)")
    con.executemany("INSERT INTO sampled VALUES (?)", [(r,) for r in sampled])
    # The baseline set is defined by the same rule the features use; (e) confirms the
    # z-scores match it. This checks that rule never reaches past the scored session.
    late_end, early_end, after_score = q(f"""
        SELECT count(*) FILTER (p."end" > c.start),
               count(*) FILTER (p."end" < c.start - INTERVAL {days} DAY),
               count(*) FILTER (p."end" > l.score_time)
        FROM sampled JOIN '{P}/labels.parquet' l USING (row_id)
        JOIN '{P}/sessions.parquet' c ON c.session_id = l.session_id
        JOIN '{P}/sessions.parquet' p ON p."user" = c."user" AND p."end" < c.start
         AND p."end" >= c.start - INTERVAL {days} DAY""")[0]
    check(f"(g) Baseline is strictly past ({BASELINE_ROWS} rows per split)",
          late_end == 0 and early_end == 0 and after_score == 0,
          f"baseline sessions ending after the scored session's start: {late_end}; ending "
          f"more than {days} days before it: {early_end}; ending after score_time: {after_score}")
    same_day = q(f"""
        WITH k AS (SELECT DISTINCT session_id FROM {sp})
        SELECT count(DISTINCT c.session_id) FILTER (p."end"::DATE = c.start::DATE),
               count(*) FILTER (p."end"::DATE = c.start::DATE), count(*)
        FROM '{P}/sessions.parquet' c JOIN k USING (session_id)
        JOIN '{P}/sessions.parquet' p ON p."user" = c."user" AND p."end" < c.start
         AND p."end" >= c.start - INTERVAL {days} DAY""")[0]
    info.append(f"Baseline sessions from the same day: {same_day[0]:,} scored sessions; "
                f"{same_day[1]:,} of {same_day[2]:,} baseline links ({same_day[1] / same_day[2]:.1%})")

    leaked = q(f"""
        WITH t AS (SELECT DISTINCT l.incident_id FROM {sp} x JOIN '{P}/labels.parquet' l USING (row_id)
                   WHERE x.split = 'test' AND l.y = 1)
        SELECT count(DISTINCT l.incident_id) FROM {sp} x JOIN '{P}/labels.parquet' l USING (row_id)
        WHERE x.split <> 'test' AND l.y = 1 AND l.incident_id IN (SELECT incident_id FROM t)""")[0][0]
    check("(h) No incident with test positives has positive rows in another split", leaked == 0,
          f"{leaked} incidents; incidents per split before purge {summary['incidents_before']}, "
          f"after {summary['incidents_after']}")

    for a, c in (("train", "valid"), ("valid", "calib")):
        both = q(f"""SELECT count(*) FROM (SELECT l.incident_id FROM {sp} x
                     JOIN '{P}/labels.parquet' l USING (row_id) WHERE l.in_positive_session = 1
                     AND x.split IN ('{a}', '{c}') GROUP BY 1 HAVING count(DISTINCT x.split) = 2)""")[0][0]
        info.append(f"Accepted incident overlap {a} to {c}: {both} incidents")
    s = summary["sessions"]
    for reason in ("dropped: crosses a boundary", "dropped: held-out user", "dropped: test purge"):
        part = s[s["status"] == reason]
        info.append(f"{reason[9:].capitalize()}: {len(part):,} sessions, {int(part['rows'].sum()):,} "
                    f"rows, {int(part['positive'].sum())} positive sessions")
    return checks, info


def write_report(checks, info, main, by_scenario):
    def md(header, rows):
        return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
               ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    lines = ["# Leakage report", "", "Generated by `python run.py leakage`.", "",
             "## Checks", ""] + md(["Check", "Result", "Details"], checks)
    lines += ["", "## Information", ""] + [f"- {x}" for x in info]
    lines += ["", "## Splits", ""] + md(SPLIT_HEADER, main)
    lines += ["", "## Positives by scenario", ""] + md(SCENARIO_HEADER, by_scenario)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")


# Stages

def run(force=False):
    cfg = yaml.safe_load(CONFIG.read_text())
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    s, summary = build(con, cfg)
    if TEST_IDS.exists() and not force:
        verify_test_checksum()
        frozen = set(TEST_IDS.read_text().split()[1:])
        if frozen != set(s.loc[s["split"] == "test", "session_id"].pipe(lambda x: con.execute(
                f"SELECT row_id FROM '{P}/labels.parquet' WHERE session_id IN "
                f"({','.join(map(str, x)) or 'NULL'})").df()["row_id"])):
            raise RuntimeError("the rebuilt test set differs from the frozen test_ids.csv; "
                               "pass --force-i-know to replace it")
    write_outputs(con, s, summary)


def run_leakage(force=False):
    cfg = yaml.safe_load(CONFIG.read_text())
    features_cfg = yaml.safe_load(FEATURES_CONFIG.read_text())
    con = duckdb.connect(config={"threads": 1})
    if TEST_IDS.exists() and not force:
        verify_test_checksum()
    sessions = load_sessions(con)
    s, summary = assign(sessions, load_involved(con), cfg)
    summary["sessions"] = s
    main, by_scenario = split_tables(s)
    checks, info = leakage_checks(con, cfg, features_cfg, summary)
    write_report(checks, info, main, by_scenario)
    for label, result, detail in checks:
        print(f"{result}  {label}: {detail}")
    for line in info:
        print(f"INFO  {line}")
    failed = [c[0] for c in checks if c[1] == "FAIL"]
    if failed:
        raise SystemExit(f"leakage checks failed: {', '.join(failed)}; test set not frozen")
    if not TEST_IDS.exists() or force:
        print(f"froze {TEST_IDS}: SHA-256 {freeze(con, force)}")
    else:
        print(f"test set already frozen: SHA-256 {verify_test_checksum()}")
