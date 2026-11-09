"""Leakage and integrity checks on the feature and text tables.

The recomputation here deliberately does not import src.features: it rebuilds every
feature from the processed event tables with separate Python logic.
"""
import os
import re
import subprocess
import sys
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import ROOT

META_COLUMNS = ["row_id", "session_id", "user", "pc", "score_time", "is_final"]
FORBIDDEN_COLUMNS = {"end", "end_type", "start"}
LDAP_FEATURES = ["role", "department", "functional_unit"]
SAMPLE_USERS = 15
SAMPLE_ROWS = 600
LATE_HOUR = 8
# Label-side code: may name the answers and read label outputs; feature code may not.
ANSWERS_ALLOWED = {"src/check_orphans.py", "src/labels.py", "src/explore_positives.py",
                   "src/splits.py", "configs/splits.yaml", "configs/rules.yaml",
                   "src/baselines/rules.py", "src/eval.py", "configs/eval.yaml"}
# Model stages read labels to train and score; they are not feature code.
MODEL_STAGES = {"src/baselines/xgb.py", "src/baselines/rules.py", "src/eval.py"}
# The frozen test row ids: row_id hashes only, no CERT data.
TRACKED_DATA_ALLOWED = {"data/processed/test_ids.csv"}
LABEL_OUTPUTS = re.compile(r"labels\.parquet|malicious_events|labels\.run|from src import labels|src\.labels")
# Aggregate counts only, checked below for user IDs.
ANSWER_DERIVED_ALLOWED = {"docs/orphan_check.txt"}
USER_ID = re.compile(r"\b[A-Z]{3}[0-9]{4}\b")
ANSWER_WORDS = re.compile(r"answers|insiders?|scenarios?", re.IGNORECASE)
HEX_HEADER = r"^[0-9A-Fa-f]{2}(-[0-9A-Fa-f]{2})+(\s|$)"


# 1. Independent recomputation

class Events:
    """Weighted events per key, sorted by time, for counting over a time range."""

    def __init__(self, frame, keys, weight=None):
        frame = frame.assign(weight=1 if weight is None else weight)
        self.times, self.totals = {}, {}
        for key, group in frame.sort_values("ts").groupby(keys, sort=False):
            self.times[key] = group["ts"].to_numpy("datetime64[ns]")
            self.totals[key] = np.concatenate([[0], group["weight"].cumsum().to_numpy()])

    def total(self, key, start, stop, after=None):
        """Sum of weights with start <= ts <= stop, and ts > after if given."""
        if key not in self.times:
            return 0
        times = self.times[key]
        first = np.searchsorted(times, np.datetime64(start), "left")
        if after is not None:
            first = max(first, np.searchsorted(times, np.datetime64(after), "right"))
        last = np.searchsorted(times, np.datetime64(stop), "right")
        return int(self.totals[key][last] - self.totals[key][first]) if last > first else 0


def within_session(frame, sessions):
    """Boolean mask of events inside any session for the same user and PC."""
    spans = defaultdict(list)
    for s in sessions.itertuples():
        spans[(s.user, s.pc)].append((s.start, s.end))
    return np.array([any(a <= t <= b for a, b in spans[(u, p)])
                     for u, p, t in zip(frame["user"], frame["pc"], frame["ts"])], dtype=bool)


def clock(text):
    return pd.Timestamp(f"2000-01-01 {text}").time()


class Recomputer:
    def __init__(self, cfg, con, table, users):
        self.cfg = cfg
        self.interval = pd.Timedelta(minutes=cfg["scoring"]["interval_minutes"])
        self.window = pd.Timedelta(minutes=cfg["windows"]["rolling_window_minutes"])
        self.work_start = clock(cfg["off_hours"]["work_start"])
        self.work_end = clock(cfg["off_hours"]["work_end"])

        names = ", ".join(f"'{u}'" for u in users)

        def load(name, columns, column='"user"'):
            return con.execute(
                f"SELECT {columns} FROM {table(name)} WHERE {column} IN ({names})").df()

        self.sessions = load("sessions", '"user", pc, start, "end"')
        logon = load("logon", '"user", pc, ts, activity')
        device = load("device", '"user", pc, ts, activity')
        files = load("file", 'id, "user", pc, ts, filename').merge(
            con.execute(f"""SELECT id, content FROM {table('file_content')}
                            WHERE id IN (SELECT id FROM {table('file')}
                                         WHERE "user" IN ({names}))""").df(), on="id")
        email = load("email", '"user", pc, ts, "to", cc, bcc, size, attachments')
        http = load("http", '"user", pc, ts, url')
        self.ldap = load("ldap", "user_id, month_file, role, department, functional_unit",
                         "user_id")
        self.ldap["month"] = pd.to_datetime(
            self.ldap["month_file"].str.extract(r"(\d{4}-\d{2})\.csv$")[0])

        logons = logon[logon["activity"] == "Logon"]
        connects = device[device["activity"] == "Connect"]
        self.logon_first = logons.groupby(["user", "pc"])["ts"].min().to_dict()

        keys = ["user", "pc"]
        self.metrics = {"unlocks_so_far": Events(logons, keys),
                        "usb_connects": Events(connects, keys),
                        "file_events": Events(files, keys)}
        self.today = {
            "logon_events_today_so_far": Events(logons, "user"),
            "usb_connects_today_so_far": Events(
                connects[within_session(connects, self.sessions)], "user"),
        }

        extension = files["filename"].str.rsplit(".", n=1).str[1].str.lower()
        for name in cfg["file_extensions"]:
            self.metrics[f"file_ext_{name}"] = Events(files[extension == name], keys)
        rules = cfg["file_header_extensions"]
        implied = files["content"].str.split().str[0].map(
            lambda h: rules["headers"].get(h, rules["default"]))
        self.metrics["file_header_mismatch"] = Events(files[implied != extension], keys)

        domain = "@" + cfg["internal_email_domain"]

        def external(row):
            everyone = {a.strip().lower() for field in (row["to"], row["cc"], row["bcc"])
                        if isinstance(field, str) for a in field.split(";") if a.strip()}
            return sum(not a.endswith(domain) for a in everyone)

        after_hours = email["ts"].dt.time.map(self.outside_hours)
        self.metrics |= {
            "emails_sent": Events(email, keys),
            "external_recipients": Events(email, keys, email.apply(external, axis=1)),
            "attachments_count": Events(email, keys, email["attachments"].astype(int)),
            "email_bytes": Events(email, keys, email["size"].astype("int64")),
            "after_hours_emails": Events(email[after_hours], keys),
            "http_visits": Events(http, keys),
        }

        host = http["url"].str.split("/").str[2].str.lower()
        for category, members in cfg["http_keyword_lists"][cfg["http_lists"]].items():
            self.metrics[f"http_{category}"] = Events(http[host.isin(members)], keys)

    def outside_hours(self, time):
        return not self.work_start <= time < self.work_end

    def cumulative(self, name, row):
        key = (row.user, row.pc)
        if name == "unlocks_so_far":   # Logons after the one that started the session
            return self.metrics[name].total(key, row.start, row.score_time, after=row.start)
        return self.metrics[name].total(key, row.start, row.score_time)

    def rolling(self, name, row):
        return self.metrics[name].total((row.user, row.pc), row.start, row.score_time,
                                        after=row.score_time - self.window)

    def elapsed_units(self, row):
        return max(int((row.score_time - row.start) / self.interval), 1)

    def z_score(self, name, row):
        windows = self.cfg["windows"]
        earliest = row.start - pd.Timedelta(days=windows["baseline_days"])
        hours = self.elapsed_units(row)
        sessions = self.sessions
        prior = sessions[(sessions["user"] == row.user) & (sessions["end"] < row.start)
                         & (sessions["end"] >= earliest)]
        if len(prior) < windows["baseline_min_sessions"]:
            return np.nan
        values = [self.metrics[name].total((p.user, p.pc), p.start,
                                           min(p.start + hours * self.interval, p.end))
                  for p in prior.itertuples()]
        spread = max(np.std(values, ddof=1), windows["std_floor"])
        return (self.cumulative(name, row) - np.mean(values)) / spread

    def ldap_value(self, name, row):
        month = row.score_time.to_period("M").to_timestamp()
        rows = self.ldap[(self.ldap["user_id"] == row.user) & (self.ldap["month"] < month)]
        value = rows.sort_values("month").iloc[-1][name] if len(rows) else None
        return self.cfg["fairness"]["missing_value"] if pd.isna(value) else value

    def expected(self, name, row):
        if name in LDAP_FEATURES:
            return self.ldap_value(name, row)
        if name == "session_minutes_so_far":
            return self.elapsed_units(row) * self.interval.total_seconds() / 60
        if name == "off_hours":
            return int(self.outside_hours(row.start.time()))
        if name == "weekend":
            return int(row.start.strftime("%a") in self.cfg["off_hours"]["weekend_days"])
        if name == "new_pc":
            return int(self.logon_first[(row.user, row.pc)] >= row.start)
        if name in self.today:
            return self.today[name].total(row.user, row.score_time.normalize(), row.score_time)
        if name.endswith("_z"):
            return self.z_score(name.removesuffix("_z"), row)
        if name.endswith("_1h"):
            return self.rolling(name.removesuffix("_1h"), row)
        return self.cumulative(name, row)


def values_match(expected, actual):
    if isinstance(expected, str):
        return expected == actual
    if isinstance(expected, (int, np.integer)):
        return int(expected) == actual
    return bool(np.isclose(expected, actual, rtol=1e-9, atol=1e-9, equal_nan=True))


@pytest.fixture(scope="module")
def sampled_rows(con, table, cfg, seed):
    """Rows from a seeded set of users, weighted towards non-final rows and late hours."""
    rng = np.random.default_rng(seed)
    everyone = sorted(r[0] for r in con.execute(
        f'SELECT DISTINCT "user" FROM {table("features")}').fetchall())
    users = rng.choice(everyone, size=min(SAMPLE_USERS, len(everyone)), replace=False).tolist()
    names = ", ".join(f"'{u}'" for u in users)
    rows = con.execute(f"""
        SELECT f.*, s.start
        FROM {table('features')} f JOIN {table('sessions')} s USING (session_id)
        WHERE f."user" IN ({names})
        ORDER BY f.row_id
    """).df()
    hours = rows["session_minutes_so_far"] / cfg["scoring"]["interval_minutes"]
    weight = 1 + 2 * (~rows["is_final"]) + 3 * (hours >= LATE_HOUR)
    picked = rng.choice(len(rows), size=min(SAMPLE_ROWS, len(rows)), replace=False,
                        p=(weight / weight.sum()).to_numpy())
    return users, rows.iloc[np.sort(picked)]


def test_sample_covers_the_hard_rows(sampled_rows, cfg):
    _, rows = sampled_rows
    assert (~rows["is_final"]).sum() > 0 and rows["is_final"].sum() > 0
    assert (rows["session_minutes_so_far"] >= LATE_HOUR * cfg["scoring"]["interval_minutes"]).any()


def test_independent_recomputation(sampled_rows, con, table, cfg):
    users, rows = sampled_rows
    recomputer = Recomputer(cfg, con, table, users)
    failures = []
    for row in rows.itertuples():
        for feature in cfg["features"]:
            name = feature["name"]
            expected, actual = recomputer.expected(name, row), getattr(row, name)
            if not values_match(expected, actual):
                failures.append(f"row_id={row.row_id} feature={name} "
                                f"expected={expected} actual={actual}")
    assert not failures, f"{len(failures)} mismatches:\n" + "\n".join(failures[:40])


# 2. Structural

def test_columns_follow_config(con, table, cfg):
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table('features')}").fetchall()]
    assert columns == META_COLUMNS + [f["name"] for f in cfg["features"]]
    assert not FORBIDDEN_COLUMNS & set(columns)


def test_row_id_unique(con, table):
    rows, ids = con.execute(
        f"SELECT count(*), count(DISTINCT row_id) FROM {table('features')}").fetchone()
    assert rows == ids


def test_score_times(con, table, cfg):
    seconds = cfg["scoring"]["interval_minutes"] * 60
    after_end, off_grid, wrong_final = con.execute(f"""
        SELECT count(*) FILTER (f.score_time > s."end"),
               count(*) FILTER (NOT f.is_final AND (
                   date_diff('second', s.start, f.score_time) % {seconds} <> 0
                   OR f.score_time <= s.start OR f.score_time >= s."end")),
               count(*) FILTER (f.is_final AND f.score_time <> s."end")
        FROM {table('features')} f JOIN {table('sessions')} s USING (session_id)
    """).fetchone()
    assert (after_end, off_grid, wrong_final) == (0, 0, 0)


def test_ldap_values_come_from_an_earlier_month(con, table, cfg):
    missing = cfg["fairness"]["missing_value"]
    no_snapshot, wrong = con.execute(f"""
        WITH used AS (
            SELECT DISTINCT "user", date_trunc('month', score_time) AS month,
                   role, department, functional_unit
            FROM {table('features')}
        ),
        snapshots AS (
            SELECT user_id, role, department, functional_unit,
                   strptime(substr(month_file, -11, 7), '%Y-%m') AS month
            FROM {table('ldap')}
        ),
        expected AS (
            SELECT u."user", u.month, max(s.month) AS snapshot
            FROM used u LEFT JOIN snapshots s ON s.user_id = u."user" AND s.month < u.month
            GROUP BY 1, 2
        )
        SELECT count(*) FILTER (e.snapshot IS NULL),
               count(*) FILTER (e.snapshot IS NOT NULL AND (
                   u.role <> coalesce(s.role, '{missing}')
                   OR u.department <> coalesce(s.department, '{missing}')
                   OR u.functional_unit <> coalesce(s.functional_unit, '{missing}')))
        FROM used u
        JOIN expected e ON e."user" = u."user" AND e.month = u.month
        LEFT JOIN snapshots s ON s.user_id = u."user" AND s.month = e.snapshot
    """).fetchone()
    assert (no_snapshot, wrong) == (0, 0)


def test_ldap_join_skips_the_current_month(tmp_path, monkeypatch, cfg):
    """LDAP values never change in r4.2, so this needs a table where they do."""
    from src import features

    con = duckdb.connect(config={"threads": 1})
    con.execute(f"""
        COPY (SELECT * FROM (VALUES
            ('u', 'x/2010-01.csv', 'january', 'd1', 'f1'),
            ('u', 'x/2010-02.csv', 'february', 'd2', 'f2'),
            ('u', 'x/2010-03.csv', 'march', 'd3', NULL)
        ) t(user_id, month_file, role, department, functional_unit))
        TO '{tmp_path}/ldap.parquet' (FORMAT parquet)
    """)
    con.execute("""
        CREATE TEMP TABLE grid AS SELECT * FROM (VALUES
            ('first_of_march', 'u', TIMESTAMP '2010-03-01 00:00:00'),
            ('end_of_march', 'u', TIMESTAMP '2010-03-31 23:59:59'),
            ('april', 'u', TIMESTAMP '2010-04-02 09:00:00')
        ) t(row_id, "user", score_time)
    """)
    monkeypatch.setattr(features, "P", tmp_path)
    sql = features.ldap_sql(con, cfg, [])
    found = dict(con.execute("SELECT row_id, role FROM ldap_rows").fetchall())
    assert found == {"first_of_march": "february", "end_of_march": "february", "april": "march"}
    unit = con.execute(
        f"SELECT {sql['functional_unit']} FROM ldap_rows WHERE row_id = 'april'").fetchone()[0]
    assert unit == cfg["fairness"]["missing_value"]


# 3. Isolation

def text_files(*folders):
    for folder in folders:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
                yield path


def tracked_files():
    return subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.splitlines()


def test_only_label_code_mentions_the_answer_files():
    mentions = sorted(
        str(path.relative_to(ROOT)) for path in text_files("src", "configs")
        if ANSWER_WORDS.search(path.read_text(errors="ignore")))
    assert set(mentions) <= ANSWERS_ALLOWED, mentions


def test_features_never_read_labels():
    readers = sorted(
        str(path.relative_to(ROOT)) for path in text_files("src", "configs")
        if str(path.relative_to(ROOT)) not in ANSWERS_ALLOWED | MODEL_STAGES | {"src/run.py"}
        and LABEL_OUTPUTS.search(path.read_text(errors="ignore")))
    assert not readers, readers


def test_no_answer_derived_file_is_tracked():
    tracked = tracked_files()
    assert not [f for f in tracked if (f.startswith("data/") or f.endswith(".parquet"))
                and f not in TRACKED_DATA_ALLOWED]
    named = [f for f in tracked if re.search(r"insider|answers|r4\.2-\d", f, re.IGNORECASE)]
    assert not named
    with_ids = [f for f in tracked if (ROOT / f).is_file()
                and USER_ID.search((ROOT / f).read_text(errors="ignore"))]
    assert not with_ids
    derived = [f for f in tracked if f.startswith("docs/") and "orphan" in f]
    assert set(derived) <= ANSWER_DERIVED_ALLOWED


# 4. Burn-in

def test_no_rows_in_burn_in(con, table, cfg):
    burn_in = cfg["burn_in"]
    first_event = con.execute("SELECT min(t) FROM (" + " UNION ALL ".join(
        f"SELECT min(ts) AS t FROM {table(t)}" for t in burn_in["tables"]) + ")").fetchone()[0]
    cutoff = first_event + pd.Timedelta(days=burn_in["days"])
    earliest = con.execute(f"SELECT min(score_time) FROM {table('features')}").fetchone()[0]
    assert earliest >= cutoff


def test_every_row_has_a_baseline(con, table, cfg):
    days, minimum = cfg["windows"]["baseline_days"], cfg["windows"]["baseline_min_sessions"]
    short = con.execute(f"""
        WITH kept AS (SELECT DISTINCT session_id FROM {table('features')})
        SELECT count(*) FROM (
            SELECT c.session_id, count(p.session_id) AS earlier
            FROM {table('sessions')} c
            JOIN kept USING (session_id)
            LEFT JOIN {table('sessions')} p
              ON p."user" = c."user" AND p."end" < c.start
             AND p."end" >= c.start - INTERVAL {days} DAY
            GROUP BY 1
        ) WHERE earlier < {minimum}
    """).fetchone()[0]
    assert short == 0


# 5. Text table

def test_text_events_fall_inside_their_session(con, table):
    outside = con.execute(f"""
        SELECT count(*)
        FROM {table('features_text')} t LEFT JOIN {table('sessions')} s USING (session_id)
        WHERE s.session_id IS NULL OR t."user" <> s."user" OR t.pc <> s.pc
           OR t.ts < s.start OR t.ts > s."end"
    """).fetchone()[0]
    assert outside == 0


def test_text_sessions_are_in_features(con, table):
    extra = con.execute(f"""
        SELECT count(*) FROM (SELECT DISTINCT session_id FROM {table('features_text')})
        WHERE session_id NOT IN (SELECT session_id FROM {table('features')})
    """).fetchone()[0]
    assert extra == 0


def test_text_sort_order(con, processed):
    out_of_order = con.execute(f"""
        SELECT count(*) FROM (
            SELECT session_id, ts, event_id,
                   lag(session_id) OVER w AS s, lag(ts) OVER w AS t, lag(event_id) OVER w AS e
            FROM read_parquet('{processed}/features_text.parquet', file_row_number = true)
            WINDOW w AS (ORDER BY file_row_number)
        ) WHERE s > session_id OR (s = session_id AND t > ts)
             OR (s = session_id AND t = ts AND e > event_id)
    """).fetchone()[0]
    assert out_of_order == 0


def test_file_headers_are_stripped(con, table, cfg):
    if not cfg["text_rules"]["strip_file_header"]:
        pytest.skip("strip_file_header is off")
    headers = con.execute(f"""
        SELECT count(*) FROM {table('features_text')}
        WHERE source = 'file' AND regexp_matches(content, '{HEX_HEADER}')
    """).fetchone()[0]
    assert headers == 0


# 6. Switch isolation

@pytest.mark.slow
def test_top300_changes_only_list_based_columns(tmp_path, processed, is_full_data, con, cfg):
    lists = cfg["http_keyword_lists"]
    changed_lists = [c for c in lists["full"] if set(lists["full"][c]) != set(lists["top300"][c])]
    names = {f["name"] for f in cfg["features"]}
    list_columns = {n for c in changed_lists
                    for n in (f"http_{c}", f"http_{c}_1h", f"http_{c}_z") if n in names}
    assert cfg["http_lists"] == "full" and len(list_columns) == 7

    variant = dict(cfg, http_lists="top300")
    (tmp_path / "features.yaml").write_text(yaml.safe_dump(variant))
    env = dict(os.environ, PROCESSED_DIR=str(processed),
               FEATURES_CONFIG=str(tmp_path / "features.yaml"), FEATURES_OUT_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "src.run", "features"], cwd=ROOT, env=env, check=True,
                   capture_output=True)

    full = f"read_parquet('{processed}/features.parquet')"
    top300 = f"read_parquet('{tmp_path}/features.parquet')"
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {full}").fetchall()]
    compare = [c for c in columns if c != "row_id"]
    counts = ", ".join(f'count(*) FILTER (a."{c}" IS DISTINCT FROM b."{c}")' for c in compare)
    differences = con.execute(
        f"SELECT count(*), {counts} FROM {full} a JOIN {top300} b USING (row_id)").fetchone()
    rows = con.execute(f"SELECT count(*) FROM {full}").fetchone()[0]
    assert differences[0] == rows
    changed = {c for c, n in zip(compare, differences[1:]) if n}
    assert changed <= list_columns
    if is_full_data:
        assert changed == list_columns
