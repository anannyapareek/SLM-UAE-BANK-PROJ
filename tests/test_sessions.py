"""Session rules on synthetic logon sequences."""
import duckdb

from src.sessions import assign_sql, build_sessions

DAY1 = "2010-01-01 "
DAY2 = "2010-01-02 "
LOGONS = [
    ("a", DAY1 + "08:00:00", "Logon"), ("a", DAY1 + "09:00:00", "Logoff"),   # plain session
    ("a", DAY1 + "09:30:00", "Logoff"),                                      # orphan Logoff
    ("a", DAY1 + "10:00:00", "Logon"), ("a", DAY1 + "12:00:00", "Logon"),    # unlock
    ("a", DAY1 + "13:00:00", "Logoff"),
    ("b", DAY1 + "00:00:00", "Logon"), ("b", DAY1 + "05:00:00", "Logon"),    # past the cap
    ("b", DAY1 + "11:00:00", "Logon"), ("b", DAY1 + "15:00:00", "Logon"),
    ("b", DAY1 + "16:00:00", "Logoff"),
    ("c", DAY1 + "00:00:00", "Logon"), ("c", DAY1 + "12:00:00", "Logon"),    # exactly at cap
    ("c", DAY1 + "13:00:00", "Logoff"),
    ("e", DAY1 + "01:00:00", "Logon"), ("e", DAY1 + "06:00:00", "Logon"),    # never closed
    ("f", DAY1 + "09:00:00", "Logon"), ("f", DAY1 + "22:30:00", "Logoff"),   # Logoff past cap
    ("g", DAY1 + "08:00:00", "Logoff"),                                      # orphan Logoff
    ("h", DAY1 + "00:00:00", "Logon"), ("h", DAY1 + "13:00:00", "Logon"),    # two caps
    ("h", DAY2 + "02:00:00", "Logon"), ("h", DAY2 + "03:00:00", "Logoff"),
]
EXPECTED = [
    ("a", DAY1 + "08:00:00", DAY1 + "09:00:00", "logoff"),
    ("a", DAY1 + "10:00:00", DAY1 + "13:00:00", "logoff"),
    ("b", DAY1 + "00:00:00", DAY1 + "12:00:00", "cap"),
    ("b", DAY1 + "15:00:00", DAY1 + "16:00:00", "logoff"),
    ("c", DAY1 + "00:00:00", DAY1 + "13:00:00", "logoff"),
    ("e", DAY1 + "01:00:00", DAY1 + "13:00:00", "cap"),
    ("f", DAY1 + "09:00:00", DAY1 + "22:30:00", "logoff"),
    ("h", DAY1 + "00:00:00", DAY1 + "12:00:00", "cap"),
    ("h", DAY1 + "13:00:00", DAY2 + "01:00:00", "cap"),
    ("h", DAY2 + "02:00:00", DAY2 + "03:00:00", "logoff"),
]


def built():
    con = duckdb.connect(config={"threads": 1})
    con.execute('CREATE TABLE lg (id VARCHAR, "user" VARCHAR, pc VARCHAR, ts TIMESTAMP, activity VARCHAR)')
    for i, (user, ts, activity) in enumerate(LOGONS):
        con.execute("INSERT INTO lg VALUES (?, ?, 'p', ?, ?)", [f"{i:03d}", user, ts, activity])
    return con, build_sessions(con, "lg", 12)


def test_session_rules():
    con, orphan_logoffs = built()
    sessions = con.execute(
        'SELECT "user", start::VARCHAR, "end"::VARCHAR, end_type FROM sessions ORDER BY session_id'
    ).fetchall()
    assert sessions == EXPECTED
    assert orphan_logoffs == 2


def test_event_assignment():
    con, _ = built()
    events = {  # event id: (user, time, expected session)
        "inside": ("b", DAY1 + "11:30:00", 3),
        "at_cap_close": ("b", DAY1 + "12:00:00", 3),
        "after_cap_close": ("b", DAY1 + "13:00:00", None),
        "after_unlock": ("a", DAY1 + "12:30:00", 2),
        "between_sessions": ("a", DAY1 + "09:15:00", None),
    }
    con.execute('CREATE TABLE ev (id VARCHAR, "user" VARCHAR, pc VARCHAR, ts TIMESTAMP)')
    for event_id, (user, ts, _) in events.items():
        con.execute("INSERT INTO ev VALUES (?, ?, 'p', ?)", [event_id, user, ts])
    assigned = dict(con.execute(f"SELECT id, session_id FROM ({assign_sql('ev')})").fetchall())
    assert assigned == {event_id: expected for event_id, (_, _, expected) in events.items()}
