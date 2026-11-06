"""Processed tables: row counts, timestamps, and that every value can be read.

`count(*)` reads the Parquet footer and cannot see damaged pages, so these tests read
the ids and content themselves.
"""
import pytest

EXPECTED_ROWS = {"logon": 854_859, "device": 405_380, "file": 445_581,
                 "email": 2_629_979, "http": 28_434_423}
CONTENT_TABLES = ["file", "email", "http"]


@pytest.mark.parametrize("name", EXPECTED_ROWS)
def test_main_table(con, table, is_full_data, name):
    rows, null_ts = con.execute(
        f"SELECT count(*), count(*) - count(ts) FROM {table(name)}").fetchone()
    assert null_ts == 0
    if is_full_data:
        assert rows == EXPECTED_ROWS[name]


def test_ldap_users(con, table, is_full_data):
    users = con.execute(f"SELECT count(DISTINCT user_id) FROM {table('ldap')}").fetchone()[0]
    assert users == 1000 or not is_full_data


@pytest.mark.slow
@pytest.mark.parametrize("name", CONTENT_TABLES)
def test_content_table_reads_in_full(con, table, name):
    rows, ids, chars = con.execute(f"""
        SELECT count(*), count(DISTINCT id), sum(length(content)) FROM {table(name + '_content')}
    """).fetchone()
    assert rows == ids and chars > 0
    missing = con.execute(f"""
        SELECT count(*) FROM {table(name + '_content')} c ANTI JOIN {table(name)} m USING (id)
    """).fetchone()[0]
    assert missing == 0
