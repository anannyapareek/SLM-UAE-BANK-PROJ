import duckdb

con = duckdb.connect(config={"threads": 1})
P = "data/processed"

for t in ["logon", "device", "file", "email", "http"]:
    n, lo, hi = con.execute(
        f"SELECT count(*), min(ts), max(ts) FROM '{P}/{t}.parquet'").fetchone()
    print(f"{t}: {n:,} rows, {lo} to {hi}")

for t in ["file", "email", "http"]:
    # Full read of id and content forces every data page to be decoded
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
