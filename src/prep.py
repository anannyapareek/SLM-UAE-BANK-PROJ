import duckdb, pathlib

RAW = pathlib.Path("data/raw/r4.2")
OUT = pathlib.Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET threads=1")

for name in ["logon", "device", "file", "email", "http"]:
    src = RAW / f"{name}.csv"
    cols = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_csv('{src}', all_varchar=true, header=true)").fetchall()]
    base = [c for c in cols if c != "content"]
    sel = ", ".join(
        "strptime(date, '%m/%d/%Y %H:%M:%S') AS ts" if c == "date" else f'"{c}"'
        for c in base)
    con.execute(f"COPY (SELECT {sel} FROM read_csv('{src}', all_varchar=true, header=true)) "
                f"TO '{OUT}/{name}.parquet' (FORMAT parquet)")
    if "content" in cols:
        con.execute(f"COPY (SELECT id, content FROM read_csv('{src}', all_varchar=true, header=true)) "
                    f"TO '{OUT}/{name}_content.parquet' (FORMAT parquet)")
    n, lo, hi = con.execute(
        f"SELECT count(*), min(ts), max(ts) FROM '{OUT}/{name}.parquet'").fetchone()
    print(f"{name}: {n:,} rows, {lo} to {hi}, columns {base}")

con.execute(f"COPY (SELECT *, filename AS month_file FROM read_csv('{RAW}/LDAP/*.csv', "
            f"all_varchar=true, header=true, filename=true, union_by_name=true)) "
            f"TO '{OUT}/ldap.parquet' (FORMAT parquet)")
print("ldap:", con.execute(f"SELECT count(*) FROM '{OUT}/ldap.parquet'").fetchone()[0], "rows")
