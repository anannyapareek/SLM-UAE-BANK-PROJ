"""Integrity check of one predictions file; no labels read, no metrics computed.

Checks: row count, NaN/inf in score and margin, duplicate row_ids, score within [0, 1], and
that the row_ids are exactly the split's rows (all rows, or the final rows with --final).
Prints one line starting with "CHECK OK" or "CHECK FAIL" and exits 0 or 2.

Run from the repo root:
    python scripts/check_preds.py --preds runs/x/preds_test.parquet --split test [--final]
"""
import argparse
import sys

import duckdb
import numpy as np
import pandas as pd

P = "data/processed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", required=True)
    p.add_argument("--split", required=True, choices=["train", "valid", "calib", "test"])
    p.add_argument("--final", action="store_true", help="file holds final (logoff) rows only")
    args = p.parse_args()

    con = duckdb.connect(config={"threads": 1})
    final = " AND f.is_final" if args.final else ""
    expected = set(con.execute(f"""
        SELECT x.row_id FROM '{P}/splits.parquet' x JOIN '{P}/features.parquet' f USING (row_id)
        WHERE x.split = '{args.split}'{final}""").df()["row_id"])
    d = pd.read_parquet(args.preds, columns=["row_id", "score", "margin"])
    bad_score = int((~np.isfinite(d["score"].to_numpy(dtype=float))).sum())
    bad_margin = int((~np.isfinite(d["margin"].to_numpy(dtype=float))).sum())
    dup = int(d["row_id"].duplicated().sum())
    out_of_range = int(((d["score"] < 0) | (d["score"] > 1)).sum())
    ids_match = set(d["row_id"]) == expected and not dup
    ok = bad_score == 0 and bad_margin == 0 and dup == 0 and out_of_range == 0 and ids_match
    print(f"CHECK {'OK' if ok else 'FAIL'} | {args.preds} | split {args.split}"
          f"{' final' if args.final else ''} | rows {len(d):,} (expected {len(expected):,}) | "
          f"NaN/inf score {bad_score} margin {bad_margin} | duplicate row_ids {dup} | "
          f"score outside [0,1] {out_of_range} | row_ids match split {ids_match}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
