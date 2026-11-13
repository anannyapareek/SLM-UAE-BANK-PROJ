"""Logoff-only copies of hourly prediction files (Comparison 2 and the text-gain comparison).

For each run and split, writes preds_<split>_final.parquet next to preds_<split>.parquet,
keeping only the rows with is_final == True in data/processed/features.parquet. Columns and
values are copied unchanged (margin kept where present); the originals are not modified.
An existing copy must equal the new one exactly, or the script stops. Computes no metrics.

Run from the repo root: python scripts/make_final_copies.py
"""
import pathlib
import sys

import duckdb
import pandas as pd

RUNS = {
    "text XGBoost": [f"results/xgb_text_s{s}" for s in (1, 2, 3)],
    "structured XGBoost": [f"results/xgb_struct_v3_s{s}" for s in (1, 2, 3)],
    "SLM Comparison 1": [f"results/c1_qwen15_r8_lr1e-4_s{s}" for s in (1, 2, 3)],
    "rules": ["results/rules"],
}
SPLITS = ("valid", "calib", "test")


def main():
    con = duckdb.connect(config={"threads": 1})
    final = set(con.execute("SELECT row_id FROM 'data/processed/features.parquet' "
                            "WHERE is_final").df()["row_id"])
    for label, folders in RUNS.items():
        for folder in folders:
            for split in SPLITS:
                src = pathlib.Path(folder) / f"preds_{split}.parquet"
                dst = pathlib.Path(folder) / f"preds_{split}_final.parquet"
                d = pd.read_parquet(src)
                copy = d[d["row_id"].isin(final)].reset_index(drop=True)
                if dst.exists():
                    if not pd.read_parquet(dst).equals(copy):
                        sys.exit(f"{dst} exists and differs from a fresh copy; not overwritten")
                    status = "exists, identical"
                else:
                    copy.to_parquet(dst, index=False)
                    status = "written"
                print(f"{label} | {dst} | {len(copy):,} of {len(d):,} rows | {status}")


if __name__ == "__main__":
    main()
