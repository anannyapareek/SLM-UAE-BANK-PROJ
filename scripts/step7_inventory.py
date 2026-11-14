"""Step 7 P0: inventory of the prediction files Step 7 builds on. Computes no metrics.

For every model, seed and split: path, rows, columns, whether a margin is present, whether
score = sigmoid(margin) (so the margin is the logit of the stored probability), and row_id
coverage (the full split for hourly files, the split's logoff rows for *_final files; test
row_ids against the frozen test_ids.csv). Also the Gate 3 status (Comparisons 1 and 2
evaluated with 3 seeds). Writes results/step7/inventory.md.

Run from the repo root: python -m scripts.step7_inventory
"""
import json
import pathlib

import duckdb
import numpy as np
import pandas as pd

from src.splits import verify_test_checksum

OUT = pathlib.Path("results/step7/inventory.md")
MODELS = [
    ("C1", "xgb_struct", "XGBoost v3 (structured, hourly)", "results/xgb_struct_v3_s{s}", "", (1, 2, 3)),
    ("C1", "slm_struct", "SLM C1 (structured, hourly)", "results/c1_qwen15_r8_lr1e-4_s{s}", "", (1, 2, 3)),
    ("C1", "rules", "Rules (hourly)", "results/rules", "", (None,)),
    ("C2", "xgb_text", "XGBoost text (logoff rows)", "results/xgb_text_s{s}", "_final", (1, 2, 3)),
    ("C2", "slm_text", "SLM text (logoff rows)", "results/c2_qwen15_r8_lr1e-4_s{s}", "_final", (1, 2, 3)),
    ("C2", "rules", "Rules (logoff rows)", "results/rules", "_final", (None,)),
]
SPLITS = ("valid", "calib", "test")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    checksum = verify_test_checksum()
    con = duckdb.connect(config={"threads": 1})
    meta = con.execute("""SELECT x.row_id, x.split, f.is_final FROM 'data/processed/splits.parquet' x
                          JOIN 'data/processed/features.parquet' f USING (row_id)""").df()
    expected = {(sp, fin): set(meta.loc[(meta.split == sp) & (meta.is_final | (not fin)), "row_id"])
                for sp in SPLITS for fin in (False, True)}
    frozen = set(pathlib.Path("data/processed/test_ids.csv").read_text().split()[1:])

    lines = ["# Step 7 inventory (P0)", "",
             f"test_ids.csv SHA-256 `{checksum}` (verified). No metrics computed.", "",
             "| Comparison | Model | Seed | Split | File | Rows | Columns | Margin | max abs(score − σ(margin)) | row_ids = expected | test = frozen |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    problems = []
    for comp, key, label, folder, suffix, seeds in MODELS:
        for seed in seeds:
            for split in SPLITS:
                path = pathlib.Path(folder.format(s=seed)) / f"preds_{split}{suffix}.parquet"
                if not path.exists():
                    problems.append(f"missing {path}")
                    continue
                d = pd.read_parquet(path)
                has_margin = "margin" in d.columns
                gap = (float(np.max(np.abs(d["score"].to_numpy(float) - sigmoid(d["margin"].to_numpy(float)))))
                       if has_margin else None)
                ids = set(d["row_id"])
                ok = ids == expected[(split, bool(suffix))] and not d["row_id"].duplicated().any()
                tst = "" if split != "test" else ("yes" if (ids <= frozen and (bool(suffix) or ids == frozen)) else "NO")
                if not ok or tst == "NO" or (gap is not None and gap > 1e-3):
                    problems.append(f"{path}: ids ok {ok}, test {tst}, gap {gap}")
                lines.append(f"| {comp} | {label} | {seed or '-'} | {split} | `{path}` | {len(d):,} | "
                             f"{', '.join(d.columns)} | {'yes' if has_margin else 'no'} | "
                             f"{'n/a' if gap is None else f'{gap:.2e}'} | {'yes' if ok else 'NO'} | {tst} |")

    gate = []
    for comp, files in (("Comparison 1", ["results/comparison1/eval/slm_test_max.json",
                                          "results/comparison1/eval/xgb_test_max.json"]),
                        ("Comparison 2", ["results/comparison2/slm_text_test_final.json",
                                          "results/comparison2/xgb_text_test_final.json"])):
        seeds = [len(json.loads(pathlib.Path(f).read_text())["preds"]) for f in files]
        gate.append(f"- {comp}: evaluated, seeds per model {seeds} ({'3 seeds' if all(s == 3 for s in seeds) else 'NOT 3 seeds'})")
    lines += ["", "## Gate 3 inputs", *gate,
              "- Calibration (7.1): to be done in P1 with the same 3 seeds.", "",
              "## Problems", *(problems or ["none"]), ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}; problems: {len(problems)}")
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main()
