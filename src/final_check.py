"""Final-compute check (S8-6; decisions log 2026-11-23), on the Mac after the 24 Nov lab session.

Final compute = SLM text, Comparison 2, seed 1, re-scored on the test logoff rows from its saved
adapter on the lab PC (results/final_compute/c2_s1_preds_test_final.parquet). PASS when:
  1. the frozen predictions results/c2_qwen15_r8_lr1e-4_s1/preds_test_final.parquet still match
     results/freeze/MANIFEST.sha256, and the re-scored file has the same row_ids and y_true;
  2. per-row Spearman correlation of the scores (re-scored vs frozen) >= 0.999 (a reproducibility
     check, not a reported metric);
  3. every Table 2 SLM-text metric of seed 1 (PR-AUC, recall at 1% and 5% FPR, precision at k),
     computed by src/eval.py on the re-scored file, is within ±0.002 of the frozen seed-1 value in
     results/comparison2/slm_text_test_final.json (session per_seed[0] = seed 1).
Writes results/final_compute/final_check.json and final_check.md; exit status 1 on FAIL.
Run from the repo root: python -m src.final_check
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = "results/c2_qwen15_r8_lr1e-4_s1/preds_test_final.parquet"
NEW = "results/final_compute/c2_s1_preds_test_final.parquet"
TABLE2 = "results/comparison2/slm_text_test_final.json"
MANIFEST = "results/freeze/MANIFEST.sha256"
OUT = ROOT / "results/final_compute"
EVAL_OUT = "results/final_compute/eval_slm_text_s1_test_final.json"
METRICS = ["pr_auc", "recall_at_fpr_0.01", "recall_at_fpr_0.05", "precision_at_k"]
SPEARMAN_MIN = 0.999
METRIC_TOL = 0.002
TAG = "frozen-2026-11-23"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_digest(rel):
    for line in (ROOT / MANIFEST).read_text().splitlines():
        digest, path = line.split("  ", 1)
        if path == rel:
            return digest
    raise SystemExit(f"{rel} not in {MANIFEST}")


def spearman(a, b):
    """Spearman rank correlation (average ranks for ties) = Pearson on ranks."""
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def check(new_path=NEW):
    from src import eval as ev
    from src.splits import verify_test_checksum
    checksum = verify_test_checksum()
    frozen_ok = sha256(ROOT / FROZEN) == manifest_digest(FROZEN)
    table2_ok = sha256(ROOT / TABLE2) == manifest_digest(TABLE2)
    frozen, new = pd.read_parquet(ROOT / FROZEN), pd.read_parquet(ROOT / new_path)
    merged = frozen.merge(new, on="row_id", how="outer", suffixes=("_frozen", "_new"), indicator=True)
    rows_ok = bool((merged["_merge"] == "both").all() and len(new) == len(frozen)
                   and (merged["y_true_frozen"] == merged["y_true_new"]).all())
    rho = spearman(merged["score_frozen"], merged["score_new"]) if rows_ok else float("nan")
    ev.main(["single", "--name", "slm_text_s1_final_compute", "--split", "test", "--allow-test", "--rows", "final",
             "--preds", new_path, "--out", EVAL_OUT])
    new_eval = json.loads((ROOT / EVAL_OUT).read_text())["session"]["per_seed"][0]
    frozen_eval = json.loads((ROOT / TABLE2).read_text())
    if not frozen_eval["preds"][0].endswith("_s1/preds_test_final.parquet"):
        raise SystemExit("Table 2 per_seed[0] is not seed 1")
    frozen_s1 = frozen_eval["session"]["per_seed"][0]
    metrics = {k: {"frozen": frozen_s1[k], "final_compute": new_eval[k], "difference": new_eval[k] - frozen_s1[k],
                   "pass": abs(new_eval[k] - frozen_s1[k]) <= METRIC_TOL} for k in METRICS}
    result = {
        "tag": TAG, "test_ids_sha256": checksum, "frozen_preds": FROZEN, "final_compute_preds": new_path,
        "frozen_preds_match_manifest": frozen_ok, "table2_json_matches_manifest": table2_ok,
        "rows": int(len(new)), "rows_match": rows_ok,
        "spearman": rho, "spearman_min": SPEARMAN_MIN, "spearman_pass": bool(rho >= SPEARMAN_MIN),
        "max_abs_score_difference": float((merged["score_frozen"] - merged["score_new"]).abs().max()) if rows_ok else None,
        "metric_tolerance": METRIC_TOL, "metrics": metrics, "eval_json": EVAL_OUT}
    result["pass"] = bool(frozen_ok and table2_ok and rows_ok and result["spearman_pass"]
                          and all(m["pass"] for m in metrics.values()))
    return result


def write(result):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_check.json").write_text(json.dumps(result, indent=1) + "\n")
    yes = lambda b: "yes" if b else "**NO**"
    md = [f"# Final compute check: {'PASS' if result['pass'] else 'FAIL'}", "",
          f"SLM text (C2) seed 1 re-scored on the {result['rows']:,} test logoff rows on the lab PC from its saved "
          f"adapter, code at tag `{result['tag']}`; compared with the frozen predictions (decisions log 2026-11-23).", "",
          "| Check | Value | Pass |", "|---|---|---|",
          f"| Frozen preds match the freeze manifest | {result['frozen_preds_match_manifest']} | {yes(result['frozen_preds_match_manifest'])} |",
          f"| Table 2 JSON matches the freeze manifest | {result['table2_json_matches_manifest']} | {yes(result['table2_json_matches_manifest'])} |",
          f"| Same row_ids and labels | {result['rows_match']} | {yes(result['rows_match'])} |",
          f"| Per-row Spearman (≥ {SPEARMAN_MIN}) | {result['spearman']:.6f} | {yes(result['spearman_pass'])} |",
          f"| Max \\|score difference\\| | {result['max_abs_score_difference']:.2e} | – |", "",
          f"| Table 2 metric, seed 1 (session = logoff row) | Frozen | Final compute | Difference | Within ±{METRIC_TOL} |",
          "|---|---|---|---|---|"]
    for k, m in result["metrics"].items():
        md.append(f"| {k} | {m['frozen']:.4f} | {m['final_compute']:.4f} | {m['difference']:+.4f} | {yes(m['pass'])} |")
    md += ["", f"Metrics from src/eval.py (`{result['eval_json']}`); Spearman is a reproducibility check, not a "
           "reported metric.", ""]
    (OUT / "final_check.md").write_text("\n".join(md))


def main():
    result = check()
    write(result)
    print(f"final compute check: {'PASS' if result['pass'] else 'FAIL'} (Spearman {result['spearman']:.6f}; "
          + ", ".join(f"{k} {m['difference']:+.4f}" for k, m in result["metrics"].items()) + ")")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
