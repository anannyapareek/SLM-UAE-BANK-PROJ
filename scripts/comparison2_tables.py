"""Comparison 2 tables from the eval.py and src.text_gain outputs (test, logoff rows).

Reads results/comparison2/*.json (no data, no labels) and writes:
  table2.md, table2.csv        SLM text, XGBoost text, rules
  text_gain.md, text_gain.csv  per model C1-final -> C2 change (paired) and the DiD
  run.csv                      one row per eval run
  fairness.md                  from fairness_{slm_text,xgb_text}_test.csv (src/fairness.py)
Run from the repo root: python scripts/comparison2_tables.py
"""
import datetime
import json
import pathlib
import subprocess
import sys

import pandas as pd

D = pathlib.Path("results/comparison2")
METRICS = [("pr_auc", "PR-AUC"), ("recall_at_fpr_0.01", "Recall @ 1% FPR"),
           ("recall_at_fpr_0.05", "Recall @ 5% FPR"), ("precision_at_k", "Precision @ k (20/day)")]
TABLE2 = [("slm_text", "SLM, text (Qwen2.5-1.5B LoRA)"), ("xgb_text", "XGBoost, text"),
          ("rules", "Rules (reference)")]
SINGLES = ["slm_text", "xgb_text", "rules", "slm_struct", "xgb_struct"]
PAIRED = {"slm_text_vs_xgb_text": "SLM text vs XGBoost text",
          "slm_text_vs_slm_struct": "SLM text vs SLM structured (final rows)",
          "xgb_text_vs_xgb_struct": "XGBoost text vs XGBoost structured (final rows)"}


def load(name):
    return json.loads((D / name).read_text())


def cell(m, k):
    std = m["std"][k]
    sd = "" if std is None else f" ± {std:.3f}"
    return f"{m['mean'][k]:.3f}{sd} [{m['ci'][k][0]:.3f}, {m['ci'][k][1]:.3f}]"


def table2():
    md = ["# Table 2: Comparison 2 on the test split (text added, logoff-only)", "",
          "| Model | " + " | ".join(t for _, t in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    rows = []
    for key, label in TABLE2:
        d = load(f"{key}_test_final.json")
        m = d["session"]
        md.append(f"| {label} | " + " | ".join(cell(m, k) for k, _ in METRICS) + " |")
        for k, _ in METRICS:
            rows.append({"model": key, "metric": k, "mean": m["mean"][k], "std": m["std"][k],
                         "ci_low": m["ci"][k][0], "ci_high": m["ci"][k][1],
                         "boot_skipped": m["boot_skipped"][k]})
    c = load("slm_text_test_final.json")["session"]["counts"]
    skipped = {key: load(f"{key}_test_final.json")["boot_skipped"] for key, _ in TABLE2}
    md += ["", f"Mean ± std over 3 seeds (rules: deterministic, 1 run) and 95% user-level "
           f"bootstrap CI (1,000 resamples, seed 0), src/eval.py with --rows final. "
           f"Logoff-only (one row per session): n_units {c['n_units']:,}, n_pos {c['n_pos']:,}, "
           f"n_users {c['n_users']:,}, n_pos_users {c['n_pos_users']:,}, positive rate "
           f"{c['positive_rate']:.4%}; boot_skipped "
           + ", ".join(f"{k} {v}" for k, v in skipped.items()) + ". Precision-at-k ceiling "
           f"{load('slm_text_test_final.json')['session']['precision_at_k_ceiling']:.3f}.", ""]
    pair = load("paired_slm_text_vs_xgb_text.json")["session"]
    md += ["## Paired test: SLM text − XGBoost text (main Comparison 2 test)", "",
           "Same users in every resample (1,000, seed 0); 95% CI; significant = CI excludes 0.", "",
           "| Metric | SLM text | XGBoost text | Difference | 95% CI | Significant |",
           "|---|---|---|---|---|---|"]
    for k, t in METRICS:
        r = pair[k]
        md.append(f"| {t} | {r['a']:.3f} | {r['b']:.3f} | {r['difference']:+.3f} | "
                  f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {'yes' if r['significant'] else 'no'} |")
    md.append("")
    (D / "table2.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(D / "table2.csv", index=False, float_format="%.6f")


def text_gain():
    did = load("text_gain_did.json")
    md = ["# Text gain on the test split (logoff rows)", "",
          "Comparison 1 at logoff rows (structured, final point; identical to Comparison 1's "
          "final-point aggregation) against Comparison 2 (text). Paired user bootstrap, the "
          "same 1,000 user draws (seed 0) for every row; 95% CI; significant = CI excludes 0.", ""]
    rows = []
    for k, title in METRICS:
        md += [f"## {title}", "", "| Comparison | Text (C2) | Structured (C1 final) | Change | 95% CI | Significant |",
               "|---|---|---|---|---|---|"]
        for pair, label in (("slm_text_vs_slm_struct", "SLM: text vs structured"),
                            ("xgb_text_vs_xgb_struct", "XGBoost: text vs structured")):
            r = load(f"paired_{pair}.json")["session"][k]
            md.append(f"| {label} | {r['a']:.3f} | {r['b']:.3f} | {r['difference']:+.3f} | "
                      f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {'yes' if r['significant'] else 'no'} |")
            rows.append({"metric": k, "comparison": pair, "text": r["a"], "structured_final": r["b"],
                         "change": r["difference"], "ci_low": r["ci"][0], "ci_high": r["ci"][1],
                         "significant": r["significant"]})
        x = did["did"][k]
        md.append(f"| **Difference-in-differences** (SLM gain − XGBoost gain) | {x['slm_gain']:+.3f} | "
                  f"{x['xgb_gain']:+.3f} | {x['difference_in_differences']:+.3f} | "
                  f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {'yes' if x['significant'] else 'no'} |")
        md.append("")
        rows.append({"metric": k, "comparison": "difference_in_differences",
                     "text": x["slm_gain"], "structured_final": x["xgb_gain"],
                     "change": x["difference_in_differences"], "ci_low": x["ci"][0],
                     "ci_high": x["ci"][1], "significant": x["significant"]})
    md += ["For the difference-in-differences row, the Text and Structured columns hold the "
           "SLM gain and the XGBoost gain. Its CI uses the same user draws as the paired tests "
           "(src/text_gain.py reuses src/eval.py; its pair CIs equal eval.py's exactly).", ""]
    (D / "text_gain.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(D / "text_gain.csv", index=False, float_format="%.6f")


def run_log():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    rows = []
    files = ([f"{k}_test_final.json" for k in SINGLES] + [f"paired_{p}.json" for p in PAIRED]
             + ["text_gain_did.json", "slm_text_valid_final.json", "xgb_text_valid_final.json"])
    for name in files:
        d = load(name)
        mode = "did" if name.startswith("text_gain") else ("paired" if name.startswith("paired") else "single")
        models = (" vs ".join([d["name_a"], d["name_b"]]) if mode == "paired" else
                  d.get("name", "slm_text, slm_struct, xgb_text, xgb_struct"))
        preds = (d.get("preds") if mode == "single" else
                 (d.get("preds_a", []) + d.get("preds_b", []) if mode == "paired" else
                  [p for v in d["preds"].values() for p in v]))
        rows.append({"timestamp": datetime.datetime.fromtimestamp((D / name).stat().st_mtime).isoformat(timespec="seconds"),
                     "eval": name.removesuffix(".json"), "mode": mode, "split": d["split"],
                     "rows": d.get("rows", "final"), "models": models, "n_preds_files": len(preds),
                     "preds": ";".join(preds), "out": str(D / name),
                     "boot_skipped": d["boot_skipped"], "runtime_seconds": d["runtime_seconds"],
                     "test_ids_sha256": d["test_ids_sha256"], "code_commit": commit})
    pd.DataFrame(rows).to_csv(D / "run.csv", index=False)


def fairness():
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from comparison1_tables import fairness as report
    report(D, models=[("slm_text", "SLM, text (Qwen2.5-1.5B LoRA)"), ("xgb_text", "XGBoost, text")],
           title="# Fairness, Comparison 2 (text) on the test split (runbook 6.4)",
           intro=("Logoff-only (one row per session, --rows final); alert at each seed's 5%-FPR "
                  "threshold from the validation final-row eval. "),
           csv="fairness_{key}_test.csv")


if __name__ == "__main__":
    table2()
    text_gain()
    run_log()
    fairness()
    print("wrote", ", ".join(sorted(p.name for p in D.glob("*.md")) + ["table2.csv", "text_gain.csv", "run.csv"]))
