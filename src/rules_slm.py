"""Step 7.3 rules + SLM (runbook 7.3; decisions log 2026-11-14).

Score = weighted rule count (configs/rules.yaml weights, without the 0.01 total-events
tie-break) + calibrated SLM C1 score in [0, 1). The rule count decides the order first; the
SLM orders rows within a count; rows with no rule fired are ranked by the SLM below every
fired row. The SLM score is the Platt row-level calibrated one (the hybrid's scale, fitted on
calib); seed i of the SLM gives seed i of the variant. Rules are deterministic and are not
calibrated; they enter by rank only.

Run: python -m src.rules_slm build --splits valid [calib test --allow-test]
                                                     (predictions; no labels used in the score)
     python -m src.rules_slm report                  (validation only: share of positives with no
                                                      rule fired, PR-AUC vs rules and SLM alone)
"""
import argparse
import json
import pathlib
import time

import duckdb
import numpy as np
import pandas as pd
import yaml

from src import eval as ev

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["rules_slm"]
RULES = yaml.safe_load(pathlib.Path("configs/rules.yaml").read_text())["rules"]
BELOW_ONE = np.nextafter(1.0, 0.0)


def rule_count(fired):
    """Weighted count of the rules named in each row's comma-separated rules_fired string."""
    weights = {r["id"]: float(r["weight"]) for r in RULES}
    return np.array([sum(weights[k] for k in s.split(",")) if s else 0.0 for s in fired])


def combine(count, p_slm):
    """count + SLM probability, the probability kept strictly below 1 so counts never mix."""
    p = np.minimum(np.asarray(p_slm, float), BELOW_ONE)
    return np.asarray(count, float) + p


def preds_path(seed, split):
    return pathlib.Path(CFG["out"]) / f"s{seed}" / f"preds_{split}.parquet"


def build(splits, allow_test=False):
    checksum = ev.verify_test_checksum()
    if "test" in splits and not allow_test:
        raise SystemExit("building test predictions needs --allow-test")
    con = duckdb.connect(config={"threads": 1})
    for split in splits:
        meta = ev.load_meta(con, split)
        rules = pd.read_parquet(pathlib.Path(CFG["rules"]) / f"preds_{split}.parquet",
                                columns=["row_id", "y_true", "rules_fired"]).set_index("row_id").reindex(meta["row_id"])
        if rules["y_true"].isna().any() or not np.array_equal(rules["y_true"].to_numpy(int), meta["y"].to_numpy()):
            raise RuntimeError(f"{split}: rules predictions do not cover the split's rows and labels")
        count = rule_count(rules["rules_fired"].to_numpy())
        slm = ev.load_preds([str(pathlib.Path(CFG["slm"].format(seed=s)) / f"preds_{split}.parquet")
                             for s in CFG["seeds"]], meta)
        for seed, p in zip(CFG["seeds"], slm):
            clipped = int((p > BELOW_ONE).sum())
            path = preds_path(seed, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"row_id": meta["row_id"], "y_true": meta["y"].to_numpy(), "score": combine(count, p),
                          "rule_count": count, "slm_score": p}).to_parquet(path, index=False)
            print(f"{split} s{seed}: {len(p):,} rows, {int((count > 0).sum()):,} with a rule fired, "
                  f"{clipped} SLM scores clipped below 1; {path}", flush=True)
    print(f"test_ids.csv SHA-256 {checksum}")


def report():
    """Validation only."""
    started = time.perf_counter()
    split = "valid"
    con = duckdb.connect(config={"threads": 1})
    meta = ev.load_meta(con, split)
    y = meta["y"].to_numpy()
    combo = ev.load_preds([str(preds_path(s, split)) for s in CFG["seeds"]], meta)
    slm = ev.load_preds([str(pathlib.Path(CFG["slm"].format(seed=s)) / f"preds_{split}.parquet")
                         for s in CFG["seeds"]], meta)
    rules = ev.load_preds([str(pathlib.Path(CFG["rules"]) / f"preds_{split}.parquet")], meta)[0]
    count = pd.read_parquet(preds_path(CFG["seeds"][0], split), columns=["row_id", "rule_count"]) \
        .set_index("row_id")["rule_count"].reindex(meta["row_id"]).to_numpy()
    units, (session_count,) = ev.aggregate_sessions(meta, [count])
    sy = units["y"].to_numpy()

    def both(scores):
        row = [ev.pr_auc(s, y) for s in scores]
        units_, per = ev.aggregate_sessions(meta, scores)
        ses = [ev.pr_auc(s, units_["y"].to_numpy()) for s in per]
        return {"row_mean": float(np.mean(row)), "row_std": float(np.std(row, ddof=1)) if len(row) > 1 else None,
                "session_mean": float(np.mean(ses)),
                "session_std": float(np.std(ses, ddof=1)) if len(ses) > 1 else None}

    result = {
        "split": split,
        "zero_rule_positives": {
            "rows": {"positives": int(y.sum()), "no_rule_fired": int(((count == 0) & (y == 1)).sum()),
                     "share": float(((count == 0) & (y == 1)).sum() / y.sum())},
            "sessions": {"positives": int(sy.sum()), "no_rule_fired": int(((session_count == 0) & (sy == 1)).sum()),
                         "share": float(((session_count == 0) & (sy == 1)).sum() / sy.sum())}},
        "rows_with_a_rule_fired_share": float((count > 0).mean()),
        "pr_auc": {"rules_slm": both(combo), "rules_alone": both([rules]), "slm_alone": both(slm)},
        "runtime_seconds": round(time.perf_counter() - started, 1)}
    out = pathlib.Path(CFG["out"])
    (out / "report_valid.json").write_text(json.dumps(result, indent=2) + "\n")
    z, r = result["zero_rule_positives"], result["pr_auc"]
    fmt = lambda d, k: f"{d[k + '_mean']:.4f}" + (f" ± {d[k + '_std']:.4f}" if d[k + "_std"] is not None else "")
    md = ["# Rules + SLM (Step 7.3, validation only)", "",
          "Score = weighted rule count (no tie-break) + Platt row-level calibrated SLM C1 score in "
          "[0, 1); seed i of the SLM gives seed i of the variant. Rules alone is the single "
          "deterministic run (with its planned tie-break). Mean ± std over seeds.", "",
          f"Rows with at least one rule fired: {result['rows_with_a_rule_fired_share']:.2%}.", "",
          f"Positives with no rule fired: rows {z['rows']['no_rule_fired']:,} / {z['rows']['positives']:,} "
          f"({z['rows']['share']:.1%}); sessions {z['sessions']['no_rule_fired']:,} / "
          f"{z['sessions']['positives']:,} ({z['sessions']['share']:.1%}).", "",
          "| Model | Row PR-AUC | Session PR-AUC |", "|---|---|---|"]
    for key, label in (("rules_slm", "Rules + SLM"), ("rules_alone", "Rules alone"), ("slm_alone", "SLM alone")):
        md.append(f"| {label} | {fmt(r[key], 'row')} | {fmt(r[key], 'session')} |")
    md.append("")
    (out / "report_valid.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "report"])
    p.add_argument("--splits", nargs="+", default=["valid"], choices=["valid", "calib", "test"])
    p.add_argument("--allow-test", action="store_true")
    args = p.parse_args()
    build(args.splits, args.allow_test) if args.command == "build" else report()


if __name__ == "__main__":
    main()
