"""Hand-written rules baseline, scored on validation only.

The rules are deterministic, so there is one run and no tuning. Rules live in
configs/rules.yaml and use only features from configs/features.yaml.
Run with: python run.py rules
"""
import json
import operator
import os
import pathlib

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.splits import verify_test_checksum

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
RULES_CONFIG = pathlib.Path("configs/rules.yaml")
FEATURES_CONFIG = pathlib.Path("configs/features.yaml")
OUT = pathlib.Path("results/rules")
SPLIT = "valid"
OPS = {"==": operator.eq, ">": operator.gt, ">=": operator.ge, "<": operator.lt}


def features_used(condition):
    if "any" in condition:
        return {f for c in condition["any"] for f in features_used(c)}
    return {condition["feature"]}


def holds(condition, frame):
    if "any" in condition:
        return np.logical_or.reduce([holds(c, frame) for c in condition["any"]])
    return OPS[condition["op"]](frame[condition["feature"]], condition["value"]).to_numpy()


def text(condition):
    if "any" in condition:
        return "(" + " or ".join(text(c) for c in condition["any"]) + ")"
    return f"{condition['feature']} {condition['op']} {condition['value']}"


def methods_paragraph(rules, tie_break):
    lower_first = lambda t: t if t[:2].isupper() else t[0].lower() + t[1:]
    parts = [f"{r['id']} ({lower_first(r['description'])}: "
             f"{' and '.join(text(c) for c in r['conditions'])})" for r in rules]
    count = {4: "four", 5: "five", 6: "six"}.get(len(rules), str(len(rules)))
    return (
        f"As a transparent baseline we score each row with {count} hand-written rules, set from "
        "domain reasoning before looking at any results and not tuned. Each rule is a "
        "conjunction of thresholds on the same structured features the models use, read at "
        "the scoring time, so a rule only sees events up to that point in the session: "
        + "; ".join(parts) + ". "
        "Every rule has weight 1, and a row's score is the sum of the weights of the rules "
        "that fire. Ties are broken by a planned tie-break that uses no labels: "
        f"{tie_break['weight']} times the percentile rank, within the split, of the row's total "
        f"session events so far ({', '.join(tie_break['features'])}). The fired rule "
        "identifiers are "
        "kept as the explanation. Because the dataset records only file copies to removable "
        "media, R1 treats any file event as a USB copy. R3 approximates external email with "
        "attachments by requiring an external recipient and an attachment in the same "
        "session, as no feature links the two on a single email. R4 targets a published CERT "
        "threat pattern that does not occur in release r4.2 and is kept as a general rule. The rules "
        "are deterministic, so we report a single run.\n")


def per_rule_table(rules, per_rule, scenarios):
    header = (["Rule", "Description", "Fire rate", "Rows fired", "Precision when fired",
               "Recall"] + [f"Recall, scenario {s}" for s in scenarios] + ["Target scenarios"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rules:
        s = per_rule[r["id"]]
        precision = "n/a" if s["precision"] is None else f"{s['precision']:.4f}"
        lines.append("| " + " | ".join(
            [r["id"], r["description"], f"{s['fire_rate']:.2%}", f"{s['rows_fired']:,}", precision,
             f"{s['recall']:.4f}"]
            + [f"{s['recall_by_scenario'][str(x)]:.4f}" for x in scenarios]
            + [", ".join(map(str, r["target_scenarios"])) or "none in r4.2"]) + " |")
    return lines


def run():
    verify_test_checksum()
    config = yaml.safe_load(RULES_CONFIG.read_text())
    rules, tie_break = config["rules"], config["tie_break"]
    known = {f["name"] for f in yaml.safe_load(FEATURES_CONFIG.read_text())["features"]}
    needed = sorted(set().union(*(features_used(c) for r in rules for c in r["conditions"]))
                    | set(tie_break["features"]))
    missing = sorted(set(needed) - known)
    if missing:
        raise RuntimeError(f"rules use features not in features.yaml: {missing}")

    con = duckdb.connect(config={"threads": 1})
    columns = ", ".join(f'f."{n}"' for n in needed)
    frame = con.execute(f"""
        SELECT f.row_id, f."user", l.y, l.scenario, {columns}
        FROM '{P}/features.parquet' f
        JOIN '{P}/splits.parquet' x USING (row_id)
        JOIN '{P}/labels.parquet' l USING (row_id)
        WHERE x.split = '{SPLIT}' ORDER BY f.row_id
    """).df()

    fired = {r["id"]: np.logical_and.reduce([holds(c, frame) for c in r["conditions"]])
             for r in rules}
    # Planned tie-break, uses no labels: rank of total session events so far in this split.
    events = frame[tie_break["features"]].sum(axis=1)
    score = (sum(r["weight"] * fired[r["id"]] for r in rules).astype(float)
             + tie_break["weight"] * events.rank(pct=True, method="average").to_numpy())
    names = np.array([",".join(r["id"] for r in rules if fired[r["id"]][i])
                      for i in range(len(frame))])
    y = frame["y"].to_numpy()
    scenario = frame["scenario"].to_numpy()
    scenarios = sorted(int(x) for x in pd.unique(scenario[y == 1]))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": frame["row_id"], "user": frame["user"], "y_true": y,
                  "score": score, "rules_fired": names}).to_parquet(OUT / "preds_valid.parquet",
                                                                    index=False)
    per_rule = {}
    print(f"{SPLIT}: {len(y):,} rows, {int(y.sum()):,} positive ({y.mean():.4%})")
    print("rule | fire rate | rows fired | precision when fired | recall | "
          + " | ".join(f"recall s{s}" for s in scenarios))
    for r in rules:
        f = fired[r["id"]]
        stats = {"fire_rate": float(f.mean()), "rows_fired": int(f.sum()),
                 "precision": float(y[f].mean()) if f.any() else None,
                 "recall": float(f[y == 1].mean()),
                 "recall_by_scenario": {str(s): float(f[(y == 1) & (scenario == s)].mean())
                                        for s in scenarios}}
        per_rule[r["id"]] = stats
        precision = "n/a" if stats["precision"] is None else f"{stats['precision']:.4f}"
        print(f"{r['id']} | {stats['fire_rate']:.4%} | {stats['rows_fired']:,} | {precision} | "
              f"{stats['recall']:.4f} | "
              + " | ".join(f"{stats['recall_by_scenario'][str(s)]:.4f}" for s in scenarios))
        for target in r["target_scenarios"]:
            if stats["recall_by_scenario"].get(str(target), 0) == 0:
                cols = sorted(set().union(*(features_used(c) for c in r["conditions"])))
                hits = frame[(y == 1) & (scenario == target)][cols]
                print(f"  ZERO RECALL: {r['id']} on its target scenario {target} "
                      f"({len(hits):,} validation positives); feature values:")
                print("  " + hits.describe().T[["mean", "min", "50%", "max"]]
                      .to_string().replace("\n", "\n  "))
                print("  share of positives where each condition holds: " + ", ".join(
                    f"{text(c)}: {holds(c, frame)[(y == 1) & (scenario == target)].mean():.2%}"
                    for c in r["conditions"]))
    any_rule = names != ""
    metrics = {
        "split": SPLIT, "rows": int(len(y)), "positives": int(y.sum()),
        "positive_rate": float(y.mean()), "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "any_rule": {"fire_rate": float(any_rule.mean()),
                     "precision": float(y[any_rule].mean()) if any_rule.any() else None,
                     "recall": float(any_rule[y == 1].mean())},
        "rules": per_rule,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "rules_methods.md").write_text(
        methods_paragraph(rules, tie_break) + "\n" + "\n".join(per_rule_table(rules, per_rule, scenarios))
        + "\n\nThe rules fire at reasonable rates but with low precision, because normal CERT "
        "users also show these patterns.\n")
    print(f"any rule | {metrics['any_rule']['fire_rate']:.4%} | precision "
          f"{metrics['any_rule']['precision']:.4f} | recall {metrics['any_rule']['recall']:.4f}")
    print(f"overall PR-AUC {metrics['pr_auc']:.4f}, ROC-AUC {metrics['roc_auc']:.4f}")
