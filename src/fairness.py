"""Group fairness at the validation operating point (runbook 6.4).

Session units (primary aggregation: max over scoring points). For each seed, a session is
alerted if its score reaches that seed's session-level 5%-FPR threshold from the VALIDATION
evaluation (--eval-json; thresholds never come from the scored split). Per group of each
attribute (the fairness proxies in configs/features.yaml: role, department, functional_unit):
FPR among negative sessions and recall among positive sessions, mean and std over seeds.
Groups with fewer than --min-sessions sessions are not reported; recall is reported only for
groups with at least --min-positives positive sessions.

Run: python -m src.fairness --name slm_c1 --split test --allow-test \\
         --eval-json results/comparison1/eval/slm_valid_max.json --preds p_s1 p_s2 p_s3 \\
         --out results/comparison1/fairness_slm_test.csv
"""
import argparse
import json
import pathlib

import duckdb
import numpy as np
import pandas as pd
import yaml

from src.splits import verify_test_checksum

P = pathlib.Path("data/processed")
FPR = "fpr_0.05"


def session_frame(con, split, attributes):
    cols = ", ".join(f'any_value(f."{a}") AS "{a}"' for a in attributes)
    return con.execute(f"""
        SELECT f.session_id, {cols}, max(l.y) AS y
        FROM '{P}/features.parquet' f JOIN '{P}/splits.parquet' x USING (row_id)
        JOIN '{P}/labels.parquet' l USING (row_id)
        WHERE x.split = '{split}' GROUP BY f.session_id""").df()


def group_rates(sessions, alerts, attribute, min_sessions, min_positives):
    frame = sessions.assign(alert=alerts)
    out = []
    for value, g in frame.groupby(attribute):
        pos, neg = g[g["y"] == 1], g[g["y"] == 0]
        out.append({"attribute": attribute, "group": value, "sessions": len(g),
                    "negatives": len(neg), "positives": len(pos),
                    "fpr": float(neg["alert"].mean()) if len(neg) else np.nan,
                    "recall": float(pos["alert"].mean()) if len(pos) >= min_positives else np.nan})
    out = pd.DataFrame(out)
    return out[out["sessions"] >= min_sessions]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--eval-json", required=True, help="hourly validation eval (thresholds)")
    p.add_argument("--preds", nargs="+", required=True, help="one preds file per seed, of --split")
    p.add_argument("--split", choices=["valid", "calib", "test"], default="valid")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--min-sessions", type=int, default=200)
    p.add_argument("--min-positives", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    verify_test_checksum()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing to score test without --allow-test")
    ev = json.loads(pathlib.Path(args.eval_json).read_text())
    if ev["split"] != "valid" or ev.get("config", {}).get("session_agg", "max") != "max":
        raise SystemExit("--eval-json must be an hourly (session_agg max) validation evaluation")
    seeds = ev["session"]["per_seed"]
    if len(seeds) != len(args.preds):
        raise SystemExit(f"{len(args.preds)} preds files but {len(seeds)} seeds in the eval")
    attributes = yaml.safe_load(pathlib.Path("configs/features.yaml").read_text())["fairness"]["proxies"]

    con = duckdb.connect(config={"threads": 1})
    sessions = session_frame(con, args.split, attributes).sort_values("session_id").reset_index(drop=True)
    rows = con.execute(f"""SELECT x.row_id, x.session_id FROM '{P}/splits.parquet' x
                           WHERE x.split = '{args.split}'""").df()
    per_seed = []
    for seed_eval, path in zip(seeds, args.preds):
        preds = pd.read_parquet(path, columns=["row_id", "score"]).merge(rows, on="row_id")
        if len(preds) != len(rows):
            raise RuntimeError(f"{path}: rows do not match the {args.split} split")
        score = preds.groupby("session_id")["score"].max().reindex(sessions["session_id"]).to_numpy()
        alerts = score >= seed_eval["thresholds"][FPR]
        tables = [group_rates(sessions, alerts, a, args.min_sessions, args.min_positives)
                  for a in attributes]
        overall = sessions.assign(alert=alerts)
        tables.append(pd.DataFrame([{
            "attribute": "all", "group": "all", "sessions": len(overall),
            "negatives": int((overall["y"] == 0).sum()), "positives": int(overall["y"].sum()),
            "fpr": float(overall.loc[overall["y"] == 0, "alert"].mean()),
            "recall": float(overall.loc[overall["y"] == 1, "alert"].mean())}]))
        per_seed.append(pd.concat(tables, ignore_index=True))

    keys = ["attribute", "group", "sessions", "negatives", "positives"]
    stacked = pd.concat(per_seed, keys=range(len(per_seed)), names=["seed_index"]).reset_index(level=0)
    table = (stacked.groupby(keys, sort=False)
             .agg(fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
                  recall_mean=("recall", "mean"), recall_std=("recall", "std"))
             .reset_index())
    table.insert(0, "model", args.name)
    table.insert(1, "split", args.split)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}: {len(table)} rows (groups with >= {args.min_sessions} sessions; recall "
          f"where >= {args.min_positives} positives); thresholds from {args.eval_json}")


if __name__ == "__main__":
    main()
