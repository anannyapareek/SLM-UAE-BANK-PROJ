"""Scoring granularity on validation: time to detection under hourly scoring.

For each seed of a model's validation predictions, take the session-level threshold at
5% FPR that src/eval.py saved for hourly scoring (session score = max over scoring points).
For every positive validation session, find the first scoring point whose score reaches
that threshold (score >= threshold, eval.py's rule), and compare it with the session start,
the first malicious event and the logoff (final scoring point).

Validation only: no other split can be passed.

Run: python -m src.granularity --name xgb_struct_v3 \\
         --eval-json results/granularity/xgb_struct_v3_hourly_valid.json \\
         --preds results/xgb_struct_v3_s{1,2,3}/preds_valid.parquet \\
         --out results/granularity/xgb_struct_v3_time_to_detection.json
"""
import argparse
import json
import pathlib

import duckdb
import numpy as np
import pandas as pd

from src.splits import verify_test_checksum

P = pathlib.Path("data/processed")
FPR = "fpr_0.05"


def positive_sessions(con):
    """Positive validation sessions: start, logoff (final scoring point), first malicious
    event time, first positive scoring point."""
    return con.execute(f"""
        WITH v AS (
            SELECT f.row_id, f.session_id, f.score_time, f.is_final, l.y
            FROM '{P}/features.parquet' f
            JOIN '{P}/splits.parquet' x USING (row_id)
            JOIN '{P}/labels.parquet' l USING (row_id)
            WHERE x.split = 'valid'),
        pos AS (SELECT session_id FROM v GROUP BY 1 HAVING max(y) = 1),
        first_event AS (
            SELECT s.session_id, min(m.ts) AS first_event
            FROM '{P}/sessions.parquet' s JOIN pos USING (session_id)
            JOIN '{P}/malicious_events.parquet' m
              ON m."user" = s."user" AND m.pc = s.pc AND m.ts BETWEEN s.start AND s."end"
            GROUP BY 1)
        SELECT p.session_id, s.start,
               max(v.score_time) FILTER (v.is_final) AS logoff,
               fe.first_event,
               min(v.score_time) FILTER (v.y = 1) AS first_positive_point,
               count(*) AS scoring_points
        FROM pos p JOIN '{P}/sessions.parquet' s USING (session_id)
        JOIN v USING (session_id) LEFT JOIN first_event fe USING (session_id)
        GROUP BY ALL ORDER BY p.session_id""").df()


def hours(delta):
    return delta.dt.total_seconds() / 3600


def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": round(float(med), 2), "q1": round(float(q1), 2),
            "q3": round(float(q3), 2), "mean": round(float(x.mean()), 2)}


def per_seed(sessions, preds, threshold):
    rows = preds.merge(sessions[["session_id"]], on="session_id")
    rows = rows.sort_values(["session_id", "score_time"])
    alert = rows[rows["score"] >= threshold].groupby("session_id")["score_time"].min()
    out = sessions.set_index("session_id").join(alert.rename("first_alert"))
    detected = out["first_alert"].notna()
    before = detected & (out["first_alert"] < out["logoff"])
    after_event = out["first_alert"] - out["first_event"]
    d = out[detected]
    return out, {
        "threshold": threshold,
        "positive_sessions": int(len(out)),
        "detected": int(detected.sum()),
        "detection_rate": round(float(detected.mean()), 4),
        "detected_before_logoff": int(before.sum()),
        "share_of_positives_detected_before_logoff": round(float(before.mean()), 4),
        "share_of_detected_before_logoff": round(float(before[detected].mean()), 4),
        "alerted_before_first_malicious_event": int((after_event[detected] < pd.Timedelta(0)).sum()),
        "hours_start_to_alert": describe(hours(d["first_alert"] - d["start"])),
        "hours_first_event_to_alert": describe(hours(after_event[detected])),
        "hours_alert_before_logoff": describe(hours(d["logoff"] - d["first_alert"])),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--eval-json", required=True, help="eval.py single output, hourly, valid")
    p.add_argument("--preds", nargs="+", required=True, help="one validation preds file per seed")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    checksum = verify_test_checksum()
    ev = json.loads(pathlib.Path(args.eval_json).read_text())
    if ev["split"] != "valid" or ev.get("config", {}).get("session_agg", "max") != "max":
        raise SystemExit("--eval-json must be an hourly (session_agg max) validation evaluation")
    seeds = ev["session"]["per_seed"]
    if len(seeds) != len(args.preds):
        raise SystemExit(f"{len(args.preds)} preds files but {len(seeds)} seeds in the eval")

    con = duckdb.connect(config={"threads": 1})
    sessions = positive_sessions(con)
    if sessions["first_event"].isna().any():
        raise RuntimeError("a positive session has no malicious event inside it")
    sessions["hours_start_to_first_event"] = hours(sessions["first_event"] - sessions["start"])
    times = con.execute(f"""
        SELECT x.row_id, x.session_id, f.score_time FROM '{P}/splits.parquet' x
        JOIN '{P}/features.parquet' f USING (row_id) WHERE x.split = 'valid'""").df()

    results, pooled = [], []
    for seed_eval, path in zip(seeds, args.preds):
        preds = pd.read_parquet(path, columns=["row_id", "y_true", "score"]).merge(times, on="row_id")
        if len(preds) != len(times):
            raise RuntimeError(f"{path}: rows do not match the validation split")
        threshold = seed_eval["thresholds"][FPR]
        out, summary = per_seed(sessions, preds, threshold)
        # The threshold must reproduce eval.py's recall at 5% FPR for this seed.
        summary["recall_at_fpr_0.05_eval"] = round(seed_eval["recall_at_fpr_0.05"], 4)
        if abs(out["first_alert"].notna().mean() - seed_eval["recall_at_fpr_0.05"]) > 1e-9:
            raise RuntimeError(f"{path}: detection rate {summary['detection_rate']} differs from "
                               f"eval.py recall {seed_eval['recall_at_fpr_0.05']}")
        summary["preds"] = path
        results.append(summary)
        pooled.append(out.assign(seed=len(results)))

    allo = pd.concat(pooled)
    det = allo[allo["first_alert"].notna()]
    report = {
        "name": args.name, "split": "valid", "fpr_level": 0.05, "test_ids_sha256": checksum,
        "rule": "alert at the first scoring point with score >= the seed's session-level "
                "threshold at 5% FPR (hourly, from eval.py); compared with the session start, "
                "the first malicious event and the logoff (final scoring point)",
        "positive_sessions": int(len(sessions)),
        "positive_sessions_with_more_than_one_scoring_point": int((sessions["scoring_points"] > 1).sum()),
        "hours_start_to_first_malicious_event": describe(sessions["hours_start_to_first_event"]),
        "hours_first_event_to_first_positive_point": describe(
            hours(sessions["first_positive_point"] - sessions["first_event"])),
        "per_seed": results,
        "pooled_over_seeds": {
            "detection_rate": round(float(allo["first_alert"].notna().mean()), 4),
            "share_of_positives_detected_before_logoff": round(float(
                (allo["first_alert"] < allo["logoff"]).mean()), 4),
            "share_of_detected_before_logoff": round(float(
                (det["first_alert"] < det["logoff"]).mean()), 4),
            "hours_start_to_alert": describe(hours(det["first_alert"] - det["start"])),
            "hours_first_event_to_alert": describe(hours(det["first_alert"] - det["first_event"])),
            "hours_alert_before_logoff": describe(hours(det["logoff"] - det["first_alert"])),
        },
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pooled_over_seeds"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
