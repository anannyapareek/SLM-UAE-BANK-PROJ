"""Step 8.1 XGBoost v3 seed-1 latency on the lab CPU (decisions log 2026-11-15).

Inputs (data/step8, built on the Mac by src.slm.step8_inputs): xgb_v3_s1_model.json and
xgb_features.parquet (1,000 test logoff rows, the latency sample, with the training
categorical encoding and the model's stored test scores as expected_score).
Check first: Booster.inplace_predict (trees up to best_iteration, as XGBClassifier) on the 1,000 rows must reproduce expected_score (max abs
difference <= 1e-6), or the run stops. Then
  batch 1: 20 warm-up calls, then inplace_predict on each row's pre-built one-row frame;
           median, p95, mean per call;
  full batch: inplace_predict on all 1,000 rows, --repeats times after one warm-up; median
           seconds -> rows per second. Secondary: the same rows tiled x100 (100,000 rows), 5
           repeats, for a large-batch throughput comparable with the Step 7 Mac timing.
Default threading (all cores) is recorded. Loading is excluded. No labels; no metrics.
This replaces the Mac timing (results/step7/xgb_timing.json) in the hybrid compute line.

Run (lab): python -m src.slm.xgb_latency     -> results/step8/latency_xgb_lab.json
"""
import argparse
import os
import pathlib
import platform
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
MACHINE = "Lab PC (RTX 4500 Ada, Windows 11)"


def summary(seconds):
    s = np.asarray(seconds)
    return {"n": int(len(s)), "median_ms": float(np.median(s) * 1e3), "p95_ms": float(np.percentile(s, 95) * 1e3),
            "mean_ms": float(s.mean() * 1e3), "min_ms": float(s.min() * 1e3), "max_ms": float(s.max() * 1e3)}


def main():
    import xgboost as xgb
    from src.splits import verify_test_checksum

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", default=str(ROOT / "data/step8"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--tile", type=int, default=100, help="secondary: full batch on the rows tiled this many times")
    p.add_argument("--machine", default=MACHINE)
    p.add_argument("--out", default=str(ROOT / "results/step8/latency_xgb_lab.json"))
    args = p.parse_args()
    checksum = verify_test_checksum()
    inputs = pathlib.Path(args.inputs)
    frame = pd.read_parquet(inputs / "xgb_features.parquet")
    expected = frame.pop("expected_score").to_numpy()
    frame = frame.drop(columns=["row_id"])
    booster = xgb.Booster()
    booster.load_model(str(inputs / "xgb_v3_s1_model.json"))
    if booster.feature_names and list(frame.columns) != list(booster.feature_names):
        raise SystemExit("feature columns differ from the model's feature names")
    best = booster.attributes().get("best_iteration")
    if best is None:
        raise SystemExit("model.json has no best_iteration (early stopping); cannot match XGBClassifier")
    trees = (0, int(best) + 1)                     # as XGBClassifier.predict_proba (early stopping)

    def predict(x):
        return booster.inplace_predict(x, iteration_range=trees)

    got = predict(frame)
    diff = float(np.max(np.abs(got - expected)))
    print(f"check: max |inplace_predict - stored test score| = {diff:.2e} over {len(frame)} rows", flush=True)
    if diff > 1e-6:
        raise SystemExit("XGBoost predictions do not reproduce the stored scores; stopping")

    singles = [frame.iloc[[i]] for i in range(len(frame))]
    for x in singles[:args.warmup]:
        predict(x)
    seconds = []
    for x in singles:
        start = time.perf_counter()
        predict(x)
        seconds.append(time.perf_counter() - start)
    predict(frame)                                                         # warm-up
    full = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        predict(frame)
        full.append(time.perf_counter() - start)
    median = float(np.median(full))
    tiled = pd.concat([frame] * args.tile, ignore_index=True)              # same rows, larger batch
    predict(tiled)
    tiled_s = []
    for _ in range(5):
        start = time.perf_counter()
        predict(tiled)
        tiled_s.append(time.perf_counter() - start)
    config = booster.save_config()
    result = {"machine": args.machine, "test_ids_sha256": checksum, "model": "xgb_struct_v3 seed 1",
              "inputs": str(inputs / "xgb_features.parquet"), "rows": len(frame), "features": frame.shape[1],
              "check_max_abs_diff": diff, "method": "Booster.inplace_predict (features in memory; loading excluded)",
              "iteration_range": list(trees), "boosted_rounds": booster.num_boosted_rounds(),
              "batch1": summary(seconds) | {"warmup_calls": args.warmup},
              "full_batch": {"rows": len(frame), "repeats": args.repeats, "seconds": full,
                             "median_seconds": median, "rows_per_second": len(frame) / median},
              "full_batch_tiled": {"rows": len(tiled), "note": f"the 1,000 rows tiled x{args.tile}: throughput at a "
                                   "large batch, comparable with results/step7/xgb_timing.json (336,737 rows)",
                                   "repeats": 5, "seconds": tiled_s, "median_seconds": float(np.median(tiled_s)),
                                   "rows_per_second": len(tiled) / float(np.median(tiled_s))},
              "environment": {"xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                              "python": platform.python_version(), "platform": platform.platform(),
                              "cpu": platform.processor(), "cpu_count": os.cpu_count(),
                              "nthread": __import__("json").loads(config)["learner"]["generic_param"].get("nthread")}}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(result, indent=2) + "\n")
    print(f"batch 1: median {result['batch1']['median_ms']:.3f} ms, p95 {result['batch1']['p95_ms']:.3f} ms; "
          f"full batch {result['full_batch']['rows_per_second']:,.0f} rows/s "
          f"({result['full_batch_tiled']['rows_per_second']:,.0f} rows/s at {len(tiled):,} rows); wrote {out}")


if __name__ == "__main__":
    main()
