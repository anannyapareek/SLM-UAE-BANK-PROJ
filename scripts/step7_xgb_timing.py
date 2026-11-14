"""XGBoost v3 scoring throughput for the Table 3 compute line (Step 7.4).

Times predict_proba of each saved seed model on the validation rows (features already in
memory; loading excluded), 3 repeats, and writes results/step7/xgb_timing.json. Validation
only: throughput does not depend on the split. Run: python -m scripts.step7_xgb_timing
"""
import json
import pathlib
import platform
import time

import duckdb
import numpy as np
import xgboost as xgb
import yaml

from src.baselines import xgb as base

OUT = pathlib.Path("results/step7/xgb_timing.json")


def main():
    best = yaml.safe_load(base.BEST_CONFIG.read_text())
    names = base.feature_names()
    categories = {c: list(v) for c, v in best["categories"].items()}
    frame = base.load_split(duckdb.connect(config={"threads": 1}), names, "valid", categories)
    X = frame[names]
    per_seed = []
    for seed in best["seeds"]:
        model = xgb.XGBClassifier()
        model.load_model(base.RESULTS / f"{best['run_prefix']}_s{seed}" / "model.json")
        model.predict_proba(X.head(1000))                    # warm-up
        seconds = []
        for _ in range(3):
            t = time.perf_counter()
            model.predict_proba(X)
            seconds.append(time.perf_counter() - t)
        per_seed.append({"seed": seed, "seconds": seconds, "rows_per_second": len(X) / float(np.median(seconds))})
        print(per_seed[-1], flush=True)
    result = {"rows": len(X), "split": "valid", "n_jobs": model.get_params().get("n_jobs"),
              "machine": f"Mac ({platform.machine()}, {platform.system()}), CPU",
              "per_seed": per_seed,
              "rows_per_second_median": float(np.median([s["rows_per_second"] for s in per_seed]))}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result["rows_per_second_median"])


if __name__ == "__main__":
    main()
