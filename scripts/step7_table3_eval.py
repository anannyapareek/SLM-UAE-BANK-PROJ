"""Step 7.4 (Comparison 3) evals on test, with src/eval.py only.

Single evals for XGBoost, SLM, hybrid, rules and rules + SLM, and the paired tests in
configs/step7.yaml (table3.paired), for each session aggregation (each JSON also holds the
row unit). Writes results/step7/table3/eval/<name>_test_<agg>.json and
paired_<a>_vs_<b>_test_<agg>.json. The band and every threshold were fixed before this.
Run from the repo root: python -m scripts.step7_table3_eval
"""
import pathlib
import time

import yaml

from src import eval as ev

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["table3"]
SEEDS = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["hybrid"]["seeds"]
OUT = pathlib.Path(CFG["out"]) / "eval"


def preds(name):
    folder = CFG["models"][name]
    seeds = SEEDS if name in CFG["seeded"] else [None]
    return [str(pathlib.Path(folder.format(seed=s) if s else folder) / "preds_test.parquet") for s in seeds]


def run(argv, label):
    started = time.perf_counter()
    ev.main(argv + ["--split", "test", "--allow-test"])
    print(f"{time.strftime('%H:%M:%S')} {label} done in {time.perf_counter() - started:.0f} s", flush=True)


def main():
    ev.verify_test_checksum()
    OUT.mkdir(parents=True, exist_ok=True)
    for agg in CFG["session_aggs"]:
        for name in CFG["models"]:
            run(["single", "--name", name, "--preds", *preds(name), "--session-agg", agg,
                 "--out", str(OUT / f"{name}_test_{agg}.json")], f"{name} {agg}")
        for a, b in CFG["paired"]:
            run(["paired", "--name-a", a, "--a", *preds(a), "--name-b", b, "--b", *preds(b),
                 "--session-agg", agg, "--out", str(OUT / f"paired_{a}_vs_{b}_test_{agg}.json")],
                f"{a} vs {b} {agg}")
    print("STEP7 TABLE3 EVAL FINISHED", flush=True)


if __name__ == "__main__":
    main()
