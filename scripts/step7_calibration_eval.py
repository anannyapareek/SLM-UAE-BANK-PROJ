"""Step 7.1: evaluate calibration before and after, on calib and test, with src/eval.py only.

For every model: the uncalibrated predictions and each fit in results/step7/calibration/
(temperature, Platt; session and row fits for C1), with `eval single --calibration
--bss-ref-split calib` (Brier skill score against the calib base rate and the split's own)
(C1 hourly files; C2 logoff files with --rows final). Writes
results/step7/calibration/<run_id>/metrics_<split>.json (<run_id> = <model>_uncalibrated for
the originals). Thresholds and fits come from calib; nothing is fitted here.

Run from the repo root: python -m scripts.step7_calibration_eval
"""
import pathlib
import time

import yaml

from src import eval as ev

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["calibration"]
ROOT = pathlib.Path(CFG["out"])
SPLITS = ("calib", "test")


def runs():
    for model in CFG["models"]:
        suffix, key = model["suffix"], model["key"]
        yield (f"{key}_uncalibrated", model,
               lambda split, m=model: [str(pathlib.Path(m["folder"].format(seed=s)) /
                                           f"preds_{split}{m['suffix']}.parquet") for s in CFG["seeds"]])
        for unit in model["fit_units"]:
            for method in CFG["methods"]:
                run_id = f"{key}_{unit}fit_{method}"
                yield (run_id, model,
                       lambda split, r=run_id, sfx=suffix: [str(ROOT / r / f"s{s}" / f"preds_{split}{sfx}.parquet")
                                                            for s in CFG["seeds"]])


def main():
    for run_id, model, files in runs():
        for split in SPLITS:
            out = ROOT / run_id / f"metrics_{split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            argv = ["single", "--name", run_id, "--split", split, "--calibration", "--bss-ref-split", "calib",
                    "--preds", *files(split), "--out", str(out)]
            if split == "test":
                argv.append("--allow-test")
            if model["suffix"] == "_final":
                argv += ["--rows", "final"]
            started = time.perf_counter()
            ev.main(argv)
            print(f"{time.strftime('%H:%M:%S')} {run_id} {split} done in "
                  f"{time.perf_counter() - started:.0f} s", flush=True)
    print("STEP7 CALIBRATION EVAL FINISHED", flush=True)


if __name__ == "__main__":
    main()
