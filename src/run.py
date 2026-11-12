"""Pipeline stages. Run from the repo root: python run.py <stage> (python -m src.run also works).

python run.py all runs the full pipeline from raw data, in order, each stage in its own
process, and stops at the first failure.
"""
import argparse
import importlib
import os
import subprocess
import sys

STAGES = ["all", "data", "eval", "eval-baselines", "explore_positives", "features",
          "features_text", "labels", "leakage", "prep_report", "rules", "sessions", "splits", "xgb"]
MODULES = {"data": "src.prep", "xgb": "src.baselines.xgb", "rules": "src.baselines.rules"}
FORCE_STAGES = ["splits", "leakage"]
SAMPLE_STAGES = ["features", "features_text"]
PIPELINE = [["data"], ["sessions"], ["features"], ["features_text"], ["labels"], ["splits"],
            ["leakage"], ["xgb", "--fit"], ["rules"], ["eval-baselines"]]


def run_all(dry_run):
    for step in PIPELINE:
        command = [sys.executable, "run.py", *step]
        print(f"==> {' '.join(step)}", flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--sample-users", type=int,
                        help="build from a small copy of the data with this many users")
    parser.add_argument("--force-i-know", action="store_true",
                        help="allow replacing a frozen test_ids.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tune", action="store_true", help="xgb: run the Optuna search")
    mode.add_argument("--fit", action="store_true", help="xgb: fit seeds from xgb_best.yaml (default)")
    mode.add_argument("--score", action="store_true",
                      help="xgb, rules: predictions only on --splits (saved xgb seed models)")
    parser.add_argument("--splits", nargs="+", choices=["valid", "calib", "test"],
                        help="--score: the splits to score")
    parser.add_argument("--allow-test", action="store_true", help="--score: permit the test split")
    parser.add_argument("--overwrite-config", action="store_true",
                        help="xgb --tune: also replace configs/xgb_best.yaml")
    parser.add_argument("--dry-run", action="store_true", help="all: list the steps only")
    args, extra = parser.parse_known_args()
    if args.stage == "eval":
        from src import eval as evaluation
        evaluation.main(extra)
        return
    if extra:
        parser.error(f"unrecognised arguments: {' '.join(extra)}")
    if args.force_i_know and args.stage not in FORCE_STAGES:
        parser.error(f"--force-i-know only applies to: {', '.join(FORCE_STAGES)}")
    if (args.tune or args.fit or args.overwrite_config) and args.stage != "xgb":
        parser.error("--tune, --fit and --overwrite-config only apply to xgb")
    if args.overwrite_config and not args.tune:
        parser.error("--overwrite-config needs --tune")
    if args.score and args.stage not in ("xgb", "rules"):
        parser.error("--score only applies to xgb and rules")
    if args.score and not args.splits:
        parser.error("--score needs --splits")
    if (args.splits or args.allow_test) and not args.score:
        parser.error("--splits and --allow-test need --score")
    if args.dry_run and args.stage != "all":
        parser.error("--dry-run only applies to all")

    if args.sample_users:
        if args.stage not in SAMPLE_STAGES:
            parser.error(f"--sample-users only applies to: {', '.join(SAMPLE_STAGES)}")
        from src import sample
        os.environ["PROCESSED_DIR"] = str(sample.build(args.sample_users))
    # Imported here so the stage picks up PROCESSED_DIR.
    if args.stage == "all":
        run_all(args.dry_run)
    elif args.stage == "eval-baselines":
        from src import eval as evaluation
        evaluation.run_baselines()
    elif args.stage == "leakage":
        from src import splits
        splits.run_leakage(force=args.force_i_know)
    elif args.stage == "splits":
        from src import splits
        splits.run(force=args.force_i_know)
    elif args.stage == "xgb" and args.score:
        from src.baselines import xgb
        xgb.score_saved(args.splits, allow_test=args.allow_test)
    elif args.stage == "xgb":
        from src.baselines import xgb
        xgb.run(tune=args.tune, overwrite_config=args.overwrite_config)
    elif args.stage == "rules" and args.score:
        from src.baselines import rules
        rules.score_splits(args.splits, allow_test=args.allow_test)
    else:
        importlib.import_module(MODULES.get(args.stage, f"src.{args.stage}")).run()


if __name__ == "__main__":
    main()
