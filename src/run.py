"""Pipeline stages. Run from the repo root: python run.py <stage> (python -m src.run also works)."""
import argparse
import importlib
import os

STAGES = ["explore_positives", "features", "features_text", "labels", "leakage", "prep_report",
          "rules", "sessions", "splits", "xgb"]
MODULES = {"xgb": "src.baselines.xgb", "rules": "src.baselines.rules"}
FORCE_STAGES = ["splits", "leakage"]
SAMPLE_STAGES = ["features", "features_text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--sample-users", type=int,
                        help="build from a small copy of the data with this many users")
    parser.add_argument("--force-i-know", action="store_true",
                        help="allow replacing a frozen test_ids.csv")
    args = parser.parse_args()
    if args.force_i_know and args.stage not in FORCE_STAGES:
        parser.error(f"--force-i-know only applies to: {', '.join(FORCE_STAGES)}")

    if args.sample_users:
        if args.stage not in SAMPLE_STAGES:
            parser.error(f"--sample-users only applies to: {', '.join(SAMPLE_STAGES)}")
        from src import sample
        os.environ["PROCESSED_DIR"] = str(sample.build(args.sample_users))
    # Imported here so the stage picks up PROCESSED_DIR.
    if args.stage == "leakage":
        from src import splits
        splits.run_leakage(force=args.force_i_know)
    elif args.stage == "splits":
        from src import splits
        splits.run(force=args.force_i_know)
    else:
        importlib.import_module(MODULES.get(args.stage, f"src.{args.stage}")).run()


if __name__ == "__main__":
    main()
