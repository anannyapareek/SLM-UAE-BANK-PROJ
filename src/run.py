"""Pipeline entry point. Run from the repo root: python -m src.run <stage>"""
import argparse
import importlib
import os

STAGES = ["features", "features_text", "prep_report", "sessions"]
SAMPLE_STAGES = ["features", "features_text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--sample-users", type=int,
                        help="build from a small copy of the data with this many users")
    args = parser.parse_args()

    if args.sample_users:
        if args.stage not in SAMPLE_STAGES:
            parser.error(f"--sample-users only applies to: {', '.join(SAMPLE_STAGES)}")
        from src import sample
        os.environ["PROCESSED_DIR"] = str(sample.build(args.sample_users))
    # Imported here so the stage picks up PROCESSED_DIR.
    importlib.import_module(f"src.{args.stage}").run()


if __name__ == "__main__":
    main()
