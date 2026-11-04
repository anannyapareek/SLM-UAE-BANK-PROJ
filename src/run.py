"""Pipeline entry point. Run from the repo root: python -m src.run <stage>"""
import argparse

from src import features, features_text, sessions

STAGES = {
    "features": features.run,
    "features_text": features_text.run,
    "sessions": sessions.run,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    STAGES[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
