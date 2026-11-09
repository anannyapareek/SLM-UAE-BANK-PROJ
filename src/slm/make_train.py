"""Build the LoRA training files (runbook 5.1).

- train.jsonl: train split only; every positive row and neg_per_pos negatives per positive,
  sampled with the configured seed (downsampling for training only).
- valid.jsonl: valid_loss_rows random validation rows at the natural positive rate, for the
  training loss only (never rebalanced; model selection scores the full validation split).
Each line: {"row_id", "prompt": serialise.scoring_prompt(row), "completion": serialise.target(row, y)}.

Run: python -m src.slm.make_train [--out-dir data/slm]
"""
import json
import pathlib

import pandas as pd

from src.slm import common, serialise
from src.splits import verify_test_checksum


def write_jsonl(path, frame):
    with open(path, "w") as f:
        for row in frame.to_dict("records"):
            f.write(json.dumps({"row_id": row["row_id"], "prompt": serialise.scoring_prompt(row),
                                "completion": serialise.target(row, int(row["y"]))}) + "\n")


def build(out_dir, log, train_rows=None, valid_rows=None):
    cfg = common.config()["training"]
    columns = ["row_id"] + list(serialise.USED)
    train = common.load_rows("train", columns=columns)
    pos, neg = train[train["y"] == 1], train[train["y"] == 0]
    n_neg = min(len(neg), cfg["neg_per_pos"] * len(pos))
    neg = neg.sample(n=n_neg, random_state=cfg["seed"])
    train = pd.concat([pos, neg]).sample(frac=1, random_state=cfg["seed"])
    if train_rows:
        train = train.head(train_rows)
    valid = common.load_rows("valid", n=valid_rows or cfg["valid_loss_rows"], seed=cfg["seed"],
                             columns=columns)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    summary = {"train_rows": len(train), "train_positives": int(train["y"].sum()),
               "neg_per_pos": cfg["neg_per_pos"], "valid_rows": len(valid),
               "valid_positives": int(valid["y"].sum()), "seed": cfg["seed"]}
    common.write_json(out_dir / "summary.json", summary)
    log.info("wrote %s: %s", out_dir, summary)
    return summary


def main():
    p = common.parser(__doc__, device=False)
    p.add_argument("--out-dir", default=str(common.ROOT / "data/slm"))
    p.add_argument("--train-rows", type=int, help="cap (for tests)")
    p.add_argument("--valid-rows", type=int, help="override valid_loss_rows (for tests)")
    args = p.parse_args()
    log = common.setup_logging("make_train", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    build(args.out_dir, log, args.train_rows, args.valid_rows)


if __name__ == "__main__":
    main()
