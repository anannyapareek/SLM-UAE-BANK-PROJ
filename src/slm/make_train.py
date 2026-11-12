"""Build the LoRA training files (runbook 5.1).

- train.jsonl: train split only; every positive row and neg_per_pos negatives per positive,
  sampled with the configured seed (downsampling for training only).
- valid.jsonl: valid_loss_rows random validation rows at the natural positive rate, for the
  training loss only (never rebalanced; model selection scores the full validation split).
Each line: {"row_id", "prompt": serialise.scoring_prompt(row), "completion": serialise.target(row, y)}.

--text (Comparison 2, runbook 6.2): the same rows, seed and 1:10 negatives, but each prompt
carries the row's text block (serialise.text_scoring_prompt), read from the text-blocks file
built by src.slm.text_blocks (on a machine with features_text.parquet). Writes to
data/slm_text by default. The completion is unchanged.

Run: python -m src.slm.make_train [--out-dir data/slm]
     python -m src.slm.make_train --text [--text-blocks data/processed/slm_text_blocks.parquet]
"""
import json
import pathlib

import pandas as pd

from src.slm import common, serialise
from src.splits import verify_test_checksum


def select_rows(train_rows=None, valid_rows=None):
    """The training and loss-validation rows, identical for the structured and text sets."""
    cfg = common.config()["training"]
    columns = ["row_id", "session_id", "score_time"] + list(serialise.USED)
    train = common.load_rows("train", columns=columns)
    pos, neg = train[train["y"] == 1], train[train["y"] == 0]
    n_neg = min(len(neg), cfg["neg_per_pos"] * len(pos))
    neg = neg.sample(n=n_neg, random_state=cfg["seed"])
    train = pd.concat([pos, neg]).sample(frac=1, random_state=cfg["seed"])
    if train_rows:
        train = train.head(train_rows)
    valid = common.load_rows("valid", n=valid_rows or cfg["valid_loss_rows"], seed=cfg["seed"],
                             columns=columns)
    return train, valid


def write_jsonl(path, frame, blocks=None):
    with open(path, "w") as f:
        for row in frame.to_dict("records"):
            if blocks is None:
                text = serialise.scoring_prompt(row)
            else:
                text = serialise.text_scoring_prompt(row, blocks[row["row_id"]])
            f.write(json.dumps({"row_id": row["row_id"], "prompt": text,
                                "completion": serialise.target(row, int(row["y"]))}) + "\n")


def load_blocks(path, row_ids):
    """row_id -> text block for these rows; stops if the file or any row is missing."""
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"text blocks file not found: {path}. Build it with "
                         "python -m src.slm.text_blocks on a machine with "
                         "data/processed/features_text.parquet, then copy it here.")
    frame = pd.read_parquet(path, columns=["row_id", "block"])
    blocks = dict(zip(frame["row_id"], frame["block"]))
    missing = [r for r in row_ids if r not in blocks]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no text block in {path} "
                         f"(e.g. {missing[:3]}); rebuild it with src.slm.text_blocks")
    return blocks


def build(out_dir, log, train_rows=None, valid_rows=None, text_blocks=None):
    cfg = common.config()["training"]
    train, valid = select_rows(train_rows, valid_rows)
    blocks = None
    if text_blocks:
        blocks = load_blocks(text_blocks, list(train["row_id"]) + list(valid["row_id"]))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train, blocks)
    write_jsonl(out_dir / "valid.jsonl", valid, blocks)
    summary = {"train_rows": len(train), "train_positives": int(train["y"].sum()),
               "neg_per_pos": cfg["neg_per_pos"], "valid_rows": len(valid),
               "valid_positives": int(valid["y"].sum()), "seed": cfg["seed"],
               "text": bool(text_blocks)}
    if text_blocks:
        summary["text_blocks"] = str(text_blocks)
    common.write_json(out_dir / "summary.json", summary)
    log.info("wrote %s: %s", out_dir, summary)
    return summary


def main():
    p = common.parser(__doc__, device=False)
    p.add_argument("--out-dir", help="default data/slm, or data/slm_text with --text")
    p.add_argument("--text", action="store_true", help="add the text block (Comparison 2)")
    p.add_argument("--text-blocks", default=str(common.ROOT / common.config()["text"]["blocks"]))
    p.add_argument("--train-rows", type=int, help="cap (for tests)")
    p.add_argument("--valid-rows", type=int, help="override valid_loss_rows (for tests)")
    args = p.parse_args()
    out_dir = args.out_dir or str(common.ROOT / ("data/slm_text" if args.text else "data/slm"))
    log = common.setup_logging("make_train", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    build(out_dir, log, args.train_rows, args.valid_rows,
          args.text_blocks if args.text else None)


if __name__ == "__main__":
    main()
