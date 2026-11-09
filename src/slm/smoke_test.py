"""Smoke test: score 10 serialised validation sessions with the base model.

Hard checks (exit 1 on failure): the answer words are single tokens, every p_yes is finite and
in [0, 1], every margin is finite, and p_yes matches the margin (sigmoid). It also greedily
generates a short completion for each session and reports whether it parses as
"<yes|no>\\nReason: ..."; an untrained base model need not produce that, so this is reported,
not enforced.

Run: python -m src.slm.smoke_test --device cuda [--model primary] [--models-dir DIR]
Writes results/smoke_test_<model>.json.
"""
import math
import re
import sys

import numpy as np

from src.slm import common, serialise
from src.slm.score import add_model_args, load_for_scoring, score_texts
from src.splits import verify_test_checksum

PARSE = re.compile(r"^\s*(yes|no)\b.*?(?:\nReason: .+)?", re.S | re.I)


def main():
    p = common.parser(__doc__)
    add_model_args(p)
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--out", help="report path (default results/smoke_test_<model>.json)")
    args = p.parse_args()
    log = common.setup_logging("smoke_test", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    common.device(args.device, log)

    rows = common.load_rows("valid", n=args.sessions, only_final=True).to_dict("records")
    model, tok, yes_id, no_id = load_for_scoring(args, log)
    texts = [serialise.scoring_prompt(r) for r in rows]
    p_yes, margin = score_texts(model, tok, texts, args.device, len(texts), yes_id, no_id)

    failures = []
    if len(rows) != args.sessions:
        failures.append(f"{len(rows)} sessions, expected {args.sessions}")
    for i, (p_, m_) in enumerate(zip(p_yes, margin)):
        if not (math.isfinite(p_) and 0.0 <= p_ <= 1.0 and math.isfinite(m_)):
            failures.append(f"row {i}: p_yes {p_}, margin {m_}")
        elif abs(p_ - 1 / (1 + math.exp(-m_))) > 1e-3:
            failures.append(f"row {i}: p_yes {p_} does not match sigmoid(margin {m_})")

    import torch
    enc = tok(texts, return_tensors="pt", padding=True).to(args.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    generated = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = [bool(PARSE.match(g)) for g in generated]
    results = []
    for r, t, g, ok, p_, m_ in zip(rows, texts, generated, parsed, p_yes, margin):
        log.info("row %s y=%d p_yes=%.4f margin=%+.3f generated=%r parses=%s",
                 r["row_id"][:8], r["y"], p_, m_, g[:80], ok)
        results.append({"row_id": r["row_id"], "y_true": int(r["y"]), "p_yes": float(p_),
                        "margin": float(m_), "generated": g, "parses": ok})
    report = {"model": common.model_entry(args.model), "device": args.device,
              "answer_ids": {"yes": yes_id, "no": no_id}, "sessions": results,
              "generation_parses": int(np.sum(parsed)), "failures": failures,
              "example_prompt": texts[0]}
    name = common.model_entry(args.model)[0].split("/")[-1]
    path = args.out or common.ROOT / f"results/smoke_test_{name}.json"
    common.write_json(path, report)
    log.info("generation parses as yes/no for %d of %d sessions (reported, not enforced)",
             sum(parsed), len(parsed))
    log.info("wrote %s", path)
    if failures:
        log.error("SMOKE TEST FAILED: %s", "; ".join(failures))
        sys.exit(1)
    log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
