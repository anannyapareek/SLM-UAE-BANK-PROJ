"""Tokenizer checks (Step 4.3), CPU only: answer tokens and the 512-token budget.

For each model in configs/slm.yaml (primary, second, fallback):
- logs the ids of "yes" and "no", with and without a leading space; fails if any is more
  than one token;
- checks that "<scoring prompt> yes" tokenizes as the prompt followed by the " yes" id, so
  the scorer reads the right position;
- measures scoring-prompt tokens (median, p95, max) and target tokens (max, incl. EOS) on
  random train rows; passes if p95(prompt) + text budget + max(target) <= max_length.

Run: python -m src.slm.check_tokens [--rows 5000] [--models-dir DIR]
Writes results/token_check.json.
"""
import sys

import numpy as np

from src.slm import common, serialise
from src.splits import verify_test_checksum


def main():
    p = common.parser(__doc__, device=False)
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--models", nargs="+", default=["primary", "second", "fallback"])
    p.add_argument("--models-dir", help="local model copies (scripts/download_models.py)")
    p.add_argument("--out", default=str(common.ROOT / "results/token_check.json"))
    args = p.parse_args()
    log = common.setup_logging("check_tokens", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    cfg = common.config()
    budget, max_length = cfg["prompt"]["text_budget_tokens"], cfg["lora"]["max_length"]

    rows = common.load_rows("train", n=args.rows, seed=cfg["training"]["seed"])
    rows = rows.to_dict("records")
    prompts = [serialise.scoring_prompt(r) for r in rows]
    targets = [serialise.target(r, label) for r in rows for label in (1, 0)]
    log.info("%d train rows sampled (seed %d)", len(rows), cfg["training"]["seed"])

    report, failed = {"rows": len(rows), "text_budget": budget, "max_length": max_length,
                      "models": {}}, []
    for name in args.models:
        model_id, revision = common.model_entry(name)
        tok = common.load_tokenizer(name, args.models_dir)
        r = {"id": model_id, "revision": revision, "answer_ids": {}}
        for word in ("yes", "no", " yes", " no"):
            ids = tok.encode(word, add_special_tokens=False)
            r["answer_ids"][word] = ids
            log.info("%s %r -> %s", model_id, word, ids)
            if len(ids) != 1:
                failed.append(f"{model_id}: {word!r} is {len(ids)} tokens")
        boundary = []
        for prompt in prompts[:200]:
            base = tok.encode(prompt, add_special_tokens=False)
            for word in (" yes", " no"):
                if tok.encode(prompt + word, add_special_tokens=False) != base + r["answer_ids"][word]:
                    boundary.append(word)
        r["boundary_mismatches"] = len(boundary)
        if boundary:
            failed.append(f"{model_id}: answer token merges with the prompt in {len(boundary)} cases")

        lengths = np.array([len(tok.encode(t, add_special_tokens=True)) for t in prompts])
        target_max = max(len(tok.encode(t, add_special_tokens=False)) for t in targets) + 1
        p95 = float(np.percentile(lengths, 95))
        total = p95 + budget + target_max
        r.update(prompt_tokens={"median": float(np.median(lengths)), "p95": p95,
                                "max": int(lengths.max())},
                 target_tokens_max_incl_eos=target_max, p95_plus_budget_plus_target=total,
                 passes_budget=bool(total <= max_length))
        log.info("%s prompt tokens median %.0f, p95 %.0f, max %d; target max %d (incl. EOS); "
                 "p95 + %d + target = %.0f (limit %d) %s", model_id, np.median(lengths), p95,
                 lengths.max(), target_max, budget, total, max_length,
                 "PASS" if total <= max_length else "FAIL")
        if total > max_length:
            failed.append(f"{model_id}: p95 prompt + budget + target = {total:.0f} > {max_length}")
        report["models"][name] = r

    report["passed"] = not failed
    report["failures"] = failed
    common.write_json(args.out, report)
    log.info("wrote %s", args.out)
    if failed:
        log.error("FAILED: %s", "; ".join(failed))
        sys.exit(1)
    log.info("all checks passed")


if __name__ == "__main__":
    main()
