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


def check_text(args, log):
    """Every text prompt (structured prompt + text block + "\nLabel:") plus the longest target
    must fit lora.max_length, with the answer token unmerged; text models: primary."""
    import duckdb
    import pandas as pd
    cfg = common.config()
    max_length = cfg["lora"]["max_length"]
    models = args.models if args.models != ["primary", "second", "fallback"] else ["primary"]
    blocks = pd.read_parquet(args.text_blocks, columns=["row_id", "block", "sets"])
    if args.rows and args.rows < len(blocks):
        blocks = blocks.sample(n=args.rows, random_state=cfg["training"]["seed"])
    con = duckdb.connect(config={"threads": 1})
    con.register("wanted", blocks[["row_id"]])
    cols = ", ".join(f'f."{c}"' for c in serialise.USED)
    feats = con.execute(f"""SELECT f.row_id, {cols} FROM '{common.P}/features.parquet' f
                            JOIN wanted USING (row_id)""").df()
    rows = blocks.merge(feats, on="row_id").to_dict("records")
    prompts = [serialise.text_scoring_prompt(r, r["block"]) for r in rows]
    targets = [serialise.target(r, label) for r in rows for label in (1, 0)]
    log.info("%d text prompts from %s", len(prompts), args.text_blocks)
    report, failed = {"rows": len(prompts), "max_length": max_length, "models": {}}, []
    for name in models:
        tok = common.load_tokenizer(name, args.models_dir)
        model_id, revision = common.model_entry(name)
        yes = tok.encode(" yes", add_special_tokens=False)
        no = tok.encode(" no", add_special_tokens=False)
        boundary = sum(tok.encode(t + w, add_special_tokens=False)
                       != tok.encode(t, add_special_tokens=False) + ids
                       for t in prompts[:200] for w, ids in ((" yes", yes), (" no", no)))
        lengths = np.array([len(i) for i in tok(prompts, add_special_tokens=True)["input_ids"]])
        target_max = max(len(i) for i in tok(targets, add_special_tokens=False)["input_ids"]) + 1
        over = int((lengths + target_max > max_length).sum())
        r = {"id": model_id, "revision": revision, "boundary_mismatches": int(boundary),
             "prompt_tokens": {"median": float(np.median(lengths)),
                               "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max())},
             "target_tokens_max_incl_eos": target_max,
             "max_prompt_plus_target": int(lengths.max()) + target_max,
             "prompts_over_limit_with_longest_target": over,
             "passes": bool(over == 0 and boundary == 0)}
        log.info("%s text prompt tokens median %.0f, p95 %.0f, max %d; target max %d; "
                 "max prompt + target = %d (limit %d); over limit %d; boundary mismatches %d %s",
                 model_id, np.median(lengths), np.percentile(lengths, 95), lengths.max(),
                 target_max, lengths.max() + target_max, max_length, over, boundary,
                 "PASS" if r["passes"] else "FAIL")
        if not r["passes"]:
            failed.append(f"{model_id}: {over} text prompts over {max_length} tokens or "
                          f"{boundary} boundary mismatches")
        report["models"][name] = r
    report["passed"], report["failures"] = not failed, failed
    out = args.out or str(common.ROOT / "results/token_check_text.json")
    common.write_json(out, report)
    log.info("wrote %s", out)
    if failed:
        log.error("FAILED: %s", "; ".join(failed))
        sys.exit(1)
    log.info("all text checks passed")


def main():
    p = common.parser(__doc__, device=False)
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--models", nargs="+", default=["primary", "second", "fallback"])
    p.add_argument("--models-dir", help="local model copies (scripts/download_models.py)")
    p.add_argument("--out", help="default results/token_check.json (text: token_check_text.json)")
    p.add_argument("--text", action="store_true",
                   help="check the Comparison 2 text prompts (from --text-blocks) instead")
    p.add_argument("--text-blocks", default=str(common.ROOT / common.config()["text"]["blocks"]))
    args = p.parse_args()
    log = common.setup_logging("check_tokens", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    if args.text:
        return check_text(args, log)
    args.out = args.out or str(common.ROOT / "results/token_check.json")
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
