"""Step 8.1 rationales (decisions log 2026-11-15): label + "Reason:" text for the rationale sample.

For each row of data/step8/rationale_sample.parquet (prompts serialised on the Mac by
src.slm.step8_inputs): p_yes from the scoring forward pass (src.slm.score.score_texts, batch 1,
fp32 yes/no head), then greedy generation from the same prompt ("...\\nLabel:"), max 64 new
tokens. The adapters were trained on " yes|no\\nReason: <text><eos>", so the generation is the
label word, then "Reason:", then the rationale. label_text = the generated label line,
reason_text = the text after "Reason:" (empty if the model never wrote it), raw_text = the whole
generation, n_tokens = new tokens generated, seconds = scoring + generation for the row
(device synchronised). Model c1 = C1 seed-1 adapter (structured prompts); c2 = C2 seed-1 adapter
(text prompts), each merged into the base for scoring as in score.py.

Resume-safe: rows already in --out (same model and row_id) are skipped; the file is rewritten
atomically every --flush rows, so the run can be stopped (Ctrl+C) and restarted with the same
command. No metrics are computed.

Run (lab): set SLM_SCORE_DTYPE=fp16, then
  python -m src.slm.rationales --device cuda --model primary --models-dir models \\
      --c1-adapter runs\\grid_qwen15_r8_lr1e-4\\adapter --c2-adapter runs\\c2_qwen15_r8_lr1e-4_s1\\adapter
  [--models c1]   (C1 only)
Writes results/step8/rationales.parquet.
"""
import argparse
import os
import pathlib
import time

import pandas as pd

from src.slm import common
from src.slm.score import load_for_scoring, score_texts
from src.splits import verify_test_checksum

INPUTS = common.ROOT / "data/step8"
COLUMNS = ["session_id", "row_id", "model", "y_true", "p_yes", "label_text", "reason_text",
           "n_tokens", "seconds", "raw_text"]


def load_model(args, adapter, log):
    """Base model + LoRA adapter merged (score.load_for_scoring), as for scoring."""
    ns = argparse.Namespace(model=args.model, models_dir=args.models_dir, adapter=adapter,
                            device=args.device)
    return load_for_scoring(ns, log)


def parse(generated):
    """(label_text, reason_text) from ' yes\\nReason: ...'."""
    head, _, rest = generated.partition("\n")
    label = head.strip()
    reason = rest.split("Reason:", 1)[1].strip() if "Reason:" in rest else ""
    return label, reason


def generate(model, tok, text, device_name, max_new_tokens):
    """Greedy continuation of one prompt: (decoded new text, number of new tokens)."""
    import torch
    enc = tok([text], return_tensors="pt", add_special_tokens=True).to(device_name)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = out[0, enc["input_ids"].shape[1]:]          # batch 1: no padding; eos counts if generated
    return tok.decode(new, skip_special_tokens=True), int(len(new))


def score_and_explain(model, tok, text, device_name, yes_id, no_id, max_new_tokens):
    """One row, batch 1: p_yes, generation, wall seconds (synchronised before and after)."""
    common.sync(device_name)
    start = time.perf_counter()
    p_yes, _ = score_texts(model, tok, [text], device_name, 1, yes_id, no_id)
    generated, n = generate(model, tok, text, device_name, max_new_tokens)
    common.sync(device_name)
    return float(p_yes[0]), generated, n, time.perf_counter() - start


def write(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    p = common.parser(__doc__)
    p.add_argument("--model", default="primary")
    p.add_argument("--models-dir")
    p.add_argument("--c1-adapter")
    p.add_argument("--c2-adapter")
    p.add_argument("--models", nargs="+", default=["c1", "c2"], choices=["c1", "c2"])
    p.add_argument("--inputs", default=str(INPUTS))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--flush", type=int, default=10, help="rewrite the output every N rows")
    p.add_argument("--limit", type=int, help="first N rows per model only (dry runs)")
    p.add_argument("--out", default=str(common.ROOT / "results/step8/rationales.parquet"))
    args = p.parse_args()
    log = common.setup_logging("rationales", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    if args.device == "cuda" and os.environ.get("SLM_SCORE_DTYPE") != "fp16":
        raise SystemExit("set SLM_SCORE_DTYPE=fp16 (scoring protocol: fp16, merged adapter)")
    common.device(args.device, log)
    sample = pd.read_parquet(pathlib.Path(args.inputs) / "rationale_sample.parquet")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=COLUMNS)
    log.info("%d rows already in %s", len(done), out)
    for key in args.models:
        adapter = getattr(args, f"{key}_adapter")
        if not adapter:
            raise SystemExit(f"--{key}-adapter is required for model {key}")
        rows = sample[sample["model"] == key]
        if args.limit:
            rows = rows.head(args.limit)
        have = set(done.loc[done["model"] == key, "row_id"])
        todo = rows[~rows["row_id"].isin(have)]
        log.info("%s: %d rows, %d done, %d to go", key, len(rows), len(rows) - len(todo), len(todo))
        if todo.empty:
            continue
        model, tok, yes_id, no_id = load_model(args, adapter, log)
        new = []
        started = time.perf_counter()
        for i, r in enumerate(todo.itertuples(), 1):
            p_yes, generated, n, seconds = score_and_explain(model, tok, r.prompt, args.device, yes_id, no_id,
                                                             args.max_new_tokens)
            label, reason = parse(generated)
            new.append({"session_id": r.session_id, "row_id": r.row_id, "model": key, "y_true": int(r.y_true),
                        "p_yes": p_yes, "label_text": label, "reason_text": reason, "n_tokens": n,
                        "seconds": seconds, "raw_text": generated})
            if i % args.flush == 0 or i == len(todo):
                done = pd.concat([done, pd.DataFrame(new)], ignore_index=True)[COLUMNS]
                write(done, out)
                new = []
                log.info("%s: %d / %d rows (%.1f s); last: %r", key, i, len(todo),
                         time.perf_counter() - started, generated[:80])
        del model
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()
    log.info("wrote %s (%d rows)", out, len(done))


if __name__ == "__main__":
    main()
