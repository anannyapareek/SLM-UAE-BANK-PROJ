"""Time scoring and project the GPU-hours of the full scoring workload.

Times 1,000 validation rows at batch sizes 8, 16, 32 and 64 (device synchronised around each
timing, one warm-up batch first) and records peak memory. Each batch size is timed twice:
- comparison 1: the structured prompt only;
- comparison 2: the structured prompt plus a 256-token text block (synthetic filler of the
  same length as the runbook 6.2 budget), since the text makes those prompts longer.
The fastest batch size that fits is then multiplied out to the full workload:
(valid + calib + test rows) x 3 seeds, for comparison 1 and comparison 2, for hourly scoring
(every row) and for logoff-only scoring (final rows). Row counts come from splits.parquet;
no labels are read.

Run: python -m src.slm.time_scoring --device cuda [--model primary] [--models-dir DIR]
Writes results/gpu_timing.json.
"""
import sys
import time

from src.slm import common, serialise
from src.slm.score import add_model_args, load_for_scoring, score_texts
from src.splits import verify_test_checksum

FILLER = " routine activity"


def filler(tok, n_tokens):
    """A text block of exactly n_tokens tokens (as near as the tokenizer allows)."""
    text = "\nText:" + FILLER * n_tokens
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def peak_memory_gb(device_name):
    import torch
    if device_name == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device_name == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    import resource                       # cpu: peak resident memory of this process
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e9 if sys.platform == "darwin" else 1e6)


def reset_peak(device_name):
    import torch
    if device_name == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device_name == "mps":
        torch.mps.empty_cache()


def main():
    cfg = common.config()
    p = common.parser(__doc__)
    add_model_args(p)
    p.add_argument("--rows", type=int, default=cfg["timing"]["rows"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=cfg["timing"]["batch_sizes"])
    p.add_argument("--out", default=str(common.ROOT / "results/gpu_timing.json"))
    args = p.parse_args()
    log = common.setup_logging("time_scoring", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    common.device(args.device, log)
    import torch

    columns = ["row_id"] + list(serialise.USED)
    rows = common.load_rows("valid", n=args.rows, columns=columns).to_dict("records")
    model, tok, yes_id, no_id = load_for_scoring(args, log)
    structured = [serialise.prompt(r) for r in rows]
    text_block = filler(tok, cfg["prompt"]["text_budget_tokens"])
    variants = {"comparison_1": [t + serialise.SUFFIX for t in structured],
                "comparison_2": [t + text_block + serialise.SUFFIX for t in structured]}

    timings = {}
    for variant, texts in variants.items():
        lengths = [len(tok.encode(t)) for t in texts]
        timings[variant] = {"mean_prompt_tokens": sum(lengths) / len(lengths), "batch": {}}
        for bs in args.batch_sizes:
            reset_peak(args.device)
            try:
                score_texts(model, tok, texts[:bs], args.device, bs, yes_id, no_id)  # warm-up
                common.sync(args.device)
                start = time.perf_counter()
                score_texts(model, tok, texts, args.device, bs, yes_id, no_id)
                common.sync(args.device)
                seconds = time.perf_counter() - start
            except torch.OutOfMemoryError as e:
                log.warning("%s batch %d: out of memory (%s)", variant, bs, e)
                timings[variant]["batch"][bs] = {"oom": True}
                continue
            peak = peak_memory_gb(args.device)
            timings[variant]["batch"][bs] = {"seconds": round(seconds, 3),
                                             "rows_per_second": round(len(texts) / seconds, 2),
                                             "peak_memory_gb": peak and round(peak, 2)}
            log.info("%s batch %3d: %.2f s for %d rows (%.1f rows/s), peak memory %s GB",
                     variant, bs, seconds, len(texts), len(texts) / seconds,
                     peak and round(peak, 2))

    counts = common.split_counts()
    seeds = cfg["timing"]["seeds"]
    workload = {}
    for mode in ("hourly", "logoff_only"):
        n = sum(counts[s][mode] for s in ("valid", "calib", "test"))
        entry = {"rows_valid_calib_test": n, "seeds": seeds, "per_comparison": {}}
        total = 0.0
        for variant in variants:
            fits = {bs: t for bs, t in timings[variant]["batch"].items() if not t.get("oom")}
            if not fits:
                entry["per_comparison"][variant] = "no batch size fits"
                continue
            best = max(fits, key=lambda bs: fits[bs]["rows_per_second"])
            hours = n * seeds / fits[best]["rows_per_second"] / 3600
            total += hours
            entry["per_comparison"][variant] = {"best_batch_size": best,
                                                "rows_per_second": fits[best]["rows_per_second"],
                                                "gpu_hours": round(hours, 2)}
        entry["gpu_hours_total"] = round(total, 2)
        workload[mode] = entry
        log.info("%s: %d rows x %d seeds x 2 comparisons -> %.2f GPU-hours (%s)", mode, n,
                 seeds, total, {v: e if isinstance(e, str) else e["gpu_hours"]
                                for v, e in entry["per_comparison"].items()})

    report = {"model": common.model_entry(args.model), "device": args.device,
              "dtype": str(next(model.parameters()).dtype), "rows_timed": len(rows),
              "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "timings": timings, "row_counts": counts, "workload": workload,
              "note": ("Scoring only, base model; LoRA adapters add little. Comparison 2 uses "
                       "a synthetic 256-token text block. LoRA grid validation scoring and "
                       "training time are not included.")}
    common.write_json(args.out, report)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
