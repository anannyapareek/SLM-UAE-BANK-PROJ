"""Step 8.1 SLM latency (decisions log 2026-11-15), on data/step8/latency_sample.parquet.

Configurations (base model + LoRA adapter merged, SLM_SCORE_DTYPE=fp16 on CUDA, as scoring):
  a  score only, structured prompts, C1 seed-1 adapter, batch 1, all sample rows
  b  score only, text prompts, C2 seed-1 adapter, batch 1, all sample rows
  c  score + rationale (greedy generation from "...Label:" through "Reason:", max 48 new
     tokens), structured prompts, C1 adapter, batch 1, first 150 rows
  a_b16, b_b16  throughput confirmation: score only at batch 16 over all sample rows
Batch-1 protocol: 20 warm-up calls (not timed), then one call per row with the device
synchronised before and after each call; median, p95, mean of the per-call seconds. A call is
src.slm.score.score_texts on one prompt (tokenise, forward, fp32 yes/no head, copy to host);
c adds src.slm.rationales.generate. Peak GPU memory = torch.cuda.max_memory_allocated, reset
before each configuration (weights included); RSS from psutil after each configuration (and
the Windows peak working set when available). No labels are read; no metrics are computed.

Run (lab): set SLM_SCORE_DTYPE=fp16, then
  python -m src.slm.latency --device cuda --model primary --models-dir models \\
      --c1-adapter runs\\grid_qwen15_r8_lr1e-4\\adapter --c2-adapter runs\\c2_qwen15_r8_lr1e-4_s1\\adapter
Writes results/step8/latency_gpu.json.
"""
import os
import pathlib
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from src.slm import common
from src.slm.rationales import generate, load_model
from src.slm.score import score_texts
from src.splits import verify_test_checksum

MACHINE = "Lab PC (RTX 4500 Ada, Windows 11)"


def summary(seconds):
    s = np.asarray(seconds)
    return {"n": int(len(s)), "median_ms": float(np.median(s) * 1e3), "p95_ms": float(np.percentile(s, 95) * 1e3),
            "mean_ms": float(s.mean() * 1e3), "min_ms": float(s.min() * 1e3), "max_ms": float(s.max() * 1e3)}


def reset_memory(device_name):
    if device_name == "cuda":
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory(device_name):
    import psutil
    info = psutil.Process().memory_info()
    out = {"rss_gb": round(info.rss / 1e9, 3)}
    if hasattr(info, "peak_wset"):                    # Windows
        out["peak_working_set_gb"] = round(info.peak_wset / 1e9, 3)
    if device_name == "cuda":
        import torch
        out["gpu_max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["gpu_max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    return out


def batch1(call, texts, warmup, device_name):
    for t in texts[:warmup]:
        call(t)
    seconds = []
    for t in texts:
        common.sync(device_name)
        start = time.perf_counter()
        call(t)
        common.sync(device_name)
        seconds.append(time.perf_counter() - start)
    return seconds


def throughput(model, tok, texts, device_name, batch, yes_id, no_id):
    score_texts(model, tok, texts[:batch], device_name, batch, yes_id, no_id)      # warm-up
    common.sync(device_name)
    start = time.perf_counter()
    score_texts(model, tok, texts, device_name, batch, yes_id, no_id)
    common.sync(device_name)
    seconds = time.perf_counter() - start
    return {"rows": len(texts), "batch_size": batch, "seconds": seconds, "rows_per_second": len(texts) / seconds}


def environment(device_name, model):
    import peft
    import torch
    import transformers
    env = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
           "peft": peft.__version__, "python": platform.python_version(), "platform": platform.platform(),
           "dtype": str(next(model.parameters()).dtype), "SLM_SCORE_DTYPE": os.environ.get("SLM_SCORE_DTYPE"),
           "cpu": platform.processor(), "cpu_count": os.cpu_count()}
    if device_name == "cuda":
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cudnn"] = torch.backends.cudnn.version()
        try:
            env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            env["driver"] = f"nvidia-smi failed: {e}"
    return env


def main():
    p = common.parser(__doc__)
    p.add_argument("--model", default="primary")
    p.add_argument("--models-dir")
    p.add_argument("--c1-adapter", required=True)
    p.add_argument("--c2-adapter", required=True)
    p.add_argument("--inputs", default=str(common.ROOT / "data/step8"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rationale-rows", type=int, default=150)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, help="first N sample rows only (dry runs)")
    p.add_argument("--machine", default=MACHINE)
    p.add_argument("--out", default=str(common.ROOT / "results/step8/latency_gpu.json"))
    args = p.parse_args()
    log = common.setup_logging("latency", args.log)
    checksum = verify_test_checksum()
    log.info("test_ids.csv SHA-256 %s", checksum)
    if args.device == "cuda" and os.environ.get("SLM_SCORE_DTYPE") != "fp16":
        raise SystemExit("set SLM_SCORE_DTYPE=fp16 (scoring protocol: fp16, merged adapter)")
    common.device(args.device, log)
    sample = pd.read_parquet(pathlib.Path(args.inputs) / "latency_sample.parquet")
    if args.limit:
        sample = sample.head(args.limit)
    structured, text = list(sample["structured_prompt"]), list(sample["text_prompt"])
    started = time.perf_counter()
    result = {"machine": args.machine, "test_ids_sha256": checksum, "device": args.device,
              "inputs": str(pathlib.Path(args.inputs) / "latency_sample.parquet"), "rows": len(sample),
              "protocol": {"warmup_calls": args.warmup, "batch": 1, "synchronize": "before and after each call",
                           "adapter": "merged into the base (score.load_for_scoring)",
                           "rationale_max_new_tokens": args.max_new_tokens, "decoding": "greedy",
                           "throughput_batch_size": args.batch_size},
              "adapters": {"c1": args.c1_adapter, "c2": args.c2_adapter}, "configs": {}}

    def run(key, adapter, plan):
        model, tok, yes_id, no_id = load_model(args, adapter, log)
        result.setdefault("environment", environment(args.device, model))
        for name, kind, texts in plan:
            reset_memory(args.device)
            tokens = [len(tok.encode(t)) for t in texts]
            if kind == "score":
                call = lambda t: score_texts(model, tok, [t], args.device, 1, yes_id, no_id)
            elif kind == "rationale":
                call = lambda t: (score_texts(model, tok, [t], args.device, 1, yes_id, no_id),
                                  generate(model, tok, t, args.device, args.max_new_tokens))
            if kind == "throughput":
                entry = throughput(model, tok, texts, args.device, args.batch_size, yes_id, no_id)
            else:
                entry = {"latency": summary(batch1(call, texts, args.warmup, args.device))}
            entry |= {"model": key, "kind": kind, "prompt_tokens_mean": float(np.mean(tokens)),
                      "prompt_tokens_max": int(max(tokens)), "memory": memory(args.device)}
            if kind == "rationale":
                gen = [generate(model, tok, t, args.device, args.max_new_tokens)[1] for t in texts[:20]]
                entry["new_tokens_mean_first20"] = float(np.mean(gen))
            result["configs"][name] = entry
            log.info("%s: %s", name, {k: v for k, v in entry.items() if k != "memory"} | entry["memory"])
            common.write_json(args.out, result | {"partial": True})
        del model
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()

    run("c1", args.c1_adapter, [("a_score_structured_b1", "score", structured),
                                ("c_score_rationale_structured_b1", "rationale", structured[:args.rationale_rows]),
                                ("a_score_structured_b16", "throughput", structured)])
    run("c2", args.c2_adapter, [("b_score_text_b1", "score", text),
                                ("b_score_text_b16", "throughput", text)])
    result["runtime_seconds"] = round(time.perf_counter() - started, 1)
    common.write_json(args.out, result)
    log.info("wrote %s in %.0f s", args.out, result["runtime_seconds"])


if __name__ == "__main__":
    main()
