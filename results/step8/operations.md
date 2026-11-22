# Step 8.1 operations table

All timings from one lab session on 17 Nov 2026 (Lab PC (RTX 4500 Ada, Windows 11)), protocol fixed before the session (decisions log 2026-11-15): batch 1 = 20 warm-up calls then one call per row, device synchronised before and after each call; SLM = Qwen2.5-1.5B + seed-1 LoRA adapter merged, fp16; a call = tokenise, forward, fp32 yes/no head, copy to host. Inputs: 1,000 test logoff rows (numpy default_rng(0)); rationale config on the first 150. Sources: `latency_gpu.json`, `latency_xgb_lab.json`. Timings, not metrics.

| Configuration | Batch-1 median (ms) | Batch-1 p95 (ms) | Throughput (rows/s) | Peak VRAM allocated (GB) | RSS (GB) | Prompt tokens (mean) |
|---|---|---|---|---|---|---|
| XGBoost v3 seed 1 | 3.35 | 3.47 | 2,455,663 (batch 100,000) | – | – | – |
| SLM structured (C1 seed 1) | 33.44 | 34.12 | 114.5 (batch 16) | 3.43 | 5.02 | 199 |
| SLM text (C2 seed 1) | 33.89 | 34.67 | 52.5 (batch 16) | 3.83 | 4.99 | 409 |
| SLM structured + rationale (C1 seed 1) | 501.99 | 598.15 | – | 3.12 | 4.72 | 199 |
| Hybrid (XGBoost; SLM C1 in band) | 10.52 | – | 533.1 | 3.43 | – | – |

Notes:

- **XGBoost v3 seed 1**: 44 features in memory; 195,775 rows/s at a 1,000-row batch; RSS not recorded.
- **SLM structured (C1 seed 1)**: score only.
- **SLM text (C2 seed 1)**: score only.
- **SLM structured + rationale (C1 seed 1)**: score + greedy rationale, max 48 new tokens (17.4 mean over the first 20 rows); batch 1 only.
- **Hybrid (XGBoost; SLM C1 in band)**: effective per-row latency = 21.5% × SLM batch-1 median + XGBoost batch-1 median (expected value, not a measured distribution; no p95); a routed row costs 36.8 ms, an unrouted row 3.3 ms; 52.7% of test sessions send at least one row to the SLM; throughput = Table 3 compute line.

Peak VRAM = `torch.cuda.max_memory_allocated`, reset per configuration, weights included (reserved: up to 4.54 GB). RSS = psutil after each configuration.

## Hardware and software

| Item | Value |
|---|---|
| gpu | NVIDIA RTX 4500 Ada Generation |
| driver | 581.80 |
| cuda | 12.8 |
| torch | 2.11.0+cu128 |
| transformers | 5.22.2 |
| peft | 0.21.2 |
| cudnn | 91900 |
| xgboost | 3.4.1 |
| python | 3.12.16 |
| os | Windows-11-10.0.26200-SP0 |
| cpu | AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD (192 logical) |

SLM latency run time 228 s. Test set SHA-256 `5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51` in both files.
