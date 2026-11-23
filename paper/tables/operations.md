# Operations (Step 8.1, one lab PC: RTX 4500 Ada, Windows 11)

| Configuration | Batch-1 median (ms) | Batch-1 p95 (ms) | Throughput (rows/s) | Peak VRAM (GB) | RSS (GB) |
|---|---|---|---|---|---|
| XGBoost v3 seed 1 | 3.35 | 3.47 | 2,455,663 | – | – |
| SLM structured (C1 seed 1) | 33.44 | 34.12 | 114.5 | 3.43 | 5.02 |
| SLM text (C2 seed 1) | 33.89 | 34.67 | 52.5 | 3.83 | 4.99 |
| SLM structured + rationale (C1 seed 1) | 501.99 | 598.15 | – | 3.12 | 4.72 |
| Hybrid (XGBoost; SLM C1 in band) | 10.52 | – | 533.1 | 3.43 | – |

Timings, not metrics (results/step8/operations.md): batch 1 after 20 warm-up calls, device synchronised per call, 1,000 test logoff rows (rationale: first 150); SLM fp16 with the merged seed-1 adapter; throughput at batch 16 (SLM) or 100,000 rows (XGBoost). Hybrid latency = 21.5% × SLM + XGBoost batch-1 medians (expected value); hybrid throughput = Table 3 compute line.
