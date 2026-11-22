# Running the detector: what it needs (Step 8.1)

Measured on one lab PC (NVIDIA RTX 4500 Ada Generation, 24 GB; AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD (192 logical)), 17 Nov 2026. Plain-language summary of `operations.md`; every assumption is listed at the end.

**Everything runs locally.** Both models run on the one machine: the language model (Qwen2.5-1.5B with a small LoRA adapter, about 3 GB on the GPU) and the gradient-boosted trees (XGBoost, CPU). Scoring calls no external service or API; the lab runs loaded the model weights from local copies.

**Time per scored row** (one row = one user session at one hourly scoring point):

- XGBoost alone: 3.3 ms.
- Language model, structured features: 33 ms (95% of calls under 34 ms); with the activity-text prompt: 34 ms.
- Language model with a written reason: 502 ms (95% under 598 ms).
- Hybrid (XGBoost on every row, the language model on the 21.5% of rows in the uncertain band): 10.5 ms per row on average; 52.7% of sessions reach the language model at least once.

**Rows per hour on one RTX 4500 Ada** (batched):

- Language model, structured: ~412,000 rows/hour; text prompt: ~189,000.
- Hybrid: ~1,919,000 rows/hour.
- XGBoost alone: over 9 billion rows/hour (not a constraint).

**GPU memory:** 3.4–3.8 GB at the peak (of 24 GB), so the card has room for other work.

**Hourly scoring for a 1,000-user organisation** (~964 sessions per weekday): at most ~964 rows in any hour and ~8,554 rows per weekday. With the hybrid, the busiest hour needs about 10 s of compute (7 s GPU, 3 s CPU) one row at a time; even the language model on every row needs 32 s. One RTX 4500 Ada is busy well under 1% of each hour; a smaller GPU with ~4 GB free would very likely do (not measured).

## Assumptions

1. Timings are from one machine, one session and one 1,000-row sample of test rows; other hardware, drivers or load will differ.
2. Per-row latency is a single request (batch 1) with the model already loaded and warmed up; model loading (seconds) and feature extraction from the logs are not included.
3. Hybrid latency is an average: 21.5% (the test share routed to the language model) × the language model's median + XGBoost's median; the routed share may shift with the data (it was 19.0% on validation).
4. ~964 sessions per weekday and 8.87 scoring points per session are the test-split averages (CERT r4.2, 1,000 users); real organisations differ.
5. "At most ~964 rows in an hour" assumes every session of the day is active in the same hour (an upper bound).
6. "Fully local": the scoring code calls no external service, and the lab runs were pointed at local copies of the weights (`--models-dir models`); network access was not blocked or monitored during the runs.
7. 24 GB is the RTX 4500 Ada's specified memory, not a measurement.
8. Written reasons are optional and cost ~15× the score alone; they are not needed for the risk score.
