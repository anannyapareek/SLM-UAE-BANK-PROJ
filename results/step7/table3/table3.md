# Table 3: Comparison 3 on the test split

Mean ± std over seeds (XGBoost, SLM, hybrid, rules + SLM: 3 seeds, seed i of the hybrid and of rules + SLM using SLM seed i; rules: deterministic, 1 run) and 95% user-level bootstrap CI (1,000 resamples, stratified, seed 0), src/eval.py. XGBoost and SLM are the Platt row-fit files the hybrid is built from; their ranking metrics equal Table 1's (table1_check.md). The hybrid band [a, b] = [0.000835543, 0.00307086] was chosen on validation (X = 20%) and not re-selected after test. Precision at k: top 20 units per day.

## Session, max over scoring points (primary)

95,570 units, 243 positive (0.2543%), 889 users; precision-at-k ceiling 0.089.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) | Test rows sent to SLM | Test sessions with ≥ 1 SLM row |
|---|---|---|---|---|---|---|
| XGBoost (structured v3) | 0.183 ± 0.005 [0.016, 0.453] | 0.276 ± 0.008 [0.044, 0.592] | 0.602 ± 0.024 [0.357, 0.860] | 0.030 ± 0.002 [0.008, 0.059] | – | – |
| SLM (Qwen2.5-1.5B LoRA) | 0.082 ± 0.007 [0.008, 0.300] | 0.215 ± 0.016 [0.000, 0.499] | 0.396 ± 0.071 [0.109, 0.665] | 0.022 ± 0.001 [0.001, 0.052] | – | – |
| Hybrid (XGBoost; SLM in band) | 0.181 ± 0.005 [0.015, 0.449] | 0.273 ± 0.006 [0.042, 0.591] | 0.546 ± 0.041 [0.313, 0.800] | 0.028 ± 0.001 [0.007, 0.058] | 21.5% ± 0.2% | 52.7% ± 0.9% |
| Rules | 0.008 [0.003, 0.019] | 0.070 [0.012, 0.173] | 0.181 [0.044, 0.339] | 0.009 [0.002, 0.019] | – | – |
| Rules + SLM | 0.022 ± 0.003 [0.008, 0.062] | 0.093 ± 0.002 [0.018, 0.201] | 0.396 ± 0.095 [0.212, 0.599] | 0.020 ± 0.000 [0.006, 0.039] | – | – |

## Session, final point (secondary)

95,570 units, 243 positive (0.2543%), 889 users; precision-at-k ceiling 0.089.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) | Test rows sent to SLM | Test sessions with ≥ 1 SLM row |
|---|---|---|---|---|---|---|
| XGBoost (structured v3) | 0.264 ± 0.007 [0.044, 0.560] | 0.413 ± 0.038 [0.186, 0.704] | 0.735 ± 0.013 [0.562, 0.917] | 0.046 ± 0.004 [0.020, 0.074] | – | – |
| SLM (Qwen2.5-1.5B LoRA) | 0.105 ± 0.021 [0.010, 0.335] | 0.237 ± 0.009 [0.009, 0.537] | 0.479 ± 0.177 [0.219, 0.732] | 0.026 ± 0.002 [0.004, 0.054] | – | – |
| Hybrid (XGBoost; SLM in band) | 0.254 ± 0.009 [0.038, 0.545] | 0.364 ± 0.041 [0.138, 0.660] | 0.675 ± 0.054 [0.482, 0.872] | 0.039 ± 0.004 [0.017, 0.068] | 21.5% ± 0.2% | 52.7% ± 0.9% |
| Rules | 0.008 [0.003, 0.020] | 0.086 [0.012, 0.180] | 0.173 [0.035, 0.334] | 0.008 [0.002, 0.018] | – | – |
| Rules + SLM | 0.024 ± 0.004 [0.007, 0.065] | 0.086 ± 0.000 [0.016, 0.285] | 0.398 ± 0.087 [0.218, 0.598] | 0.019 ± 0.001 [0.005, 0.038] | – | – |

## Row (secondary)

848,137 units, 2,169 positive (0.2557%), 889 users; precision-at-k ceiling 0.473.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) | Test rows sent to SLM | Test sessions with ≥ 1 SLM row |
|---|---|---|---|---|---|---|
| XGBoost (structured v3) | 0.172 ± 0.002 [0.012, 0.429] | 0.307 ± 0.016 [0.060, 0.601] | 0.625 ± 0.010 [0.405, 0.858] | 0.155 ± 0.002 [0.006, 0.279] | – | – |
| SLM (Qwen2.5-1.5B LoRA) | 0.091 ± 0.010 [0.010, 0.300] | 0.241 ± 0.011 [0.004, 0.541] | 0.465 ± 0.174 [0.199, 0.743] | 0.112 ± 0.016 [0.000, 0.212] | – | – |
| Hybrid (XGBoost; SLM in band) | 0.168 ± 0.002 [0.012, 0.426] | 0.289 ± 0.015 [0.050, 0.584] | 0.576 ± 0.047 [0.354, 0.806] | 0.155 ± 0.002 [0.005, 0.280] | 21.5% ± 0.2% | 52.7% ± 0.9% |
| Rules | 0.005 [0.002, 0.011] | 0.053 [0.009, 0.103] | 0.126 [0.047, 0.212] | 0.012 [0.002, 0.030] | – | – |
| Rules + SLM | 0.017 ± 0.002 [0.007, 0.040] | 0.127 ± 0.009 [0.036, 0.256] | 0.334 ± 0.005 [0.198, 0.466] | 0.024 ± 0.005 [0.003, 0.051] | – | – |

## Hybrid routing and compute

Test rows sent to the SLM: 21.5% ± 0.2% (per seed 21.3%, 21.4%, 21.7%); above the 20% cap set on validation (19.0% there), from the validation-to-test shift in the XGBoost score distribution; reported as-is, band not changed (decisions log 2026-11-14). Test sessions with at least one SLM row: 52.7% ± 0.9%.

Compute line: the SLM scores 21.5% of 848,137 test rows at 116.8 rows/s (RTX 4500 Ada, fp16 merged adapter) = 1,559 s; XGBoost scores all rows at 685,822 rows/s (Mac (arm64, Darwin), CPU, results/step7/xgb_timing.json) = 1.2 s. Implied hybrid throughput 544 rows/s, 4.65× the SLM scoring every row (7,261 s for the test split). XGBoost was timed on a different machine (CPU) from the SLM (GPU); its share of the time is negligible either way.
