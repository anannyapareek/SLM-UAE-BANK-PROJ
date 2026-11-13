# Table 1: Comparison 1 on the test split

Mean ± std over seeds (SLM and XGBoost: 3 seeds; rules: deterministic, 1 run) and 95% user-level bootstrap CI (1,000 resamples). Recall at FPR is the pre-declared curve metric of src/eval.py (highest recall with FPR at or below the level on this split). Precision at k: top 20 units per day.

## Session, max over scoring points (primary)

95,570 units, 243 positive (0.2543%), 889 users; precision-at-k ceiling 0.089.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) |
|---|---|---|---|---|
| SLM (Qwen2.5-1.5B LoRA) | 0.082 ± 0.007 [0.008, 0.300] | 0.215 ± 0.016 [0.000, 0.499] | 0.396 ± 0.071 [0.109, 0.665] | 0.022 ± 0.001 [0.001, 0.052] |
| XGBoost (structured v3) | 0.183 ± 0.005 [0.016, 0.453] | 0.276 ± 0.008 [0.044, 0.592] | 0.602 ± 0.024 [0.357, 0.860] | 0.030 ± 0.002 [0.008, 0.059] |
| Rules | 0.008 [0.003, 0.019] | 0.070 [0.012, 0.173] | 0.181 [0.044, 0.339] | 0.009 [0.002, 0.019] |

## Session, final point (secondary)

95,570 units, 243 positive (0.2543%), 889 users; precision-at-k ceiling 0.089.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) |
|---|---|---|---|---|
| SLM (Qwen2.5-1.5B LoRA) | 0.105 ± 0.021 [0.010, 0.335] | 0.237 ± 0.009 [0.009, 0.537] | 0.479 ± 0.177 [0.219, 0.732] | 0.026 ± 0.002 [0.004, 0.054] |
| XGBoost (structured v3) | 0.264 ± 0.007 [0.044, 0.560] | 0.413 ± 0.038 [0.186, 0.704] | 0.735 ± 0.013 [0.562, 0.917] | 0.046 ± 0.004 [0.020, 0.074] |
| Rules | 0.008 [0.003, 0.020] | 0.086 [0.012, 0.180] | 0.173 [0.035, 0.334] | 0.008 [0.002, 0.018] |

## Row (secondary)

848,137 units, 2,169 positive (0.2557%), 889 users; precision-at-k ceiling 0.473.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) |
|---|---|---|---|---|
| SLM (Qwen2.5-1.5B LoRA) | 0.091 ± 0.010 [0.010, 0.300] | 0.241 ± 0.011 [0.004, 0.541] | 0.465 ± 0.174 [0.199, 0.743] | 0.112 ± 0.016 [0.000, 0.212] |
| XGBoost (structured v3) | 0.172 ± 0.002 [0.012, 0.429] | 0.307 ± 0.016 [0.060, 0.601] | 0.625 ± 0.010 [0.405, 0.858] | 0.155 ± 0.002 [0.006, 0.279] |
| Rules | 0.005 [0.002, 0.011] | 0.053 [0.009, 0.103] | 0.126 [0.047, 0.212] | 0.012 [0.002, 0.030] |
