# Table 2: Comparison 2 on the test split (text added, logoff-only)

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) |
|---|---|---|---|---|
| SLM, text (Qwen2.5-1.5B LoRA) | 0.093 ± 0.009 [0.037, 0.247] | 0.350 ± 0.042 [0.195, 0.556] | 0.794 ± 0.004 [0.670, 0.916] | 0.048 ± 0.001 [0.025, 0.073] |
| XGBoost, text | 0.203 ± 0.005 [0.030, 0.486] | 0.346 ± 0.050 [0.119, 0.663] | 0.671 ± 0.011 [0.487, 0.850] | 0.040 ± 0.006 [0.017, 0.067] |
| Rules (reference) | 0.008 [0.003, 0.020] | 0.086 [0.012, 0.180] | 0.173 [0.035, 0.334] | 0.008 [0.002, 0.018] |

Mean ± std over 3 seeds (rules: deterministic, 1 run) and 95% user-level bootstrap CI (1,000 resamples, seed 0), src/eval.py with --rows final. Logoff-only (one row per session): n_units 95,570, n_pos 243, n_users 889, n_pos_users 19, positive rate 0.2543%; boot_skipped slm_text 0, xgb_text 0, rules 0. Precision-at-k ceiling 0.089.

## Paired test: SLM text − XGBoost text (main Comparison 2 test)

Same users in every resample (1,000, seed 0); 95% CI; significant = CI excludes 0.

| Metric | SLM text | XGBoost text | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.093 | 0.203 | -0.110 | [-0.276, +0.016] | no |
| Recall @ 1% FPR | 0.350 | 0.346 | +0.004 | [-0.139, +0.141] | no |
| Recall @ 5% FPR | 0.794 | 0.671 | +0.123 | [-0.019, +0.251] | no |
| Precision @ k (20/day) | 0.048 | 0.040 | +0.009 | [-0.005, +0.023] | no |
