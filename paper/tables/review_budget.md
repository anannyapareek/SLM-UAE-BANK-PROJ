# Review budget at 20 alerts per analyst-day (test, session level)

| Model | Precision @ 20/day | Recall @ 20/day |
|---|---|---|
| **Calendar-day top-k (primary)** | | |
| XGBoost (C1) | 0.030 ± 0.002 [0.008, 0.059] | 0.339 ± 0.026 [0.116, 0.643] |
| SLM (C1) | 0.022 ± 0.001 [0.001, 0.052] | 0.244 ± 0.006 [0.016, 0.550] |
| Hybrid (C1) | 0.028 ± 0.001 [0.007, 0.058] | 0.318 ± 0.017 [0.097, 0.626] |
| Rules | 0.009 [0.002, 0.019] | 0.099 [0.024, 0.198] |
| Rules + SLM | 0.020 ± 0.000 [0.006, 0.039] | 0.226 ± 0.004 [0.081, 0.401] |
| XGBoost text (C2) | 0.040 ± 0.006 [0.017, 0.067] | 0.446 ± 0.068 [0.228, 0.714] |
| SLM text (C2) | 0.048 ± 0.001 [0.025, 0.073] | 0.542 ± 0.017 [0.379, 0.702] |
| **Weekday-only (sensitivity)** | | |
| XGBoost (C1) | 0.042 ± 0.003 [0.011, 0.083] | 0.335 ± 0.026 [0.109, 0.638] |
| SLM (C1) | 0.030 ± 0.001 [0.001, 0.073] | 0.240 ± 0.006 [0.010, 0.537] |
| Hybrid (C1) | 0.040 ± 0.002 [0.010, 0.082] | 0.314 ± 0.017 [0.094, 0.619] |
| Rules | 0.012 [0.002, 0.026] | 0.095 [0.021, 0.194] |
| Rules + SLM | 0.028 ± 0.001 [0.007, 0.054] | 0.222 ± 0.004 [0.074, 0.397] |
| XGBoost text (C2) | 0.056 ± 0.009 [0.023, 0.094] | 0.442 ± 0.068 [0.220, 0.707] |
| SLM text (C2) | 0.068 ± 0.002 [0.035, 0.102] | 0.538 ± 0.017 [0.377, 0.697] |

Mean ± std over seeds 1–3 (rules: deterministic, one run, no std) and 95% user-level bootstrap CI (1,000 stratified resamples, seed 0), src/eval.py. Test split (frozen). Top 20 sessions per day, 1 analyst. Calendar-day top-k (primary, weekends included) and weekday-only (sensitivity).
