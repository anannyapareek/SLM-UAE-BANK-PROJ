# Review budget (Step 7.6, test, session level)

Capacity: **1 analyst × 20 alerts per calendar day** (configs/eval.yaml alerts_per_day; decisions log 2026-11-15), set before any curve; 10 and 50 per day are sensitivity values. Each test day alerts on its top k sessions by score (src/eval.py top_k_per_day; session score = max over scoring points, C2 = logoff point; ties by eval.py's hash key). Mean ± std over seeds 1–3 (rules: one deterministic run) and 95% user-level bootstrap CI (1,000 stratified resamples, seed 0, the same draws as src/eval.py and for every model).

Test: 95,570 sessions, 243 positive, 137 days (97 weekdays). Calendar-day top-k (primary, D4) alerts on weekends too; weekday-only (sensitivity) gives weekends no alerts, so their 1 positive session(s) count as missed. Precision@k ceilings are in capacity_facts.md.

## Calendar-day top-k (primary)

| Model | Recall @ 10/day | Recall @ 20/day | Recall @ 50/day | Precision @ 20/day |
|---|---|---|---|---|
| XGBoost (structured v3) | 0.277 ± 0.010 [0.051, 0.596] | **0.339 ± 0.026 [0.116, 0.643]** | 0.602 ± 0.021 [0.360, 0.859] | 0.030 ± 0.002 [0.008, 0.059] |
| SLM C1 | 0.214 ± 0.021 [0.003, 0.489] | **0.244 ± 0.006 [0.016, 0.550]** | 0.390 ± 0.058 [0.119, 0.670] | 0.022 ± 0.001 [0.001, 0.052] |
| Hybrid | 0.272 ± 0.007 [0.049, 0.590] | **0.318 ± 0.017 [0.097, 0.626]** | 0.540 ± 0.035 [0.316, 0.798] | 0.028 ± 0.001 [0.007, 0.058] |
| Rules | 0.082 [0.013, 0.155] | **0.099 [0.024, 0.198]** | 0.185 [0.048, 0.348] | 0.009 [0.002, 0.019] |
| Rules + SLM | 0.096 ± 0.006 [0.026, 0.222] | **0.226 ± 0.004 [0.081, 0.401]** | 0.410 ± 0.088 [0.224, 0.604] | 0.020 ± 0.000 [0.006, 0.039] |
| XGBoost text (C2) | 0.331 ± 0.028 [0.117, 0.640] | **0.446 ± 0.068 [0.228, 0.714]** | 0.674 ± 0.023 [0.486, 0.847] | 0.040 ± 0.006 [0.017, 0.067] |
| SLM text (C2) | 0.328 ± 0.039 [0.180, 0.534] | **0.542 ± 0.017 [0.379, 0.702]** | 0.787 ± 0.010 [0.661, 0.909] | 0.048 ± 0.001 [0.025, 0.073] |

## Weekday-only top-k (sensitivity)

| Model | Recall @ 10/day | Recall @ 20/day | Recall @ 50/day | Precision @ 20/day |
|---|---|---|---|---|
| XGBoost (structured v3) | 0.273 ± 0.010 [0.049, 0.591] | **0.335 ± 0.026 [0.109, 0.638]** | 0.598 ± 0.021 [0.352, 0.855] | 0.042 ± 0.003 [0.011, 0.083] |
| SLM C1 | 0.213 ± 0.020 [0.001, 0.489] | **0.240 ± 0.006 [0.010, 0.537]** | 0.385 ± 0.058 [0.117, 0.665] | 0.030 ± 0.001 [0.001, 0.073] |
| Hybrid | 0.272 ± 0.007 [0.049, 0.587] | **0.314 ± 0.017 [0.094, 0.619]** | 0.536 ± 0.035 [0.313, 0.791] | 0.040 ± 0.002 [0.010, 0.082] |
| Rules | 0.082 [0.013, 0.155] | **0.095 [0.021, 0.194]** | 0.181 [0.044, 0.339] | 0.012 [0.002, 0.026] |
| Rules + SLM | 0.092 ± 0.006 [0.022, 0.214] | **0.222 ± 0.004 [0.074, 0.397]** | 0.406 ± 0.088 [0.221, 0.599] | 0.028 ± 0.001 [0.007, 0.054] |
| XGBoost text (C2) | 0.326 ± 0.028 [0.111, 0.632] | **0.442 ± 0.068 [0.220, 0.707]** | 0.669 ± 0.023 [0.481, 0.843] | 0.056 ± 0.009 [0.023, 0.094] |
| SLM text (C2) | 0.324 ± 0.039 [0.176, 0.529] | **0.538 ± 0.017 [0.377, 0.697]** | 0.783 ± 0.010 [0.658, 0.904] | 0.068 ± 0.002 [0.035, 0.102] |

## Paired SLM text (C2) − XGBoost text (C2)

Pre-declared (runbook 7.6 project decision) at the chosen capacity, calendar-day top-k, 20/day; the other rows are sensitivity. Same user draws for both models; difference of seed-averaged metrics; significant = 95% CI excludes 0.

| Top-k | Alerts/day | Metric | SLM text (C2) | XGBoost text (C2) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|---|---|
| calendar (pre-declared) | 20 | precision | 0.048 | 0.040 | +0.0086 | [-0.0054, +0.0227] | no |
| calendar (pre-declared) | 20 | recall | 0.542 | 0.446 | +0.0960 | [-0.0687, +0.2444] | no |
| calendar | 10 | precision | 0.058 | 0.059 | -0.0005 | [-0.0250, +0.0210] | no |
| calendar | 10 | recall | 0.328 | 0.331 | -0.0027 | [-0.1422, +0.1184] | no |
| calendar | 50 | precision | 0.028 | 0.024 | +0.0041 | [-0.0008, +0.0101] | no |
| calendar | 50 | recall | 0.787 | 0.674 | +0.1139 | [-0.0241, +0.2526] | no |
| weekday | 20 | precision | 0.068 | 0.056 | +0.0121 | [-0.0076, +0.0322] | no |
| weekday | 20 | recall | 0.538 | 0.442 | +0.0960 | [-0.0687, +0.2444] | no |
| weekday | 10 | precision | 0.082 | 0.082 | -0.0007 | [-0.0353, +0.0298] | no |
| weekday | 10 | recall | 0.324 | 0.326 | -0.0027 | [-0.1422, +0.1184] | no |
| weekday | 50 | precision | 0.040 | 0.034 | +0.0058 | [-0.0010, +0.0140] | no |
| weekday | 50 | recall | 0.783 | 0.669 | +0.1139 | [-0.0241, +0.2526] | no |

Figures: `review_budget_main.png/.pdf` (C1 models, hybrid, rules, rules + SLM), `review_budget_c2.png/.pdf` (C2 pair), `review_budget_weekday.png/.pdf` (weekday-only, both groups). Recall vs alerts per analyst-day (k = 1–100); thick = mean over seeds, thin = each seed; solid line = capacity 20/day, dashed = 10 and 50/day. Per-k values for every seed are in curves.csv.
