# Abstention (Step 7.5, test, session level)

Each model and seed defers the test sessions closest to its validation session-level 5%-FPR threshold, |logit(p) − logit(threshold)|, to analysts; alert = score ≥ threshold on the rest. Mean ± std over seeds 1–3 and 95% user-level bootstrap CI (1,000 stratified resamples, seed 0, the same draws as src/eval.py), from src/eval.py abstention(). System recall assumes analysts catch every deferred positive (an upper bound). Scores: calibrated logits (Platt session fit; hybrid: its Platt row-fit logit); any affine logit map gives the same thresholds and deferral order. Rules and rules + SLM are not calibrated (rank only) and are not included.

Test: 95,570 sessions, 243 positive, 137 days. Deferring 10% = 9,557 sessions = 69.8 sessions per test day for analysts.

## Comparison 1 models and the hybrid (main)

| Model | Deferred | Precision (auto-decided) | Recall (auto-decided) | FPR (auto-decided) | System recall (upper bound) | Positives deferred |
|---|---|---|---|---|---|---|
| XGBoost (structured v3) | 0% | 0.028 ± 0.001 [0.014, 0.046] | 0.639 ± 0.006 [0.411, 0.891] | 0.056 ± 0.002 [0.045, 0.069] | 0.639 ± 0.006 [0.411, 0.891] | 0.000 ± 0.000 [0.000, 0.000] |
| XGBoost (structured v3) | 10% (operating point) | 0.038 ± 0.003 [0.013, 0.071] | 0.739 ± 0.017 [0.412, 0.948] | 0.030 ± 0.001 [0.021, 0.041] | 0.855 ± 0.014 [0.729, 0.965] | 0.443 ± 0.026 [0.228, 0.618] |
| SLM C1 | 0% | 0.019 ± 0.004 [0.006, 0.036] | 0.418 ± 0.101 [0.163, 0.682] | 0.055 ± 0.000 [0.042, 0.071] | 0.418 ± 0.101 [0.163, 0.682] | 0.000 ± 0.000 [0.000, 0.000] |
| SLM C1 | 10% (operating point) | 0.037 ± 0.003 [0.000, 0.100] | 0.444 ± 0.190 [0.011, 0.764] | 0.017 ± 0.002 [0.009, 0.028] | 0.657 ± 0.195 [0.451, 0.850] | 0.427 ± 0.192 [0.232, 0.656] |
| Hybrid | 0% | 0.026 ± 0.002 [0.012, 0.042] | 0.575 ± 0.051 [0.351, 0.813] | 0.055 ± 0.001 [0.045, 0.068] | 0.575 ± 0.051 [0.351, 0.813] | 0.000 ± 0.000 [0.000, 0.000] |
| Hybrid | 10% (operating point) | 0.037 ± 0.003 [0.010, 0.076] | 0.691 ± 0.099 [0.328, 0.925] | 0.025 ± 0.002 [0.017, 0.035] | 0.846 ± 0.055 [0.716, 0.956] | 0.508 ± 0.046 [0.252, 0.702] |

## Comparison 2 models (appendix)

| Model | Deferred | Precision (auto-decided) | Recall (auto-decided) | FPR (auto-decided) | System recall (upper bound) | Positives deferred |
|---|---|---|---|---|---|---|
| XGBoost text (C2) | 0% | 0.030 ± 0.001 [0.016, 0.046] | 0.722 ± 0.026 [0.568, 0.880] | 0.060 ± 0.002 [0.048, 0.075] | 0.722 ± 0.026 [0.568, 0.880] | 0.000 ± 0.000 [0.000, 0.000] |
| XGBoost text (C2) | 10% (operating point) | 0.046 ± 0.003 [0.021, 0.079] | 0.833 ± 0.034 [0.694, 0.937] | 0.032 ± 0.001 [0.022, 0.045] | 0.892 ± 0.026 [0.833, 0.947] | 0.354 ± 0.057 [0.165, 0.513] |
| SLM text (C2) | 0% | 0.035 ± 0.001 [0.020, 0.054] | 0.840 ± 0.022 [0.740, 0.932] | 0.059 ± 0.000 [0.046, 0.074] | 0.840 ± 0.022 [0.740, 0.932] | 0.000 ± 0.000 [0.000, 0.000] |
| SLM text (C2) | 10% (operating point) | 0.066 ± 0.003 [0.034, 0.106] | 0.941 ± 0.029 [0.884, 0.987] | 0.023 ± 0.002 [0.015, 0.034] | 0.964 ± 0.017 [0.937, 0.992] | 0.396 ± 0.026 [0.260, 0.543] |

## Workload view: low deferral shares

Deferral shares an analyst team can absorb. Sessions per test day are the same for every model (a share of all test sessions). Mean ± std over seeds and 95% user-bootstrap CI.

| Deferred | Sessions/day | Model | Positives deferred | System recall (upper bound) |
|---|---|---|---|---|
| 1% | 7.0 | XGBoost (structured v3) | 0.056 ± 0.013 [0.027, 0.087] | 0.668 ± 0.005 [0.452, 0.910] |
| 1% | 7.0 | SLM C1 | 0.047 ± 0.056 [0.024, 0.073] | 0.443 ± 0.126 [0.185, 0.699] |
| 1% | 7.0 | Hybrid | 0.073 ± 0.027 [0.036, 0.114] | 0.623 ± 0.070 [0.404, 0.846] |
| 1% | 7.0 | XGBoost text (C2) | 0.045 ± 0.004 [0.017, 0.071] | 0.745 ± 0.022 [0.600, 0.891] |
| 1% | 7.0 | SLM text (C2) | 0.045 ± 0.004 [0.023, 0.081] | 0.861 ± 0.021 [0.773, 0.948] |
| 2% | 14.0 | XGBoost (structured v3) | 0.106 ± 0.019 [0.052, 0.150] | 0.687 ± 0.014 [0.479, 0.923] |
| 2% | 14.0 | SLM C1 | 0.092 ± 0.102 [0.046, 0.144] | 0.462 ± 0.145 [0.205, 0.722] |
| 2% | 14.0 | Hybrid | 0.139 ± 0.041 [0.068, 0.204] | 0.652 ± 0.070 [0.440, 0.864] |
| 2% | 14.0 | XGBoost text (C2) | 0.089 ± 0.002 [0.036, 0.134] | 0.763 ± 0.019 [0.629, 0.896] |
| 2% | 14.0 | SLM text (C2) | 0.097 ± 0.006 [0.050, 0.153] | 0.892 ± 0.005 [0.814, 0.960] |
| 10% | 69.8 | XGBoost (structured v3) | 0.443 ± 0.026 [0.228, 0.618] | 0.855 ± 0.014 [0.729, 0.965] |
| 10% | 69.8 | SLM C1 | 0.427 ± 0.192 [0.232, 0.656] | 0.657 ± 0.195 [0.451, 0.850] |
| 10% | 69.8 | Hybrid | 0.508 ± 0.046 [0.252, 0.702] | 0.846 ± 0.055 [0.716, 0.956] |
| 10% | 69.8 | XGBoost text (C2) | 0.354 ± 0.057 [0.165, 0.513] | 0.892 ± 0.026 [0.833, 0.947] |
| 10% | 69.8 | SLM text (C2) | 0.396 ± 0.026 [0.260, 0.543] | 0.964 ± 0.017 [0.937, 0.992] |
| P7 capacity share: to be filled after P7 (Step 7.6) | – | – | – | – |

## Exploratory (not pre-declared): paired SLM text (C2) − XGBoost text (C2)

Not pre-declared; reported as exploratory (decisions log 2026-11-14). Same user draws for both models (1,000 stratified resamples, seed 0); difference of seed-averaged metrics; significant = 95% CI excludes 0.

| Deferred | Metric | SLM text (C2) | XGBoost text (C2) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|---|
| 0% | System recall (upper bound) | 0.840 | 0.722 | +0.1180 | [-0.0053, +0.2365] | no |
| 0% | Precision (auto-decided) | 0.035 | 0.030 | +0.0053 | [-0.0013, +0.0139] | no |
| 0% | Recall (auto-decided) | 0.840 | 0.722 | +0.1180 | [-0.0053, +0.2365] | no |
| 10% | System recall (upper bound) | 0.964 | 0.892 | +0.0727 | [+0.0260, +0.1263] | yes |
| 10% | Precision (auto-decided) | 0.066 | 0.046 | +0.0203 | [+0.0008, +0.0425] | yes |
| 10% | Recall (auto-decided) | 0.941 | 0.833 | +0.1076 | [+0.0259, +0.2299] | yes |

Figures: `abstention_main.png/.pdf` (C1 + hybrid) and `abstention_appendix.png/.pdf` (C2); deferral share 0–50% in steps of 1%, mean over seeds (line) and each seed (thin), the 10% operating point marked. Precision and recall on auto-decided sessions become unstable once few alerts or positives remain among them (e.g. the SLM curves at high deferral shares, where their kept alerts run out); counts per point and seed are in curves.csv (alerts_auto, positives_auto).
