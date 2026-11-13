# Paired test: SLM - XGBoost on the test split

Same users in every bootstrap resample (1,000, stratified); difference of seed-averaged metrics, 95% CI. Significant = CI excludes 0.

## Session, max over scoring points (primary)

| Metric | SLM | XGBoost | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.082 | 0.183 | -0.102 | [-0.214, -0.007] | yes |
| Recall @ 1% FPR | 0.215 | 0.276 | -0.060 | [-0.140, -0.014] | yes |
| Recall @ 5% FPR | 0.396 | 0.602 | -0.206 | [-0.468, +0.027] | no |
| Precision @ k (20/day) | 0.022 | 0.030 | -0.008 | [-0.017, -0.002] | yes |

## Session, final point (secondary)

| Metric | SLM | XGBoost | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.105 | 0.264 | -0.159 | [-0.311, -0.031] | yes |
| Recall @ 1% FPR | 0.237 | 0.413 | -0.176 | [-0.332, -0.064] | yes |
| Recall @ 5% FPR | 0.479 | 0.735 | -0.257 | [-0.520, -0.016] | yes |
| Precision @ k (20/day) | 0.026 | 0.046 | -0.020 | [-0.038, -0.005] | yes |

## Row (secondary)

| Metric | SLM | XGBoost | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.091 | 0.172 | -0.081 | [-0.159, -0.001] | yes |
| Recall @ 1% FPR | 0.241 | 0.307 | -0.066 | [-0.132, -0.021] | yes |
| Recall @ 5% FPR | 0.465 | 0.625 | -0.160 | [-0.381, +0.042] | no |
| Precision @ k (20/day) | 0.112 | 0.155 | -0.044 | [-0.079, -0.006] | yes |
