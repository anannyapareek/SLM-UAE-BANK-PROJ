# Paired tests on the test split (Comparison 3)

Same users in every bootstrap resample (1,000, stratified, seed 0); difference of seed-averaged metrics, 95% CI. Significant = CI excludes 0. Hybrid − XGBoost is the key test (runbook 7.4).

## Hybrid − XGBoost (key test)

### Session, max over scoring points (primary)

| Metric | Hybrid (XGBoost; SLM in band) | XGBoost (structured v3) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.181 | 0.183 | -0.0022 | [-0.0068, +0.0015] | no |
| Recall @ 1% FPR | 0.273 | 0.276 | -0.0027 | [-0.0091, +0.0000] | no |
| Recall @ 5% FPR | 0.546 | 0.602 | -0.0562 | [-0.1562, +0.0520] | no |
| Precision @ k (20/day) | 0.028 | 0.030 | -0.0018 | [-0.0045, +0.0000] | no |

### Session, final point (secondary)

| Metric | Hybrid (XGBoost; SLM in band) | XGBoost (structured v3) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.254 | 0.264 | -0.0098 | [-0.0259, -0.0017] | yes |
| Recall @ 1% FPR | 0.364 | 0.413 | -0.0494 | [-0.1043, -0.0132] | yes |
| Recall @ 5% FPR | 0.675 | 0.735 | -0.0604 | [-0.1233, -0.0022] | yes |
| Precision @ k (20/day) | 0.039 | 0.046 | -0.0071 | [-0.0130, -0.0009] | yes |

### Row (secondary)

| Metric | Hybrid (XGBoost; SLM in band) | XGBoost (structured v3) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.168 | 0.172 | -0.0034 | [-0.0106, +0.0016] | no |
| Recall @ 1% FPR | 0.289 | 0.307 | -0.0181 | [-0.0487, -0.0011] | yes |
| Recall @ 5% FPR | 0.576 | 0.625 | -0.0489 | [-0.1154, +0.0217] | no |
| Precision @ k (20/day) | 0.155 | 0.155 | -0.0002 | [-0.0015, +0.0002] | no |

## Hybrid − SLM

### Session, max over scoring points (primary)

| Metric | Hybrid (XGBoost; SLM in band) | SLM (Qwen2.5-1.5B LoRA) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.181 | 0.082 | +0.0997 | [+0.0054, +0.2132] | yes |
| Recall @ 1% FPR | 0.273 | 0.215 | +0.0576 | [+0.0124, +0.1396] | yes |
| Recall @ 5% FPR | 0.546 | 0.396 | +0.1495 | [+0.0117, +0.3143] | yes |
| Precision @ k (20/day) | 0.028 | 0.022 | +0.0066 | [+0.0016, +0.0128] | yes |

### Session, final point (secondary)

| Metric | Hybrid (XGBoost; SLM in band) | SLM (Qwen2.5-1.5B LoRA) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.254 | 0.105 | +0.1489 | [+0.0252, +0.2946] | yes |
| Recall @ 1% FPR | 0.364 | 0.237 | +0.1262 | [+0.0386, +0.2465] | yes |
| Recall @ 5% FPR | 0.675 | 0.479 | +0.1962 | [+0.0041, +0.4296] | yes |
| Precision @ k (20/day) | 0.039 | 0.026 | +0.0130 | [+0.0020, +0.0262] | yes |

### Row (secondary)

| Metric | Hybrid (XGBoost; SLM in band) | SLM (Qwen2.5-1.5B LoRA) | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.168 | 0.091 | +0.0775 | [+0.0016, +0.1536] | yes |
| Recall @ 1% FPR | 0.289 | 0.241 | +0.0475 | [+0.0197, +0.0860] | yes |
| Recall @ 5% FPR | 0.576 | 0.465 | +0.1108 | [-0.0293, +0.2712] | no |
| Precision @ k (20/day) | 0.155 | 0.112 | +0.0436 | [+0.0052, +0.0782] | yes |

## Rules + SLM − Rules

### Session, max over scoring points (primary)

| Metric | Rules + SLM | Rules | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.022 | 0.008 | +0.0147 | [+0.0035, +0.0504] | yes |
| Recall @ 1% FPR | 0.093 | 0.070 | +0.0233 | [-0.0220, +0.0816] | no |
| Recall @ 5% FPR | 0.396 | 0.181 | +0.2154 | [+0.1039, +0.3700] | yes |
| Precision @ k (20/day) | 0.020 | 0.009 | +0.0114 | [+0.0010, +0.0246] | yes |

### Session, final point (secondary)

| Metric | Rules + SLM | Rules | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.024 | 0.008 | +0.0161 | [+0.0034, +0.0535] | yes |
| Recall @ 1% FPR | 0.086 | 0.086 | +0.0000 | [-0.0037, +0.1551] | no |
| Recall @ 5% FPR | 0.398 | 0.173 | +0.2250 | [+0.1120, +0.3670] | yes |
| Precision @ k (20/day) | 0.019 | 0.008 | +0.0106 | [+0.0015, +0.0252] | yes |

### Row (secondary)

| Metric | Rules + SLM | Rules | Difference | 95% CI | Significant |
|---|---|---|---|---|---|
| PR-AUC | 0.017 | 0.005 | +0.0116 | [+0.0038, +0.0327] | yes |
| Recall @ 1% FPR | 0.127 | 0.053 | +0.0735 | [+0.0027, +0.1799] | yes |
| Recall @ 5% FPR | 0.334 | 0.126 | +0.2078 | [+0.1231, +0.2996] | yes |
| Precision @ k (20/day) | 0.024 | 0.012 | +0.0120 | [-0.0005, +0.0304] | no |
