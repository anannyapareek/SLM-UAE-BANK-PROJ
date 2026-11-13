# Text gain on the test split (logoff rows)

Comparison 1 at logoff rows (structured, final point; identical to Comparison 1's final-point aggregation) against Comparison 2 (text). Paired user bootstrap, the same 1,000 user draws (seed 0) for every row; 95% CI; significant = CI excludes 0.

## PR-AUC

| Comparison | Text (C2) | Structured (C1 final) | Change | 95% CI | Significant |
|---|---|---|---|---|---|
| SLM: text vs structured | 0.093 | 0.105 | -0.012 | [-0.112, +0.045] | no |
| XGBoost: text vs structured | 0.203 | 0.264 | -0.061 | [-0.153, -0.007] | yes |
| **Difference-in-differences** (SLM gain − XGBoost gain) | -0.012 | -0.061 | +0.049 | [-0.059, +0.145] | no |

## Recall @ 1% FPR

| Comparison | Text (C2) | Structured (C1 final) | Change | 95% CI | Significant |
|---|---|---|---|---|---|
| SLM: text vs structured | 0.350 | 0.237 | +0.112 | [-0.024, +0.254] | no |
| XGBoost: text vs structured | 0.346 | 0.413 | -0.067 | [-0.148, +0.014] | no |
| **Difference-in-differences** (SLM gain − XGBoost gain) | +0.112 | -0.067 | +0.180 | [+0.004, +0.355] | yes |

## Recall @ 5% FPR

| Comparison | Text (C2) | Structured (C1 final) | Change | 95% CI | Significant |
|---|---|---|---|---|---|
| SLM: text vs structured | 0.794 | 0.479 | +0.316 | [+0.113, +0.525] | yes |
| XGBoost: text vs structured | 0.671 | 0.735 | -0.064 | [-0.175, +0.038] | no |
| **Difference-in-differences** (SLM gain − XGBoost gain) | +0.316 | -0.064 | +0.380 | [+0.104, +0.661] | yes |

## Precision @ k (20/day)

| Comparison | Text (C2) | Structured (C1 final) | Change | 95% CI | Significant |
|---|---|---|---|---|---|
| SLM: text vs structured | 0.048 | 0.026 | +0.022 | [+0.005, +0.037] | yes |
| XGBoost: text vs structured | 0.040 | 0.046 | -0.007 | [-0.016, +0.001] | no |
| **Difference-in-differences** (SLM gain − XGBoost gain) | +0.022 | -0.007 | +0.029 | [+0.007, +0.049] | yes |

For the difference-in-differences row, the Text and Structured columns hold the SLM gain and the XGBoost gain. Its CI uses the same user draws as the paired tests (src/text_gain.py reuses src/eval.py; its pair CIs equal eval.py's exactly).
