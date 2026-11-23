# Text gain on the test split (logoff rows): C2 text − C1 structured, and DiD

| Comparison | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Precision @ k (20/day) |
|---|---|---|---|---|
| SLM: text − structured | -0.012 [-0.112, +0.045] | +0.112 [-0.024, +0.254] | +0.316 [+0.113, +0.525] * | +0.022 [+0.005, +0.037] * |
| XGBoost: text − structured | -0.061 [-0.153, -0.007] * | -0.067 [-0.148, +0.014] | -0.064 [-0.175, +0.038] | -0.007 [-0.016, +0.001] |
| Difference-in-differences (SLM gain − XGBoost gain) | +0.049 [-0.059, +0.145] | +0.180 [+0.004, +0.355] * | +0.380 [+0.104, +0.661] * | +0.029 [+0.007, +0.049] * |

Change = text (C2) − structured (C1, final point) for the same model; difference-in-differences = SLM gain − XGBoost gain. Paired user bootstrap, the same 1,000 user draws (seed 0); 95% CI; * = CI excludes 0. src/text_gain.py reuses src/eval.py.
