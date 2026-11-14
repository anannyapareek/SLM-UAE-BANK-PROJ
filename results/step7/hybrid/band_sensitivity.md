# Hybrid cap sensitivity (Step 7.2, validation only, secondary)

X = 20% is the primary, pre-declared cap; the other caps are a sensitivity analysis for the discussion. Same search as band_selection.md (quantile step 0.01, share <= X for every seed, objective = mean row-level validation PR-AUC over the 3 seed pairings). No test predictions are built for the other caps. Gain = hybrid minus XGBoost alone, per seed pairing (mean ± std over the 3 pairings).

XGBoost alone: row 0.3614, session 0.4162. SLM alone: row 0.2292, session 0.2418.

| X | Best [a, b] (quantile levels) | Share to SLM (mean, max) | Row PR-AUC | Session PR-AUC | Row gain over XGBoost | Session gain over XGBoost |
|---|---|---|---|---|---|---|
| 5% | [0.0017294, 0.00244535] (0.86–0.90) | 0.040, 0.040 | 0.3637 ± 0.0044 | 0.4199 ± 0.0058 | +0.0024 ± 0.0015 | +0.0037 ± 0.0028 |
| 10% | [0.00140522, 0.00307086] (0.83–0.92) | 0.090, 0.091 | 0.3653 ± 0.0043 | 0.4214 ± 0.0078 | +0.0039 ± 0.0033 | +0.0052 ± 0.0049 |
| 20% (primary) | [0.000835543, 0.00307086] (0.73–0.92) | 0.190, 0.192 | 0.3674 ± 0.0049 | 0.4227 ± 0.0092 | +0.0060 ± 0.0054 | +0.0066 ± 0.0067 |
| 30% | [0.000569394, 0.00307086] (0.63–0.92) | 0.290, 0.295 | 0.3683 ± 0.0054 | 0.4230 ± 0.0096 | +0.0069 ± 0.0065 | +0.0068 ± 0.0073 |
