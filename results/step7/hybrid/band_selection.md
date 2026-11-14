# Hybrid band selection (Step 7.2, validation only)

1,674 admissible bands (at most 20% of validation rows sent to the SLM for every seed). Scores: Platt row-level calibrated XGBoost v3 and SLM C1; seed i paired with seed i; objective = row-level validation PR-AUC averaged over the 3 pairs.

References: XGBoost alone row 0.3614 / session 0.4162; SLM alone row 0.2292 / session 0.2418.

| Rank | a (level) | b (level) | Share to SLM (mean, max) | Row PR-AUC (mean ± std) | Session PR-AUC (mean) |
|---|---|---|---|---|---|
| 1 | 0.000835543 (0.73) | 0.00307086 (0.92) | 0.190, 0.192 | 0.3674 ± 0.0049 | 0.4227 |
| 2 | 0.000800679 (0.72) | 0.00272355 (0.91) | 0.190, 0.192 | 0.3673 ± 0.0047 | 0.4229 |
| 3 | 0.000835543 (0.73) | 0.00272355 (0.91) | 0.180, 0.182 | 0.3673 ± 0.0047 | 0.4230 |
| 4 | 0.000873192 (0.74) | 0.00307086 (0.92) | 0.180, 0.182 | 0.3672 ± 0.0047 | 0.4228 |
| 5 | 0.000873192 (0.74) | 0.00352352 (0.93) | 0.190, 0.192 | 0.3672 ± 0.0052 | 0.4223 |
| 6 | 0.000913575 (0.75) | 0.00307086 (0.92) | 0.170, 0.172 | 0.3672 ± 0.0046 | 0.4228 |
| 7 | 0.000913575 (0.75) | 0.00352352 (0.93) | 0.180, 0.182 | 0.3672 ± 0.0051 | 0.4223 |
| 8 | 0.000913575 (0.75) | 0.00412572 (0.94) | 0.190, 0.192 | 0.3672 ± 0.0057 | 0.4219 |
| 9 | 0.000767883 (0.71) | 0.00244535 (0.90) | 0.190, 0.193 | 0.3672 ± 0.0045 | 0.4230 |
| 10 | 0.00095806 (0.76) | 0.00307086 (0.92) | 0.160, 0.161 | 0.3671 ± 0.0046 | 0.4229 |

Flatness: {'neighbours': 5, 'largest_drop_to_a_neighbour': 0.00034297856643683566, 'bands_within_0.001': 39, 'bands_within_0.005': 462}
Edge: {'a_at_lowest_level': False, 'b_at_highest_level': False, 'width_at_share_cap': False}
