# Table 3 vs Table 1: XGBoost, SLM and rules

Means and CI bounds of the ranking metrics must agree within 1e-06 (Platt row fits are strictly increasing; XGBoost float32 ties, see calibration/ranking_check.md).

| Model | Aggregation | Unit | Largest abs. difference | Pass |
|---|---|---|---|---|
| xgb | max | session | 0.0e+00 | yes |
| xgb | max | row | 2.1e-07 | yes |
| xgb | final | session | 0.0e+00 | yes |
| xgb | final | row | 2.1e-07 | yes |
| slm | max | session | 0.0e+00 | yes |
| slm | max | row | 0.0e+00 | yes |
| slm | final | session | 0.0e+00 | yes |
| slm | final | row | 0.0e+00 | yes |
| rules | max | session | 0.0e+00 | yes |
| rules | max | row | 0.0e+00 | yes |
| rules | final | session | 0.0e+00 | yes |
| rules | final | row | 0.0e+00 | yes |

Failures: none.
