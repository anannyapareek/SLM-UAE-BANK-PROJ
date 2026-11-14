# Ranking check: calibration must not change ranking metrics

Every fit is strictly increasing in the margin, so PR-AUC, recall at FPR and precision/recall at k (means, per-seed values and CI bounds) must equal the uncalibrated ones, per unit and split. Exact equality holds for the SLM and XGBoost text. XGBoost v3 stores its probability as float32, which ties a few rows whose margins differ (test: 778,673 distinct scores vs 778,678 distinct margins); the float64 calibrated scores separate those ties, which moves row-level test PR-AUC by at most 6.3e-07. The check therefore passes within 1e-06 and reports the largest absolute difference.

| Run | Split | Unit | Largest abs. difference | Pass |
|---|---|---|---|---|
| xgb_struct_sessionfit_temperature | calib | session | 1.1e-16 | yes |
| xgb_struct_sessionfit_temperature | calib | row | 5.6e-17 | yes |
| xgb_struct_sessionfit_temperature | test | session | 2.8e-17 | yes |
| xgb_struct_sessionfit_temperature | test | row | 6.3e-07 | yes |
| xgb_struct_sessionfit_platt | calib | session | 1.1e-16 | yes |
| xgb_struct_sessionfit_platt | calib | row | 5.6e-17 | yes |
| xgb_struct_sessionfit_platt | test | session | 2.8e-17 | yes |
| xgb_struct_sessionfit_platt | test | row | 6.3e-07 | yes |
| xgb_struct_rowfit_temperature | calib | session | 1.1e-16 | yes |
| xgb_struct_rowfit_temperature | calib | row | 5.6e-17 | yes |
| xgb_struct_rowfit_temperature | test | session | 2.8e-17 | yes |
| xgb_struct_rowfit_temperature | test | row | 6.3e-07 | yes |
| xgb_struct_rowfit_platt | calib | session | 1.1e-16 | yes |
| xgb_struct_rowfit_platt | calib | row | 5.6e-17 | yes |
| xgb_struct_rowfit_platt | test | session | 2.8e-17 | yes |
| xgb_struct_rowfit_platt | test | row | 6.3e-07 | yes |
| slm_struct_sessionfit_temperature | calib | session | 0.0e+00 | yes |
| slm_struct_sessionfit_temperature | calib | row | 0.0e+00 | yes |
| slm_struct_sessionfit_temperature | test | session | 0.0e+00 | yes |
| slm_struct_sessionfit_temperature | test | row | 0.0e+00 | yes |
| slm_struct_sessionfit_platt | calib | session | 0.0e+00 | yes |
| slm_struct_sessionfit_platt | calib | row | 0.0e+00 | yes |
| slm_struct_sessionfit_platt | test | session | 0.0e+00 | yes |
| slm_struct_sessionfit_platt | test | row | 0.0e+00 | yes |
| slm_struct_rowfit_temperature | calib | session | 0.0e+00 | yes |
| slm_struct_rowfit_temperature | calib | row | 0.0e+00 | yes |
| slm_struct_rowfit_temperature | test | session | 0.0e+00 | yes |
| slm_struct_rowfit_temperature | test | row | 0.0e+00 | yes |
| slm_struct_rowfit_platt | calib | session | 0.0e+00 | yes |
| slm_struct_rowfit_platt | calib | row | 0.0e+00 | yes |
| slm_struct_rowfit_platt | test | session | 0.0e+00 | yes |
| slm_struct_rowfit_platt | test | row | 0.0e+00 | yes |
| xgb_text_sessionfit_temperature | calib | session | 0.0e+00 | yes |
| xgb_text_sessionfit_temperature | calib | row | 0.0e+00 | yes |
| xgb_text_sessionfit_temperature | test | session | 0.0e+00 | yes |
| xgb_text_sessionfit_temperature | test | row | 0.0e+00 | yes |
| xgb_text_sessionfit_platt | calib | session | 0.0e+00 | yes |
| xgb_text_sessionfit_platt | calib | row | 0.0e+00 | yes |
| xgb_text_sessionfit_platt | test | session | 0.0e+00 | yes |
| xgb_text_sessionfit_platt | test | row | 0.0e+00 | yes |
| slm_text_sessionfit_temperature | calib | session | 0.0e+00 | yes |
| slm_text_sessionfit_temperature | calib | row | 0.0e+00 | yes |
| slm_text_sessionfit_temperature | test | session | 0.0e+00 | yes |
| slm_text_sessionfit_temperature | test | row | 0.0e+00 | yes |
| slm_text_sessionfit_platt | calib | session | 0.0e+00 | yes |
| slm_text_sessionfit_platt | calib | row | 0.0e+00 | yes |
| slm_text_sessionfit_platt | test | session | 0.0e+00 | yes |
| slm_text_sessionfit_platt | test | row | 0.0e+00 | yes |

Failures: none.
