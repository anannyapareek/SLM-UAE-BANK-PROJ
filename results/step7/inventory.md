# Step 7 inventory (P0)

test_ids.csv SHA-256 `5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51` (verified). No metrics computed.

| Comparison | Model | Seed | Split | File | Rows | Columns | Margin | max abs(score − σ(margin)) | row_ids = expected | test = frozen |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | XGBoost v3 (structured, hourly) | 1 | valid | `results/xgb_struct_v3_s1/preds_valid.parquet` | 336,737 | row_id, user, y_true, score, margin | yes | 8.16e-08 | yes |  |
| C1 | XGBoost v3 (structured, hourly) | 1 | calib | `results/xgb_struct_v3_s1/preds_calib.parquet` | 309,413 | row_id, user, y_true, score, margin | yes | 8.27e-08 | yes |  |
| C1 | XGBoost v3 (structured, hourly) | 1 | test | `results/xgb_struct_v3_s1/preds_test.parquet` | 848,137 | row_id, user, y_true, score, margin | yes | 8.42e-08 | yes | yes |
| C1 | XGBoost v3 (structured, hourly) | 2 | valid | `results/xgb_struct_v3_s2/preds_valid.parquet` | 336,737 | row_id, user, y_true, score, margin | yes | 8.13e-08 | yes |  |
| C1 | XGBoost v3 (structured, hourly) | 2 | calib | `results/xgb_struct_v3_s2/preds_calib.parquet` | 309,413 | row_id, user, y_true, score, margin | yes | 7.99e-08 | yes |  |
| C1 | XGBoost v3 (structured, hourly) | 2 | test | `results/xgb_struct_v3_s2/preds_test.parquet` | 848,137 | row_id, user, y_true, score, margin | yes | 8.68e-08 | yes | yes |
| C1 | XGBoost v3 (structured, hourly) | 3 | valid | `results/xgb_struct_v3_s3/preds_valid.parquet` | 336,737 | row_id, user, y_true, score, margin | yes | 8.53e-08 | yes |  |
| C1 | XGBoost v3 (structured, hourly) | 3 | calib | `results/xgb_struct_v3_s3/preds_calib.parquet` | 309,413 | row_id, user, y_true, score, margin | yes | 8.27e-08 | yes |  |
| C1 | XGBoost v3 (structured, hourly) | 3 | test | `results/xgb_struct_v3_s3/preds_test.parquet` | 848,137 | row_id, user, y_true, score, margin | yes | 8.37e-08 | yes | yes |
| C1 | SLM C1 (structured, hourly) | 1 | valid | `results/c1_qwen15_r8_lr1e-4_s1/preds_valid.parquet` | 336,737 | row_id, y_true, score, margin | yes | 8.44e-08 | yes |  |
| C1 | SLM C1 (structured, hourly) | 1 | calib | `results/c1_qwen15_r8_lr1e-4_s1/preds_calib.parquet` | 309,413 | row_id, y_true, score, margin | yes | 8.16e-08 | yes |  |
| C1 | SLM C1 (structured, hourly) | 1 | test | `results/c1_qwen15_r8_lr1e-4_s1/preds_test.parquet` | 848,137 | row_id, y_true, score, margin | yes | 8.50e-08 | yes | yes |
| C1 | SLM C1 (structured, hourly) | 2 | valid | `results/c1_qwen15_r8_lr1e-4_s2/preds_valid.parquet` | 336,737 | row_id, y_true, score, margin | yes | 8.16e-08 | yes |  |
| C1 | SLM C1 (structured, hourly) | 2 | calib | `results/c1_qwen15_r8_lr1e-4_s2/preds_calib.parquet` | 309,413 | row_id, y_true, score, margin | yes | 8.16e-08 | yes |  |
| C1 | SLM C1 (structured, hourly) | 2 | test | `results/c1_qwen15_r8_lr1e-4_s2/preds_test.parquet` | 848,137 | row_id, y_true, score, margin | yes | 8.11e-08 | yes | yes |
| C1 | SLM C1 (structured, hourly) | 3 | valid | `results/c1_qwen15_r8_lr1e-4_s3/preds_valid.parquet` | 336,737 | row_id, y_true, score, margin | yes | 8.30e-08 | yes |  |
| C1 | SLM C1 (structured, hourly) | 3 | calib | `results/c1_qwen15_r8_lr1e-4_s3/preds_calib.parquet` | 309,413 | row_id, y_true, score, margin | yes | 8.23e-08 | yes |  |
| C1 | SLM C1 (structured, hourly) | 3 | test | `results/c1_qwen15_r8_lr1e-4_s3/preds_test.parquet` | 848,137 | row_id, y_true, score, margin | yes | 8.48e-08 | yes | yes |
| C1 | Rules (hourly) | - | valid | `results/rules/preds_valid.parquet` | 336,737 | row_id, user, y_true, score, rules_fired | no | n/a | yes |  |
| C1 | Rules (hourly) | - | calib | `results/rules/preds_calib.parquet` | 309,413 | row_id, user, y_true, score, rules_fired | no | n/a | yes |  |
| C1 | Rules (hourly) | - | test | `results/rules/preds_test.parquet` | 848,137 | row_id, user, y_true, score, rules_fired | no | n/a | yes | yes |
| C2 | XGBoost text (logoff rows) | 1 | valid | `results/xgb_text_s1/preds_valid_final.parquet` | 38,504 | row_id, user, y_true, score, margin | yes | 8.19e-08 | yes |  |
| C2 | XGBoost text (logoff rows) | 1 | calib | `results/xgb_text_s1/preds_calib_final.parquet` | 35,249 | row_id, user, y_true, score, margin | yes | 7.04e-08 | yes |  |
| C2 | XGBoost text (logoff rows) | 1 | test | `results/xgb_text_s1/preds_test_final.parquet` | 95,570 | row_id, user, y_true, score, margin | yes | 8.11e-08 | yes | yes |
| C2 | XGBoost text (logoff rows) | 2 | valid | `results/xgb_text_s2/preds_valid_final.parquet` | 38,504 | row_id, user, y_true, score, margin | yes | 8.82e-08 | yes |  |
| C2 | XGBoost text (logoff rows) | 2 | calib | `results/xgb_text_s2/preds_calib_final.parquet` | 35,249 | row_id, user, y_true, score, margin | yes | 8.43e-08 | yes |  |
| C2 | XGBoost text (logoff rows) | 2 | test | `results/xgb_text_s2/preds_test_final.parquet` | 95,570 | row_id, user, y_true, score, margin | yes | 7.55e-08 | yes | yes |
| C2 | XGBoost text (logoff rows) | 3 | valid | `results/xgb_text_s3/preds_valid_final.parquet` | 38,504 | row_id, user, y_true, score, margin | yes | 7.68e-08 | yes |  |
| C2 | XGBoost text (logoff rows) | 3 | calib | `results/xgb_text_s3/preds_calib_final.parquet` | 35,249 | row_id, user, y_true, score, margin | yes | 8.57e-08 | yes |  |
| C2 | XGBoost text (logoff rows) | 3 | test | `results/xgb_text_s3/preds_test_final.parquet` | 95,570 | row_id, user, y_true, score, margin | yes | 6.96e-08 | yes | yes |
| C2 | SLM text (logoff rows) | 1 | valid | `results/c2_qwen15_r8_lr1e-4_s1/preds_valid_final.parquet` | 38,504 | row_id, y_true, score, margin | yes | 8.25e-08 | yes |  |
| C2 | SLM text (logoff rows) | 1 | calib | `results/c2_qwen15_r8_lr1e-4_s1/preds_calib_final.parquet` | 35,249 | row_id, y_true, score, margin | yes | 8.38e-08 | yes |  |
| C2 | SLM text (logoff rows) | 1 | test | `results/c2_qwen15_r8_lr1e-4_s1/preds_test_final.parquet` | 95,570 | row_id, y_true, score, margin | yes | 8.17e-08 | yes | yes |
| C2 | SLM text (logoff rows) | 2 | valid | `results/c2_qwen15_r8_lr1e-4_s2/preds_valid_final.parquet` | 38,504 | row_id, y_true, score, margin | yes | 8.23e-08 | yes |  |
| C2 | SLM text (logoff rows) | 2 | calib | `results/c2_qwen15_r8_lr1e-4_s2/preds_calib_final.parquet` | 35,249 | row_id, y_true, score, margin | yes | 8.20e-08 | yes |  |
| C2 | SLM text (logoff rows) | 2 | test | `results/c2_qwen15_r8_lr1e-4_s2/preds_test_final.parquet` | 95,570 | row_id, y_true, score, margin | yes | 7.96e-08 | yes | yes |
| C2 | SLM text (logoff rows) | 3 | valid | `results/c2_qwen15_r8_lr1e-4_s3/preds_valid_final.parquet` | 38,504 | row_id, y_true, score, margin | yes | 8.64e-08 | yes |  |
| C2 | SLM text (logoff rows) | 3 | calib | `results/c2_qwen15_r8_lr1e-4_s3/preds_calib_final.parquet` | 35,249 | row_id, y_true, score, margin | yes | 8.43e-08 | yes |  |
| C2 | SLM text (logoff rows) | 3 | test | `results/c2_qwen15_r8_lr1e-4_s3/preds_test_final.parquet` | 95,570 | row_id, y_true, score, margin | yes | 8.21e-08 | yes | yes |
| C2 | Rules (logoff rows) | - | valid | `results/rules/preds_valid_final.parquet` | 38,504 | row_id, user, y_true, score, rules_fired | no | n/a | yes |  |
| C2 | Rules (logoff rows) | - | calib | `results/rules/preds_calib_final.parquet` | 35,249 | row_id, user, y_true, score, rules_fired | no | n/a | yes |  |
| C2 | Rules (logoff rows) | - | test | `results/rules/preds_test_final.parquet` | 95,570 | row_id, user, y_true, score, rules_fired | no | n/a | yes | yes |

## Gate 3 inputs
- Comparison 1: evaluated, seeds per model [3, 3] (3 seeds)
- Comparison 2: evaluated, seeds per model [3, 3] (3 seeds)
- Calibration (7.1): to be done in P1 with the same 3 seeds.

## Problems
none
