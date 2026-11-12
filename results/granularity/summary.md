# Scoring granularity: hourly vs logoff-only (XGBoost v3, validation)

Validation only; no test data. XGBoost v3 seeds 1–3 (`results/xgb_struct_v3_s{1,2,3}/preds_valid.parquet`), evaluated with `src/eval.py`:
- **Hourly:** the session score is the max over all scoring points (the default).
- **Logoff-only:** `--session-agg final`, the score at the session's final scoring point.

In both cases the session label is "any positive row". Validation has 38,504 sessions, of which 244 are positive (0.63%). Every validation session's last scoring point is its logoff (`is_final`) row.

## Session-level metrics (mean of 3 seeds, 95% user-bootstrap CI)

| Metric | Hourly (max over points) | Logoff-only (final point) |
|---|---|---|
| PR-AUC | 0.416 [0.219, 0.637] | 0.499 [0.298, 0.724] |
| Recall at 1% FPR | 0.507 [0.307, 0.750] | 0.583 [0.360, 0.841] |
| Recall at 5% FPR | 0.622 [0.412, 0.868] | 0.683 [0.495, 0.905] |
| Precision at k (20/day; ceiling 0.200) | 0.113 [0.064, 0.165] | 0.124 [0.071, 0.184] |
| Recall at k | 0.563 [0.351, 0.799] | 0.619 [0.408, 0.855] |

Files:
- `xgb_struct_v3_hourly_valid.json`, which is identical to `results/eval/xgb_struct_v3_valid.json` apart from name and runtime;
- `xgb_struct_v3_logoff_valid.json`.

No paired test was run: `eval.py paired` compares two models under one aggregation, not one model under two.

## Time to detection, hourly scoring (`xgb_struct_v3_time_to_detection.json`)

**Rule:** the alert is the first scoring point with score ≥ the seed's session-level 5%-FPR threshold. The thresholds are 0.00164, 0.00142 and 0.00128, taken from `eval.py`. For each seed they reproduce `eval.py`'s recall at 5% FPR exactly.

**Context:**
- 211 of the 244 positive sessions have more than one scoring point.
- The first malicious event comes a median 0.61 h after session start (IQR 0.08–1.84).
- The first positive hourly scoring point comes a median 0.57 h after that event (IQR 0.23–0.85). That gap is the latency of the hourly grid itself.

| Pooled over 3 seeds (455 detections) | Median | IQR |
|---|---|---|
| Hours from session start to alert | 1.0 | 1.0–1.0 |
| Hours from first malicious event to alert | 0.37 | −0.02–1.00 |
| Hours by which the alert precedes logoff | 7.0 | 0.76–8.05 |

- Detected at 5% FPR: 62.2% of positive sessions (60.2–64.8% across seeds).
- **Detected before logoff:** 49.3% of all positive sessions, which is 79.3% of those detected.
- **Alerts before the first malicious event:** 39–40 sessions per seed (about 25% of detections). These sessions already looked anomalous before the malicious event, so the alert is earlier than the event itself.

## What mid-session scoring adds for XGBoost

1. **Ranking is worse with hourly scoring.** Taking the max over a session's hourly points lowers session-level ranking: PR-AUC is 0.416 against 0.499, and recall at 5% FPR is 0.62 against 0.68. Long negative sessions get many chances to spike, which raises the 5%-FPR threshold about eightfold (0.0014 against 0.0002). The confidence intervals overlap widely.
2. **Hourly scoring is earlier.** Mid-session scoring buys timeliness: about half of all positive sessions (79% of the detected ones) are flagged before logoff, a median 7 h before it, and a median 0.4 h after the first malicious event. That is close to the grid's own 0.57 h latency.
3. **The trade-off:** hourly is the choice when time to response matters, and logoff-only when only the session verdict matters and compute is limited. Logoff-only scores 169,323 rows against 1,494,287 across valid, calib and test, about 9× fewer.
