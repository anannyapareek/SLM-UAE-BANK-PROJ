# Step 7 summary (calibration, hybrid, rules + SLM, Comparison 3, abstention, review budget)

Wrap-up (P8), 2026-11-15. Index of Step 7 outputs and their headline numbers; nothing here is
new — every number is copied from the file named next to it. Test split frozen throughout:
`data/processed/test_ids.csv` SHA-256 `5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51`,
verified before every step that touched test. Every choice (temperatures, Platt fits, band,
thresholds, capacity) was fitted on validation or calibration and logged in `decisions_log.md`
before the test numbers it affects. Session level (max over scoring points) is primary
throughout; mean ± std over seeds 1–3 and 95% user-level bootstrap CI (1,000 stratified
resamples, seed 0) from `src/eval.py`.

## Commits

| Step | What | Commit |
|---|---|---|
| P0 + 7.1 | Inventory; calibration (temperature primary, Platt sensitivity), Gate 3 pass | 410fef2 |
| 7.2 | Hybrid band on validation (X = 20%, [0.000836, 0.00307]) | ce8824c |
| 7.3 | Rules + SLM (rules first), validation | a28055b |
| 7.4 | Table 3 on test, paired tests, routing shares, compute line | e1e0e08 |
| 7.5 | Abstention (risk–coverage), session level | d10c466 |
| 7.6 | Review budget at 1 analyst × 20 alerts/day | e602989 |

Hostname clean-up (fix forward, history not rewritten): 5deb353, 4a78b28.

## Results by step

**7.1 Calibration** (`calibration/table_calibration.md`, `calibration/reliability_test.png`).
Temperature (pre-declared primary) leaves the SLM's prior shift in place (T ≈ 1; mean
predicted 5.5% vs 0.25% observed); Platt's intercept corrects both models (test session ECE:
XGBoost v3 0.0051 → 0.0016, SLM C1 0.0525 → 0.0020). Platt Brier ≈ climatology (BSS CIs span
0): the probabilities carry little skill beyond the base rate, and ranking metrics are unchanged
by either map (`calibration/ranking_check.md`). Isotonic cut.

**7.2 Hybrid** (`hybrid/band_selection.md`, `hybrid/band_sensitivity.md`). Platt row-level
scores for both models; band [0.000836, 0.00307] (XGBoost quantiles 0.73–0.92) sends 19.0% of
validation rows to the SLM; validation row PR-AUC 0.3674 vs 0.3614 for XGBoost alone. Gains at
X = 5/10/30% (sensitivity, validation only) have std about the size of the mean.

**7.3 Rules + SLM** (`rules_slm/report_valid.md`). Score = weighted rule count + SLM
probability in [0, 1). 33.6% of validation positive sessions fire no rule.

**7.4 Comparison 3** (`table3/table3.md`, `table3/paired.md`, `table3/compute.json`). Test
session PR-AUC: XGBoost 0.183, SLM 0.082, hybrid 0.181, rules 0.008, rules + SLM 0.022. Key
test hybrid − XGBoost: −0.0022 [−0.0068, +0.0015], not significant (final point: −0.0098
[−0.0259, −0.0017], significant). Hybrid − SLM and rules + SLM − rules: significant gains. The
hybrid routes 21.5% of test rows (52.7% of sessions) to the SLM. Compute line (recomputed 2026-11-22 from the Step 8.1 lab
measurements, one machine: SLM 114.5 rows/s at batch 16, XGBoost 2,455,663 rows/s on the same
PC's CPU): implied hybrid throughput 533 rows/s, 4.66× the SLM on every row. Superseded 7.4 line
(116.8 rows/s SLM, XGBoost on the Mac CPU): 544 rows/s, 4.65×. No metric changes.

**7.5 Abstention** (`abstention/abstention.md`, `abstention_main.png`, `abstention_appendix.png`).
At 10% deferred (69.8 sessions/day) system recall (upper bound): XGBoost 0.855, SLM 0.657,
hybrid 0.846, XGBoost text 0.892, SLM text 0.964. At 3% deferred (20.9 sessions/day, the
closest grid share to the 20/day capacity): 0.717, 0.492, 0.691, 0.781, 0.901. Exploratory (not
pre-declared) SLM text − XGBoost text at 10%: +0.073 [+0.026, +0.126].

**7.6 Review budget** (`review_budget/review_budget.md`, `capacity_facts.md`, three figures).
Capacity 1 analyst × 20 alerts per calendar day (2.87% of test sessions per day), set before any
curve. Recall at 20/day: XGBoost 0.339, SLM 0.244, hybrid 0.318, rules 0.099, rules + SLM 0.226,
XGBoost text 0.446, SLM text 0.542; identical to Tables 2–3 recall@k (point and CI). Pre-declared
paired SLM text − XGBoost text at 20/day: recall +0.096 [−0.069, +0.244], precision +0.0086
[−0.0054, +0.0227], not significant. Weekday-only top-k (sensitivity) changes recall by ≤ 0.005.

## Checks at wrap-up

- Fast suite: 132 passed, 2 skipped. Test checksum verified.
- `results/step7/run.csv` has rows for every Step 7 run (7.1–7.6). 7.2 and 7.3 rows were
  backfilled at wrap-up from their recorded JSON outputs (`scripts/step7_runlog_backfill.py`;
  flagged `"backfilled": true`), since those steps wrote no run-log rows when they ran.
- Tracked files: no user IDs, hostname, personal name or home path (grep at wrap-up).
- Consistency: Table 3 XGBoost/SLM/rules equal Table 1 within 1e-6 (`table3/table1_check.md`);
  review-budget recall/precision@20 equal Tables 2–3 exactly.

## Limitations and open items

- Calibrated probabilities after Platt still over-predict on test (mean 0.31–0.34% vs 0.25%):
  residual calib → test base-rate shift.
- Hybrid compute line mixed machines (SLM on the lab GPU, XGBoost on the Mac CPU). Resolved
  2026-11-22: recomputed from the Step 8.1 lab measurements (`results/step8/operations.md`).
- Calendar-day top-k (D4) gives weekends 20 alerts for ~52 sessions; weekday-only reported as
  sensitivity.
- Abstention system recall assumes analysts catch every deferred positive (upper bound).
- Commit 410fef2 (pushed) still contains the machine hostname in its history; rewriting history
  is the project's decision.
- Step 8.1: full clean rebuild at the final compute (decisions log 2026-11-09).
