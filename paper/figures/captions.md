# Figure captions

Venue: TODO. Colours and markers: one per model (Okabe-Ito) in every figure. Thin lines = each seed (1–3); thick = mean over seeds; no confidence intervals are drawn: 95% user-level bootstrap CIs (1,000 resamples, stratified, seed 0, src/eval.py) are given at the marked points in the paper tables.

## reliability_test

Reliability diagrams on the test split, unit = session (C1: max over scoring points; C2: the logoff row), n = 95,570 sessions (243 positive, 889 users) per model and seed. One panel per model (C1 XGBoost, C1 SLM, C2 XGBoost text, C2 SLM text); uncalibrated (dotted, open markers), temperature scaling (dashed, open; pre-declared primary) and Platt scaling (solid, filled; sensitivity), both fitted on the calib split at the session level. 15 equal-mass bins, one line per seed; log–log axes; bins with no observed positive are drawn at the 1e-5 floor. No CIs drawn (calibration table: ECE, Brier, BSS with 95% CIs).

## abstention_main / abstention_appendix

Abstention on the test split, unit = session, n = 95,570 (243 positive). Each model and seed defers the sessions closest to its validation 5%-FPR threshold (calibrated logit distance) to analysts and alerts on the rest. Main: XGBoost (C1), SLM (C1), Hybrid (C1); appendix: XGBoost text (C2), SLM text (C2). Vertical lines: 3% deferred (≈ 20.9 sessions/day, close to the capacity of 20 alerts/day; solid line) and 10% (operating point, 69.8 sessions/day; dashed). System recall assumes analysts catch every deferred positive (upper bound). Precision and recall on auto-decided sessions become unstable once few alerts or positives remain (e.g. the SLM beyond ~25% deferred, where its kept alerts run out; counts per point and seed in results/step7/abstention/curves.csv). Curves: 0–50% in steps of 1%; no CIs drawn (abstention table: 2%, 3%, 10% with 95% CIs).

## review_budget_main / review_budget_c2

Recall vs alerts per analyst-day on the test split, unit = session, n = 95,570 (243 positive), calendar-day top-k over 137 days (weekends included). Main: XGBoost (C1), SLM (C1), Hybrid (C1), Rules, Rules + SLM; C2: XGBoost text (C2), SLM text (C2). Solid line: capacity 20/day (1 analyst); dashed: 10 and 50/day (sensitivity). Rules are deterministic (one run). No CIs drawn (review-budget table at 20/day with 95% CIs).

## review_budget_weekday (appendix)

As review_budget_main and _c2, with weekday-only top-k (97 weekdays; weekends get no alerts, so their 1 positive session counts as missed). Test split (frozen), 95,570 sessions, 243 positive, 889 users, 137 days.
