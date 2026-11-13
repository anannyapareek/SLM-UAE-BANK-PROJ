# Methods notes

Points to carry into the paper's methods section. Details and reasons are in
decisions_log.md.

- **XGBoost search space:** max_depth is capped at 14 by design. The v3 best trial chose
  14, the bound, and the range was not widened further.
- **XGBoost search space (struct v3 and text): as stored in the Optuna DBs; supersedes
  Stage 4 guide ranges.** Both searches used the same space (checked 2026-11-13 from
  `results/xgb_struct_v3_tuning/optuna.db` and `results/xgb_text_tuning/optuna.db`):
  max_depth int [3, 14]; learning_rate [0.001, 0.3] log; min_child_weight int [1, 20];
  subsample [0.5, 1]; colsample_bytree [0.5, 1]; gamma [0, 5]; reg_lambda [0.001, 10] log;
  reg_alpha [0.001, 10] log. 50 trials, TPE seed 1, median pruning after 100 rounds,
  early stopping 200, at most 2,000 trees, objective row-level validation PR-AUC.
- **Model selection and reporting:** model selection for both XGBoost and the SLM uses
  row-level validation PR-AUC. Reported metrics are session-level (primary) and
  row-level (secondary).

## From the decisions log

- **Label rule (2026-11-09):** a scoring row is positive once a malicious event has
  occurred in its session (first_malicious_time ≤ score_time); earlier rows in a
  positive session are negative. The 20 scenario 3 sessions logged under the supervisor
  are kept positive, because the insider logged in as the supervisor and the session is
  what gets scored.
- **Sessions (2026-11-04):** a session runs from a Logon to the next Logoff for the same
  user and PC. A Logon within 12 hours of the session start without a Logoff is a screen
  unlock and continues the session; past 12 hours the session is closed at start + 12 h.
  The cap never fired in r4.2; 2.5% of sessions run over 12 hours to a real Logoff.
- **Behavioural baseline (2026-11-09):** z-scores compare the session with the same
  user's sessions that ended before it started and ended within the previous 30 days,
  at the same elapsed hour.
- **Splits (2026-11-09):** train 2010-02-01 to 08-31, validation 09-01 to 10-31,
  calibration 11-01 to 12-31, test 2011-01-01 to 05-17, by session start. 17.5% of users
  are held out (seed 1), with held-out insiders drawn from incidents active in the test
  period and whole incidents held out together; they appear only in test. Every incident
  with test positives has its positive sessions purged from the earlier splits.
- **Rules carry prior knowledge (2026-11-04, 2026-11-09):** the rules and keyword lists
  were written by authors who knew the public CERT scenario descriptions. Rule R1 has
  zero recall on its target scenario 1, which exfiltrates by web upload, not file copies.
- **XGBoost tuning budget (2026-11-09):** v3 used 50 Optuna trials, of which 27
  completed, in 24 minutes of trial time, with about 3.5 of 11 cores busy.
- **Evaluation (2026-11-09):** sessions are the primary unit (max row score, any
  positive row); recall at 1% and 5% FPR from the ROC curve; precision and recall at a
  per-day top-20 alert budget; 1,000 user-level bootstrap resamples stratified by
  positive users, with seeds averaged inside each resample; calibration by ECE with 15
  equal-mass bins.
