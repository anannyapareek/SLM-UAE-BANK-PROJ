# Runbook excerpts

The runbook sections the SLM code follows, as provided on 2026-11-09.

## 4.4 Serialisation

Fixed field order and wording, for example:

```
Role: IT Admin | Dept: Research | Tenure: 14 months
Session: 3.2 h so far, off-hours, weekday, new PC: no
USB connects: 2 | File events during USB: 37 (.zip 12, .doc 20)
External emails: 4 (3 with attachments, 2.1 MB) | Job-search visits: 6 | Leak-site visits: 0
Deviation from user's 30-day norm: USB +3.1σ, files +2.4σ
```

Output format: "Label: <yes|no>" then "Reason: <one sentence>". yes/no must be single tokens, with and without a leading space; log the ids used.

Project note: our serialiser (docs/serialisation.md) uses its own wording, which is allowed, and carries the same content:
- role and department;
- session time;
- USB and file counts;
- email and web counts;
- the deviation line.

Tenure is absent because the feature was dropped (decisions log, 2026-11-04: all users first appear in the first LDAP file, so tenure only measures calendar time).

## 5.1 Training set

- Train split only.
- Target "Label: yes|no" then "Reason: …", templated from the top deviating features.
- Downsample negatives for training only (1:5 to 1:10). Never rebalance validation, calibration or test.
- Write train.jsonl and a small valid.jsonl for the training loss.

## 5.2 LoRA grid

- lr {1e-4, 2e-4} × rank {8, 16}, seed 1.
- Fixed: alpha = 2×rank, dropout 0.05, 1 epoch, max length 512, bf16, LoRA on attention and MLP projections.
- Selected on validation PR-AUC (row level, per the decisions log).

## 6.2 Comparison 2

Append text with a fixed budget of 256 tokens per session; the truncation rule is written in the spec.

## Token budget

Max length 512 = structured prompt + 256 text tokens + target. For each tokenizer:
- measure structured-prompt tokens (median, p95, max) and target tokens (max);
- pass if p95(prompt) + 256 + max(target) ≤ 512;
- if it fails, shorten the wording (not the fields) and log it.

# Step 7 (calibration, hybrid, rules + SLM, Comparison 3, abstention, review budget)

Provided 2026-11-14. Lines marked **[Project decision]** are not runbook text: they were added by the project (2026-11-14) and are also in decisions_log.md.

## 7.1 Calibration (16 Nov, Gate 3)
- On the calibration set, compute ECE (15 bins) and Brier for XGBoost and the SLM (best configuration of each comparison).
- Temperature scaling: fit one temperature T on calibration-set margins (SLM: yes−no logit difference; XGBoost: output_margin=True). Apply to test; report ECE and Brier before and after. fit_T = minimise NLL of sigmoid(margin/T), minimize_scalar bounds (0.05, 20), method="bounded".
- Isotonic regression on the calibration set, only if time allows (first thing to cut).
- Reliability diagrams for each model, before and after.
- Gate 3: Comparisons 1 and 2 plus calibration done with 3 seeds? If not, drop Comparison 3 now.
- **[Project decision]** Platt sensitivity row; fit unit as in decision B; assert ranking metrics unchanged by temperature; report mean predicted probability vs observed rate (cal, test) as a diagnostic of prior shift.

## 7.2 Hybrid design (17 Nov)
- XGBoost (calibrated) scores every row.
- Choose the uncertain band [a, b] on validation only: the band that maximises PR-AUC while sending at most X% of rows to the SLM (X = 20%). Log a, b and X.
- Hybrid score = SLM score inside the band, XGBoost score outside. Record the share sent to the SLM.
- Runbook says copy band row IDs to the GPU and score them there. **[Project decision]** Not needed, because Comparison 1 already scored every hourly valid/calib/test row with all 3 SLM seeds; band rows are looked up. Logged as a simplification.
- **[Project decision]** Structured (C1) models only; seed i pairs with seed i; one band for all seeds. Objective = row-level validation PR-AUC averaged over the 3 pairings (same selection criterion as elsewhere); also record session-level. Candidates for a, b = quantiles of calibrated XGBoost val scores. Report top 10 bands and flatness near the optimum. If the best band is on the grid edge, widen once and log it.

## 7.3 Rules + SLM variant (17 Nov, cut third)
- Rules fire first; rank rule-fired rows by SLM score (or add the SLM score to the rule score). Log the choice.
- **[Project decision]** The choice: score = weighted rule count (without the 0.01 total-events tie-break) + calibrated SLM score in [0, 1). Rule count decides the order first; the SLM orders rows within a count; rows with 0 rules are ranked by the SLM below all fired rows. Unit-test the ordering. Report the share of positives with 0 rules fired.

## 7.4 Comparison 3 (18 Nov)
- Table 3: XGBoost, SLM, hybrid, rules, rules + SLM; 3 seeds, same metrics and CIs as Tables 1–2.
- Key paired test: hybrid vs XGBoost.
- **[Project decision]** Also hybrid vs SLM and rules+SLM vs rules; share of test rows and sessions sent to the SLM; compute line = share × SLM throughput (116.8 rows/s, RTX 4500 Ada, fp16 merged adapter). No band re-selection after test.

## 7.5 Abstention (19 Nov)
- On calibrated test scores, defer the rows closest to the decision threshold to analysts.
- For deferral shares from 0% to 50%, plot precision and recall on the remaining auto-decided rows.
- Mark one operating point (10% deferred) and report its numbers. One curve per model plus the hybrid.
- **[Project decision]** Session level. Threshold = each model/seed's validation 5%-FPR threshold. Uncertainty = |logit(p) − logit(threshold)|. Also report system recall assuming analysts catch every deferred positive (labelled as an upper bound), and the deferral workload in sessions per test day. Main figure: C1 models + hybrid; C2 models in the appendix.

## 7.6 Review budget (19 Nov)
- State the assumed analyst capacity (N analysts × M alerts per day) and justify it.
- For each model and the hybrid, rank rows by score within each day; plot recall against alerts per analyst-day.
- Mark the assumed capacity on the plot.
- **[Project decision]** Before any curve, report eval.ALERTS_PER_DAY (placeholder 20, already used for precision@k in Tables 1–2), CERT r4.2 user count, test days and sessions per day, and 2–3 capacity options. Then stop for the project's choice. If the chosen value differs from 20, rerun precision@k for Tables 1–3 and log it. Second figure: C2 XGBoost-text vs SLM-text, with a paired test at the chosen capacity.
