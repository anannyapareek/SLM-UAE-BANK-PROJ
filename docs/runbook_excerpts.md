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
