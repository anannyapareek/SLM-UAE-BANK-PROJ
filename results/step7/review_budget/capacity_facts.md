# Review budget (Step 7.6), part A: capacity facts and options

Computed from the processed splits and configs/eval.yaml (no model scores). Stop for the project's
capacity choice before any curve.

- eval `alerts_per_day` (configs/eval.yaml, used by precision@k / recall@k in Tables 1–3): **20** (placeholder, "TODO: justify").
- CERT r4.2: 1,000 simulated employees; 992 appear in the processed splits (train 826, valid 779, calib 751, test 889 users).
- Test: 95,570 sessions over 137 calendar days (2011-01-01 to 2011-05-17); 697.6 sessions/day on average (median 954); weekdays 963.9/day (97 days), weekends 51.9/day (40 days).
- Test positives: 243 sessions from 19 users; 1.77 positive sessions/day on average; 71 of 137 days have at least one; at most 7 on one day.
- Validation (for reference): 631.2 sessions/day (weekday 875.3), 4.00 positive sessions/day.
- Precision@k counts the top k sessions of each calendar day (weekends included).

| Option | Capacity (alerts/day) | Share of mean daily test sessions | Share of weekday sessions | Test precision@k ceiling |
|---|---|---|---|---|
| A: 1 analyst × 20 (current placeholder) | 20 | 2.87% | 2.07% | 0.089 |
| B: 1 analyst × 10 (deeper per-alert insider review) | 10 | 1.43% | 1.04% | 0.177 |
| C: 2 analysts × 25 (small team / tiered triage) | 50 | 7.17% | 5.19% | 0.035 |

Abstention link (Step 7.5): 1% deferred = 7.0 sessions/day, 2% = 14.0, 10% = 69.8.
