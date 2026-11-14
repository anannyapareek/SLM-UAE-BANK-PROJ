# Rules + SLM (Step 7.3, validation only)

Score = weighted rule count (no tie-break) + Platt row-level calibrated SLM C1 score in [0, 1); seed i of the SLM gives seed i of the variant. Rules alone is the single deterministic run (with its planned tie-break). Mean ± std over seeds.

Rows with at least one rule fired: 12.99%.

Positives with no rule fired: rows 991 / 1,777 (55.8%); sessions 82 / 244 (33.6%).

| Model | Row PR-AUC | Session PR-AUC |
|---|---|---|
| Rules + SLM | 0.0522 ± 0.0038 | 0.0611 ± 0.0047 |
| Rules alone | 0.0152 | 0.0226 |
| SLM alone | 0.2292 ± 0.0042 | 0.2418 ± 0.0142 |
