# Step 8.1 rationales: counts

Sample (decisions log 2026-11-15): all 243 positive test sessions + 100 negative (numpy default_rng(0)); C1 = each session's max-score row under the C1 seed-1 test scores, C2 = its logoff row; greedy generation, max 64 new tokens, from the scoring prompt; batch 1 on the lab GPU (fp16, merged seed-1 adapter). `rationales.parquet` (untracked). Examples and the faithfulness check follow in S8-2.

## Counts

| Model | Sessions | Positive | Negative | Generated "yes" | Generated "no" | Other label | Empty reason |
|---|---|---|---|---|---|---|---|
| C1 | 343 | 243 | 100 | 97 | 246 | 0 | 0 |
| C2 | 343 | 243 | 100 | 182 | 161 | 0 | 0 |

## Generated tokens and time per row

| Model | New tokens mean | Median | Min | Max | Seconds per row (median, scoring + generation) |
|---|---|---|---|---|---|
| C1 | 18.6 | 17 | 17 | 27 | 0.512 |
| C2 | 18.4 | 17 | 17 | 30 | 0.509 |

## Generated label vs score

In 11 of 686 rows the generated label is on the other side of 0.5 from the score p_yes: C1 1/343, C2 10/343.

| Model | Session | y_true | p_yes (rationale run) | Stored test score | Generated label |
|---|---|---|---|---|---|
| C1 | 64092 | 1 | 0.501 | 0.505 | no |
| C2 | 282868 | 1 | 0.106 | 0.107 | yes |
| C2 | 221642 | 1 | 0.158 | 0.158 | yes |
| C2 | 66877 | 1 | 0.283 | 0.284 | yes |
| C2 | 66857 | 1 | 0.347 | 0.345 | yes |
| C2 | 82660 | 1 | 0.378 | 0.376 | yes |
| C2 | 64111 | 1 | 0.388 | 0.387 | yes |
| C2 | 64116 | 1 | 0.390 | 0.388 | yes |
| C2 | 66873 | 1 | 0.405 | 0.403 | yes |
| C2 | 82682 | 1 | 0.444 | 0.445 | yes |
| C2 | 64113 | 1 | 0.564 | 0.564 | no |

- **Scoring is unaffected.** The lab p_yes matches the stored test scores within 0.011 (C1) and 0.005 (C2) over all 343 + 343 rows.
- 1 row sits within p_yes [0.45, 0.55]: C1 at 0.501: fp16 noise between the scoring pass and greedy generation.
- 10 rows disagree outside [0.45, 0.55] (C2 10; p_yes 0.106–0.564). Scoring and generation use the same prompt string; the cause is under investigation (24 Nov lab diagnostic, `src/slm/rationale_diag.py`).
- In all reported results the decision is the score (p_yes), never the generated label. The rationale sample is not regenerated whatever the diagnostic finds (decisions log 2026-11-22).
