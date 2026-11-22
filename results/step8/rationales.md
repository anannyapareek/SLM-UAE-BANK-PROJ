# Step 8.1 rationales: counts

Sample (decisions log 2026-11-15): all 243 positive test sessions + 100 negative (numpy default_rng(0)); C1 = each session's max-score row under the C1 seed-1 test scores, C2 = its logoff row; greedy generation, max 64 new tokens, from the scoring prompt; batch 1 on the lab GPU (fp16, merged seed-1 adapter). `rationales.parquet` (untracked). Examples and the faithfulness check follow in S8-2 on `rationales_v2.parquet` (decisions log 2026-11-23).

## Label/score agreement

Generated label equal to the score's side of 0.5 (p_yes > 0.5 ↔ "yes"); the decision in every reported result is the score.

| File | Model | Rows | Agree | Disagree | Disagree with \|p_yes − 0.5\| ≥ 0.01 | Max \|lab p_yes − stored score\| |
|---|---|---|---|---|---|---|
| rationales.parquet (v1: 17 Nov, repetition_penalty 1.1 inherited from the model's generation config) | C1 | 343 | 342 | 1 | 0 | 0.0107 |
| rationales.parquet (v1: 17 Nov, repetition_penalty 1.1 inherited from the model's generation config) | C2 | 343 | 333 | 10 | 10 | 0.0054 |

`rationales_v2.parquet`: not yet generated (24 Nov lab session, `docs/final_compute_lab.md`).
When present, it must show 0 disagreements outside fp16 noise (|p_yes − 0.5| < 0.01) or this report stops.

## rationales.parquet (17 Nov, repetition_penalty 1.1 inherited from the model's generation config; kept for the bug log)

| Model | Sessions | Positive | Negative | Generated "yes" | Generated "no" | Other label | Empty reason |
|---|---|---|---|---|---|---|---|
| C1 | 343 | 243 | 100 | 97 | 246 | 0 | 0 |
| C2 | 343 | 243 | 100 | 182 | 161 | 0 | 0 |

| Model | New tokens mean | Median | Min | Max | Seconds per row (median, scoring + generation) |
|---|---|---|---|---|---|
| C1 | 18.6 | 17 | 17 | 27 | 0.512 |
| C2 | 18.4 | 17 | 17 | 30 | 0.509 |

## Generated label vs score (v1, 17 Nov)

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
- 10 rows disagree outside [0.45, 0.55] (C2 10; p_yes 0.106–0.564). Scoring and generation use the same prompt string.
- In all reported results the decision is the score (p_yes), never the generated label. The rationale sample is not regenerated whatever the diagnostic finds (decisions log 2026-11-22).

## Likely cause: repetition penalty from the model's generation config (consistency check, not a metric)

`rationales.py` called `generate()` with `do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None` (and pad/eos ids); no seed is needed or set, since decoding is greedy. Everything else came from Qwen2.5-1.5B-Instruct's `generation_config.json` at the pinned revision (`do_sample: true, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1`). The merged config is greedy, so sampling predicts **0** flips; but `repetition_penalty 1.1` is not overridden and stays active (`RepetitionPenaltyLogitsProcessor` is the only logits processor). It divides the positive logit of every token already in the input by 1.1, so the label flips deterministically when exactly one of " yes" (id 9834) / " no" (id 902) appears in the prompt and the margin is small.

| Model | Rows | Observed flips | Prompts with " no" only | with " yes" only | with neither | Flips on the penalised side |
|---|---|---|---|---|---|---|
| C1 | 343 | 1 (Poisson 95% 0.0–5.6) | 0 | 0 | 343 | 0 |
| C2 | 343 | 10 (Poisson 95% 4.8–18.4) | 38 | 2 | 303 | 10 |

- C2, " no" in the prompt and p_yes < 0.5 (22 rows): the 9 flips are exactly the rows with margin ≥ -2.14; every row at ≤ -2.96 kept "no" (one threshold separates them). That implies a " no" logit of 23.5–32.6, to be measured on 24 Nov.
- C2, " yes" in the prompt and p_yes > 0.5: 1 row, flipped to "no" (margin 0.26).
- No flip on a row where the penalty cannot act (C2: 0 of 303 neither-token rows). C1 prompts contain neither token; its one flip (p_yes 0.501) is fp16 noise.
- Observed C2 flips are inconsistent with greedy decoding without a penalty (expected 0, observed 10) and consistent with the penalty in every group. Confirmation: 24 Nov diagnostic step (f) (`src/slm/rationale_diag.py`).

## Interpretation (decisions log 2026-11-23)

- Padding (cause 3) is ruled out for the 17 Nov rows: they were generated at batch 1 without padding; diagnostic step (e) stays as a check only.
- Cause accepted (2026-11-23): greedy decoding with repetition_penalty 1.1 inherited from Qwen's generation config, not plain greedy as logged on 15 Nov. The penalty acts on every generated token, not only the label: reasons cite feature names and values that are in the prompt, so all 686 v1 reason texts (C1 and C2) were generated under it. Scores are unaffected (score_texts does not use generate()).
- Fix: the rationale sample is regenerated on 24 Nov with the logged protocol (greedy, repetition_penalty=1.0 passed explicitly) to `rationales_v2.parquet`; the 17 Nov file is kept for the bug log. No reported metric changes. The v2 file must show 0 label/score disagreements outside fp16 noise (|p_yes − 0.5| < 0.01).
- Diagnostic step (f) still runs on 24 Nov to confirm the cause in the lab: (f-i) the 17 Nov arguments, 20 repeats, should reproduce the 17 Nov labels every time; (f-ii) the corrected arguments should agree with the scoring pass. If (f-i) varied across repeats, the labels would have been sampled instead.
- S8-2 (examples, faithfulness check) runs on `rationales_v2.parquet`.
