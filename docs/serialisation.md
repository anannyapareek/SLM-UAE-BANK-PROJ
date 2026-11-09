# Serialisation

How one scoring row of `features.parquet` becomes SLM text.
- Code: `src/slm/serialise.py`.
- Tests: `tests/test_serialise.py`.
- Runbook spec: `docs/runbook_excerpts.md` (4.4).

This is pure Python; no model or GPU is involved.

## Example

A training example: the scoring prompt, then the completion (`Label:` onwards). The label shown is illustrative.

```
Is this session malicious? [n] = last hour.
Session: hour 3, working hours, weekday, known PC
User: ElectricalEngineer; 2 - ManufacturingEngineering; 3 - Manufacturing
Logons: unlocks 0, today 1
USB: 0 [0], today 0
Files: 0 [0]; doc 0 pdf 0 zip 0 exe 0 jpg 0 txt 0; mismatch 0
Email: sent 9 [0], after hours 0; external 12 [0]; attachments 5 [0]; 248 KB
Web: visits 66; job 14 [2]; cloud 0 [0]; leak 0 [0]; hacking 0
30-day z: USB 0.0 files 0.0 emails 2.9 external 4.5 job 11.5 cloud -0.8 leak 0.0
Label: yes
Reason: highest 30-day z: job-search visits +11.5, external recipients +4.5.
```

## Functions

| Function | Output | Used by |
|---|---|---|
| `prompt(row)` | The header line and the eight field lines | — |
| `scoring_prompt(row)` | `prompt(row) + "\nLabel:"` | Scoring (runbook 5.4) and training |
| `target(row, label)` | `" yes"` or `" no"`, then `"\nReason: <one sentence>"` | Training completion (runbook 5.1) |
| `training_text(row, label)` | `scoring_prompt(row) + target(row, label)` | — |

There is no chat template: base and instruct models see the same plain text in training and scoring (decisions log, 2026-11-09). The scorer reads the next-token logits of ` yes` and ` no` after `Label:`.

## Field order and wording

The order and wording are fixed. Each line reads only the columns listed, and each of the 44 features in `configs/features.yaml` appears exactly once. Counts are so far in the session; `[n]` is the count in the last hour (rolling 1 h), as the header says.

| Line | Text | Columns |
|---|---|---|
| Header | `Is this session malicious? [n] = last hour.` | — |
| `Session:` | `hour <h>, <off-hours\|working hours>, <weekend\|weekday>, <new PC\|known PC>` | session_minutes_so_far (h = minutes // 60), off_hours, weekend, new_pc |
| `User:` | `<role>; <department>; <functional_unit>` | role, department, functional_unit |
| `Logons:` | `unlocks <n>, today <n>` | unlocks_so_far, logon_events_today_so_far |
| `USB:` | `<connects> [<1h>], today <n>` | usb_connects, usb_connects_1h, usb_connects_today_so_far |
| `Files:` | `<copies> [<1h>]; doc <n> pdf <n> zip <n> exe <n> jpg <n> txt <n>; mismatch <n>` | file_events, file_events_1h, file_ext_*, file_header_mismatch |
| `Email:` | `sent <n> [<1h>], after hours <n>; external <n> [<1h>]; attachments <n> [<1h>]; <k> KB` | emails_sent(_1h), after_hours_emails, external_recipients(_1h), attachments_count(_1h), email_bytes (k = bytes / 1024, rounded) |
| `Web:` | `visits <n>; job <n> [<1h>]; cloud <n> [<1h>]; leak <n> [<1h>]; hacking <n>` | http_visits, http_job_search(_1h), http_cloud_storage(_1h), http_filesharing_leak(_1h), http_hacking_tools |
| `30-day z:` | `USB <z> files <z> emails <z> external <z> job <z> cloud <z> leak <z>` | the seven `*_z` columns; one decimal, `-0.0` written `0.0`, `n/a` if missing |

The runbook's 4.4 example uses different wording, which is allowed. The same content is here: role and department, session time, USB and file counts, email and web counts, and the deviation line. Tenure is absent because that feature was dropped (decisions log, 2026-11-04).

## Excluded columns

These columns of `features.parquet` are not features and never enter the text:

| Column | Reason |
|---|---|
| row_id, session_id, pc | identifiers |
| user | identifier; the model must not memorise users |
| score_time | metadata; time of day enters only through off_hours and after_hours_emails |
| is_final | metadata only, never a feature (decisions log) |

## Labels

- **The prompt has no labels.** It reads feature columns only, and label columns in the input row are ignored (tested).
- **The label is an argument.** `target()` takes it explicitly and uses it only for the `yes`/`no` word.
- **`serialise.py` names no label column and reads no label file.** The isolation tests treat it as ordinary, label-blind code.

## Reason

The reason is one sentence, templated from the features alone, so it is the same for a positive and a negative row with identical features:

- **Which deviations count:** z-score features with z ≥ 2. The two largest are named in descending order, with ties broken by name.
- **Wording when some count:** "highest 30-day z: job-search visits +11.5, external recipients +4.5."
- **Wording when none count:** "no 30-day z is 2 or more."

## Answer tokens

`yes` and `no` are single tokens, with and without a leading space, for all three models (`results/token_check.json`, 2026-11-09). The scorer uses the leading-space ids, because the word follows `Label:`. Appending the answer never merges with the prompt's last token (0 mismatches in 200 prompts per model).

| Model | `yes` | `no` | ` yes` (used) | ` no` (used) |
|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 9693 | 2152 | 9834 | 902 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 10407 | 4607 | 9805 | 787 |
| Qwen/Qwen2.5-0.5B-Instruct | 9693 | 2152 | 9834 | 902 |

## Token budget

Max length 512 = structured prompt + 256 text tokens (Comparison 2) + target. Measured on 5,000 random train rows (seed 1) by `python -m src.slm.check_tokens`:

| Model | Prompt median | Prompt p95 | Prompt max | Target max (incl. EOS) | p95 + 256 + target |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 197 | 202 | 209 | 30 | 488 (pass) |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 205 | 209 | 217 | 33 | 498 (pass) |
| Qwen/Qwen2.5-0.5B-Instruct | 197 | 202 | 209 | 30 | 488 (pass) |

The first wording failed the budget (SmolLM2: 258 + 256 + 42 = 556). The wording was then shortened without dropping any field (decisions log, 2026-11-09):
- `[n]` replaces "(last hour n)";
- shorter labels;
- file types written as `doc 16 pdf 7 …`;
- session flags as words;
- no `+` or commas in the z line;
- a shorter reason.

The truncation rule for the appended text (runbook 6.2) is set with Comparison 2.
