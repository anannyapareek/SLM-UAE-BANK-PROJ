# Freeze, 2026-11-23 (tag `frozen-2026-11-23`)

Decisions log 2026-11-23. After the freeze: no new runs that change a reported number; allowed are bug
fixes that change no design decision, the 24 Nov final compute, the rationale diagnostic and the v2
rationale regeneration.

## Results-level rebuild in a clean clone

- Code: clean clone of the pushed commit `d94eea5` (the parent of the freeze commit), Mac (arm64,
  Darwin), fresh environment: Python 3.12.16 via uv, `requirements-mac.txt` (SHA-256 in the manifest).
- Inputs: the stored predictions, processed data and Step 8 inputs, copied copy-on-write from the
  working tree; test set checksum `5ace66c2…818e51` verified. Nothing retrained (XGBoost included).
- Run: `python -m scripts.freeze_rebuild run --workers 3`. 58 jobs (79 eval.py JSONs re-run through
  src/eval.py, calibration, hybrid, rules+SLM, Table 3, abstention, review budget, text gain,
  fairness, time to detection, Comparison 1/2 tables, Step 8 report, figures, paper tables):
  58 done, 0 failed, 4,853 s (`rebuild_run.json`).
- Compare: `python -m scripts.freeze_rebuild compare` against the frozen tree (`rebuild.json`),
  tolerance 1e-6 (float32), timing fields ignored (runtime_seconds, fit_seconds, date, …).

| Regenerated files | Count |
|---|---|
| Byte-identical | 205 |
| Within 1e-6 | 122 |
| PDF differs in metadata only (PNG twin identical) | 6 |
| Run logs (not compared) | 3 |
| Differ (metadata only, below) | 2 |
| **Total** | **338** |

Differences, all metadata only; no reported number differs:

- 34 calibration files (`results/step7/calibration/fits.json` and 33 `*/s*/params.json`) differ only
  in `fit_seconds` (wall time) and `date`; every fitted parameter matches. Counted above as within
  1e-6 once `fit_seconds` was added to the timing fields.
- `results/step7/hybrid/band_selection.json`: the rebuilt `config` block has two extra keys,
  `sensitivity_caps` and `build_splits`, from the later band-sensitivity code; the band and every
  number match.
- `results/granularity/xgb_struct_v3_time_to_detection.json` (untracked) gains `thresholds_from` and a
  per-seed `session_fpr_at_threshold`, written by code added in c755613 after this file was made;
  every number in the frozen file matches.

Rebuild driver bug, fixed; rerun within 1e-6: `scripts/freeze_rebuild.py` did not pass `--check` to
`src.text_gain`, so the rebuilt `text_gain_did.json` lacked `draws_checked_against`. Fixed in the
freeze commit; the rerun confirmed the shared bootstrap draws against all 3 paired tests and is
within 1e-6.

The frozen files are kept as they are; the manifest hashes the frozen versions, not the rebuilt ones.
20 tracked files under results/ are not produced by the rebuild (Gate 1 and Step 3 reports, GPU and
lab timing records, inventories, `run.csv`, token checks) and are covered by the manifest only;
listed in `rebuild.json` (`tracked_not_regenerated`).

## Data rebuild

Full data rebuild not re-run at freeze; verified at Gate 1 (9–10 Nov).

## Manifest

`results/freeze/MANIFEST.sha256`: 488 files (everything under results/ except results/freeze/,
data/processed/, data/step8/, requirements-mac.txt, requirements-gpu.txt).
`python -m scripts.freeze_manifest verify` checks the files present (the lab PC holds a subset).
Not covered: raw CERT data (Gate 1), model weights and LoRA adapters (lab PC).

## Also at the freeze

- `docs/bug_log.md`: three entries (rationale repetition penalty; 9 Nov concurrent commits;
  410fef2 hostname), none changes a design decision.
