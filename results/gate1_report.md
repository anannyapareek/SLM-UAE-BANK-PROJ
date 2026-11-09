# Gate 1: reproducibility check (Mac side)

**Result: PASS.** A rebuild from raw data in a fresh clone at commit `814a63d` reproduces the frozen test set, all processed tables, the leakage report, the XGBoost v3 validation results (seeds 1–3) and the rules validation results, byte for byte.

- Date: 2026-11-09
- Commit reproduced: `814a63d6096227c77ebf7aed0e0c7432ee25e4a3`
- Frozen test set SHA-256: `5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51`
- GPU side: pending (lab, 10 Nov)

## Setup

- Fresh `git clone` of `origin` checked out at `814a63d`; the working tree was clean.
- A fresh venv was built from `requirements-mac.txt`. Every package installed and `pip check` was clean. The installed set matches the main repo's venv exactly; the only difference is that the clone's venv has pip.
- `data/raw` was a symlink to the main repo's raw data and was only read. Nothing was copied from the main repo's `data/processed/` or `results/`.
- Before the rebuild, the clone's `data/processed/` and `results/` held only tracked files. The tracked `test_ids.csv` was moved aside as `test_ids.tracked.csv` so that the leakage stage had to freeze it again.
- `--force-i-know` was never passed, and XGBoost was not tuned again.

| | Main repo | Clone |
|---|---|---|
| Python | 3.12.16 | 3.12.16 |
| XGBoost | 3.4.1 | 3.4.1 |
| XGBoost threads | `n_jobs=-1` (11 logical cores) | `n_jobs=-1` (11 logical cores) |
| OMP_NUM_THREADS | unset | unset |
| Machine | Apple M3 Pro, 5 performance + 6 efficiency cores, 18 GB, macOS 14.4 | same machine |

## Checks

| Check | Result | Detail |
|---|---|---|
| a. test_ids.csv | PASS | The SHA-256 equals the frozen hash, the tracked copy and the main repo's copy (848,137 rows) |
| b. sessions.parquet | PASS | byte-identical |
| b. features.parquet | PASS | byte-identical |
| b. features_text.parquet | PASS | byte-identical (7.9 GB) |
| b. labels.parquet | PASS | byte-identical |
| b. splits.parquet | PASS | byte-identical |
| b. heldout_users.csv | PASS | byte-identical |
| c. Leakage report | PASS | Text identical to the main repo; checks (a)–(h) all PASS with the same numbers |
| d. XGBoost v3 validation preds, seed 1 | PASS | byte-identical, max abs score diff 0, best iteration 356; `model.json` also byte-identical |
| d. XGBoost v3 validation preds, seed 2 | PASS | byte-identical, max abs score diff 0, best iteration 450 |
| d. XGBoost v3 validation preds, seed 3 | PASS | byte-identical, max abs score diff 0, best iteration 770 |
| e. Rules validation preds | PASS | byte-identical; `metrics.json` and `rules_methods.md` identical |
| f. eval `xgb_struct_v3_valid.json` | PASS | identical except `runtime_seconds` |
| f. eval `rules_valid.json` | PASS | identical except `runtime_seconds` |
| f. eval `xgb_v3_vs_rules_valid.json` | PASS | identical except `runtime_seconds` |
| g. Fast test suite (`pytest -m "not slow"`) in the clone | PASS | 41 passed, 4 deselected |

## Wall-clock per stage (clone)

| Stage | Command | Seconds |
|---|---|---|
| data | `python src/prep.py` | 57 |
| sessions | `python run.py sessions` | 24 |
| features | `python run.py features` | 53 |
| features_text | `python run.py features_text` | 387 |
| labels | `python run.py labels` | 4 |
| splits | `python run.py splits` | 1 |
| leakage | `python run.py leakage` | 33 |
| xgb (fit seeds 1–3, no tuning) | driver script, see below | 116 |
| rules | `python run.py rules` | 2 |
| eval xgb v3 | `python run.py eval single …` | 27 |
| eval rules | `python run.py eval single …` | 7 |
| eval paired | `python run.py eval paired …` | 31 |
| **Total** | | **742 (12.4 min)** |

## Reproducibility gaps to fix

1. **No `data` stage in `run.py`.** Raw-to-parquet conversion is `python src/prep.py`, which only the Makefile's `stage2` target calls. A `run.py prep` (or `data`) stage would close the gap.
2. **No fit-only XGBoost.** `python run.py xgb` always runs the 50-trial Optuna search and then overwrites `configs/xgb_best.yaml` and appends to `decisions_log.md`, so it cannot refit the frozen config. Gate 1 used a driver script that reads `configs/xgb_best.yaml` and calls `src.baselines.xgb.load()` and `train_seed()` for seeds 1–3. Before fitting, the script asserted that the computed `scale_pos_weight` (440.38788522848034) and the categorical levels equal the yaml's. Suggested fix: a `--fit-only` flag (or a `xgb_fit` stage) that reads `xgb_best.yaml`.
3. **Eval arguments typed by hand.** The `--name`, `--preds` and `--out` values for the three eval runs were taken from the `preds` fields in the main repo's eval JSONs. Suggested fix: a Makefile target or config that lists the standard eval runs.
4. **`features_text` was missing from the Gate 1 runbook's stage list.** Four fast tests in `tests/test_leakage.py` need `features_text.parquet`: `test_text_events_fall_inside_their_session`, `test_text_sessions_are_in_features`, `test_text_sort_order` and `test_file_headers_are_stripped`. The runbook should list the stage after `features`.

Other observations:
- No hard-coded user paths were found in `src/`, `tests/`, `configs/`, `run.py`, the `Makefile` or `pytest.ini`.
- No package is missing from `requirements-mac.txt`.
- When `test_ids.csv` is absent, the leakage stage freezes it again and appends a "Frozen test_ids.csv" line to `decisions_log.md`. The XGBoost fit appends three rows to `results/run.csv`. In the clone these were the only changes to tracked files. The re-frozen hash equals the frozen hash, and the regenerated `labels_report.md` and `leakage_report.md` matched the tracked versions.
