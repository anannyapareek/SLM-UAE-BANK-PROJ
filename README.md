# SLM-ZTA: small language models vs gradient boosting for insider-threat session risk scoring

A benchmark on the CERT Insider Threat dataset r4.2. Each user session is scored for risk, the
way a zero-trust access layer would score it. Three model families are compared:

- **SLM:** Qwen2.5-1.5B-Instruct fine-tuned with LoRA; it answers yes/no, scored from the fp32
  " yes"/" no" logits.
- **XGBoost** on the same features.
- **A hand-written rules baseline.**

The work is split into two comparisons:

- **Comparison 1 (C1):** structured features, hourly scoring points.
- **Comparison 2 (C2):** structured features plus event text, scored at logoff only.

Step 7 adds calibration, an XGBoost→SLM hybrid, rules+SLM, abstention, review budget, fairness and
time to detection. Step 8 adds latency, rationales and the paper figures and tables.

Everything is frozen at tag **`frozen-2026-11-23`** (commit `f94c5c5`). Every reported number comes
from `src/eval.py`, the only metric implementation. Results are reported as mean ± std over 3 seeds
[95% user-level bootstrap CI]. Every design decision is in [`decisions_log.md`](decisions_log.md).

## Machines

| Tag | Machine | Used for |
|---|---|---|
| [Mac] | Mac (arm64, Darwin): Apple M3 Pro, 18 GB, macOS 14.4, Python 3.12.16 (uv) | data, features, splits, XGBoost, rules, evaluation, Step 7, tables, figures |
| [GPU] | Lab PC (RTX 4500 Ada, Windows 11): 24 GB, driver 581.80 (CUDA 13.0), torch 2.11.0+cu128, Python 3.12.16 | LoRA training, SLM scoring, latency, rationales |

Environments:
- [`requirements-mac.txt`](requirements-mac.txt): exact pins.
- [`requirements-gpu.txt`](requirements-gpu.txt): exact pins from the lab PC, `torch==2.11.0+cu128`.
- [`requirements-gpu.in`](requirements-gpu.in): GPU inputs; torch is installed first, for the
  driver's CUDA version. See [`docs/gpu_setup.md`](docs/gpu_setup.md) §3.

## Checksums and how to verify

- Frozen test IDs, `data/processed/test_ids.csv` (848,137 row IDs): SHA-256
  `5ace66c27da40b2bf3d9d5eb9706322340cd16f2b83fb5ceeb8fe58571818e51`. Every script that touches
  the test split checks this first (`src.splits.verify_test_checksum`).
- Freeze manifest, [`results/freeze/MANIFEST.sha256`](results/freeze/MANIFEST.sha256): SHA-256 of
  488 files. It covers results/ (except results/freeze/), data/processed/, data/step8/,
  requirements-mac.txt and requirements-gpu.txt. Verify it from the repo root:

  ```bash
  python -m scripts.freeze_manifest verify     # expected: "488 OK, 0 mismatched, 0 not present here" on the Mac
  ```

  On a machine with only some of the files, missing files count as "not present here"; any
  mismatch exits 1.
- How the freeze was checked: [`results/freeze/freeze.md`](results/freeze/freeze.md). A
  clean-clone rebuild of every Mac-side result from the stored predictions matched the frozen
  tree: 58 jobs, 0 failures, no reported number differs.
- Text blocks for C2, `data/processed/slm_text_blocks.parquet`: SHA-256 `2568539b…af8faa8`
  (decisions log 2026-11-12).

## Seeds

| What | Seeds |
|---|---|
| Global, splits, training sample (1:10 negatives), text features | 1 (`configs/global.yaml`, `splits.yaml`, `slm.yaml`, `features_text.yaml`) |
| XGBoost (structured v3 and text) | `random_state` 1, 2, 3 on fixed data; Optuna TPE seed 1 |
| SLM LoRA | 1, 2, 3 (LoRA init, data order, dropout; data fixed). C1 seed 1 is the Gate 2 grid adapter, trained before the reseeding change 1c10259 (decisions log 2026-11-12) |
| Bootstrap CIs | 1,000 user-level resamples, seed 0 (`configs/eval.yaml`) |
| Per-day top-k tie break | seed 0 (`configs/eval.yaml`) |
| Step 7 | seeds 1–3, seed *i* paired with seed *i* (`configs/step7.yaml`) |

## Run order: raw CERT r4.2 → every table and figure

Put CERT r4.2 in `data/raw/r4.2/` and the answer key in `data/raw/answers/`. The data is not
redistributable; see [`docs/licence_facts.md`](docs/licence_facts.md). Run from the repo root.

The times are recorded runtimes:
- Gate 1 rebuild from raw (`results/gate1_report.md`);
- GPU queue logs (`logs/summary 2.txt`, decisions log 2026-11-10/11);
- the freeze rebuild (`results/freeze/rebuild_run.json`).

### 0. Environment [Mac]

```bash
uv venv --python 3.12.16 .venv && uv pip install -r requirements-mac.txt
make test-fast
```

### 1. Data, features, splits, baselines [Mac] (Gate 1 rebuild: 12.4 min in total)

```bash
python run.py all          # = the stages below, in order, stopping at the first failure
```

| Stage | Command | Time | Output |
|---|---|---|---|
| raw → parquet | `python run.py data` | 57 s | data/processed/*.parquet (events) |
| sessions | `python run.py sessions` | 24 s | sessions.parquet |
| features | `python run.py features` | 53 s | features.parquet (hourly scoring points) |
| event text | `python run.py features_text` | 387 s | features_text.parquet (7.9 GB) |
| labels | `python run.py labels` | 4 s | labels.parquet |
| splits | `python run.py splits` | 1 s | splits.parquet, heldout_users.csv |
| leakage + test freeze | `python run.py leakage` | 33 s | test_ids.csv (must give the checksum above), results/leakage_report.md |
| text features (C2) | `python run.py text_features` | ≈ 7 min | features_with_text.parquet |
| XGBoost structured | `python run.py xgb --fit` | 116 s | results/xgb_struct_v3_s{1,2,3}/ (from configs/xgb_best.yaml; `--tune` reruns the 50-trial search, 24.0 min) |
| rules | `python run.py rules` | 2 s | results/rules/ (configs/rules.yaml) |
| validation evals | `python run.py eval-baselines` | ≈ 1 min | results/eval/ |

Also on the Mac:
- Text XGBoost: `bash scripts/run_xgb_text.sh` (50-trial search, 76.2 min, plus seeds 1–3).
- Test and calib scoring: `python run.py xgb --score --splits valid calib test --allow-test`
  (`--variant text` for C2) and `python run.py rules --score --splits valid calib test --allow-test`.

Then build the GPU inputs:
- `python -m src.slm.text_blocks --allow-test`, which writes `data/processed/slm_text_blocks.parquet`;
- the data bundle, `bash scripts/make_bundle.sh` → `bundle/slm_bundle.tar`;
- for Step 8, `python -m src.slm.step8_inputs build`.

### 2. SLM training and scoring [GPU]

Setup, bundle check, model download (pinned revisions in `configs/slm.yaml`), token check and
smoke test are in [`docs/gpu_setup.md`](docs/gpu_setup.md). The prompt format is in
[`docs/serialisation.md`](docs/serialisation.md).

| Step | Command | Recorded time |
|---|---|---|
| LoRA grid, seed 1 (4 configs: lr 2e-4/1e-4 × rank 16/8), hourly validation scoring | `scripts/run_queue.ps1` | 5.9 h (≈ 1.5 h per config) |
| C1 seeds 2–3 training, then C1 scoring (valid/calib/test, hourly, batch 16) | `scripts/run_night2.ps1 -IncludeText` (Phases A–B) | training 38.5 min per seed; scoring per seed: test 122.7 min, valid 49.0 min, calib 45.0 min |
| C2 training seeds 1–3 (text), scoring logoff rows (batch 8, `--text --only-final`) | same script, Phase C | training 31.2 / 109.9 / 89.1 min; scoring per seed: test 27.7 min, valid 11.3 min, calib 10.3 min |
| Step 8.1 latency and rationales | [`docs/step8_lab.md`](docs/step8_lab.md) | latency (SLM and XGBoost) and 686 rationales (343 C1, 343 C2) |
| Final compute (C2 seed 1 re-scored), rationale diagnostic, rationales v2 | [`docs/final_compute_lab.md`](docs/final_compute_lab.md) | ≈ 1 h |

Bring the prediction files back to `results/<run>/` (for example
`results/c1_qwen15_r8_lr1e-4_s{1,2,3}/`, `results/c2_qwen15_r8_lr1e-4_s{1,2,3}/`), then
`python scripts/make_final_copies.py` writes the logoff-only copies.

### 3. Evaluation, Step 7 and tables [Mac] (freeze rebuild: 81 min with 3 workers)

`python -m scripts.freeze_rebuild run` runs every step below from the stored predictions, in
dependency order. Run it in a clean clone with the results copied in; see its docstring.
Individually:

| Step | Command | Time (freeze rebuild, per job) |
|---|---|---|
| Comparison 1 evals (79 eval JSONs over C1, C2, validation, granularity) | `python run.py eval single\|paired …` (arguments as recorded in each JSON) | 12 s – 10.6 min each |
| Calibration (Platt, temperature; session/row fit) | `python -m src.calibrate`, `scripts.step7_calibration_eval`, `scripts.step7_calibration_tables` | 15 s + 78.7 min + 10 s |
| Hybrid | `python -m src.hybrid select` / `sensitivity` / `build` | 8.1 + 30.9 + 1.2 min |
| Rules + SLM | `python -m src.rules_slm build` / `report` | 33 s |
| Table 3 | `python -m scripts.step7_table3_eval`, `scripts.step7_table3` | 55.3 min + 2 s |
| Abstention | `python -m src.abstention`, `scripts.step7_abstention_report` | 6.5 min |
| Review budget | `python -m src.review_budget` (+ report) | 10.0 min |
| Text gain / DiD | `python -m src.text_gain …` | 33 s |
| Fairness, time to detection | `python -m src.fairness …`, `python -m src.granularity …` | ≈ 5 s each |
| Comparison 1/2 tables | `python -m scripts.comparison1_tables`, `scripts.comparison2_tables` | 1 s |
| Step 8 report (operations, rationales) | `python -m src.slm.step8_report` | 1 s |
| Paper figures and tables | `make paper` (= `python -m src.figures` + `python -m src.paper_tables`) | 4 s |
| Final-compute check (after the 24 Nov lab run) | `python -m src.final_check` | < 1 min |

Tests: `make test-fast` (≈ 40 s) and `make test` (full suite, ≈ 2 min).

## Which script produces which table and figure

`make paper` reads results/ only. `paper/tables/provenance.json` records the source file and JSON
path of every number in every table. Figure captions are in `paper/figures/captions.md`.

| Paper output | Producer | Inputs (results/) |
|---|---|---|
| `paper/tables/table1` (C1: SLM vs XGBoost vs rules) | `src.paper_tables` | comparison1/eval/*.json (`src/eval.py`) |
| `paper/tables/table2` (C2, logoff rows) | `src.paper_tables` | comparison2/{slm_text,xgb_text,rules}_test_final.json, paired_slm_text_vs_xgb_text.json |
| `paper/tables/text_gain` (C1→C2 gain, DiD) | `src.paper_tables` | comparison2/text_gain_did.json (`src.text_gain`), paired_*_vs_*_struct.json |
| `paper/tables/table3` (hybrid, rules+SLM) | `src.paper_tables` | step7/table3/ (`scripts.step7_table3_eval`, `scripts.step7_table3`) |
| `paper/tables/calibration` | `src.paper_tables` | step7/calibration/ (`src.calibrate`, `scripts.step7_calibration_*`) |
| `paper/tables/abstention` | `src.paper_tables` | step7/abstention/ (`src.abstention`) |
| `paper/tables/review_budget` | `src.paper_tables` | step7/review_budget/ (`src.review_budget`) |
| `paper/tables/fairness_c1`, `fairness_c2` | `src.paper_tables` | comparison{1,2}/fairness_*_test.csv (`src.fairness`) |
| `paper/tables/operations` (latency, throughput) | `src.paper_tables` | step8/operations.csv (`src.slm.step8_report`) |
| `paper/figures/reliability_test` | `src.figures` | step7/calibration/ |
| `paper/figures/abstention_main`, `abstention_appendix` | `src.figures` | step7/abstention/ |
| `paper/figures/review_budget_main`, `review_budget_c2`, `review_budget_weekday` | `src.figures` | step7/review_budget/ |

The intermediate tables (results/comparison1/*.md, comparison2/*.md, step7/*/…md, step8/*.md)
are written by the Step 7/8 scripts above.

## Records and documents

| Document | Content |
|---|---|
| [`decisions_log.md`](decisions_log.md) | every decision: date, decision, reason |
| [`docs/bug_log.md`](docs/bug_log.md) | bugs found, effect on numbers (none changes a design decision) |
| [`results/freeze/freeze.md`](results/freeze/freeze.md), [`MANIFEST.sha256`](results/freeze/MANIFEST.sha256) | freeze check and manifest |
| [`configs/`](configs) | all settings: features, splits, XGBoost (`xgb_best.yaml`, `xgb_text_best.yaml`), SLM (`slm.yaml`), eval, Step 7 |
| [`configs/rules.yaml`](configs/rules.yaml) | the five rules of the rules baseline |
| [`docs/serialisation.md`](docs/serialisation.md) | the SLM prompt (structured and text) |
| `data/processed/test_ids.csv` (+ `splits.parquet`, `heldout_users.csv`, rebuilt by `run.py splits`) | frozen test IDs; checksum above |
| [`docs/gpu_setup.md`](docs/gpu_setup.md) | GPU machine setup, bundle, models |
| [`docs/step8_lab.md`](docs/step8_lab.md) | Step 8.1 lab session (latency, rationales) |
| [`docs/final_compute_lab.md`](docs/final_compute_lab.md) | 24 Nov lab session (final compute, diagnostic, rationales v2) |
| [`docs/model_record.md`](docs/model_record.md), [`docs/dataset_record.md`](docs/dataset_record.md), [`docs/compute_record.md`](docs/compute_record.md) | model revisions and licences, dataset, Mac hardware |
| [`results/gate1_report.md`](results/gate1_report.md) | Gate 1: rebuild from raw reproduces data, splits and baselines byte for byte |
| [`docs/licence_facts.md`](docs/licence_facts.md) | licences and what could be shared |

## Known limitations

- One dataset, synthetic. CERT r4.2 has 70 insider incidents; the test split has 243 positive
  sessions out of 95,570 (logoff rows), so CIs are wide.
- Test PR-AUC is about half of validation for both SLM and XGBoost (decisions log 2026-11-13).
- C2 is scored at logoff only (GPU budget); C1 hourly. The two are compared through the
  logoff-only copies of the C1 predictions.
- SLM C1 seed 1 (grid adapter) was trained before the reseeding change 1c10259, and its recall at
  5% FPR is well above seeds 2–3, so the SLM seed std is large.
- Fairness groups below 200 sessions (FPR) or 5 positives (recall) are shown as "–".
- Latency and throughput come from a single lab machine. The 17 Nov rationales were generated with
  Qwen's default repetition_penalty 1.1 (bug log #1); rationales v2 (24 Nov) fixes it. Rationales
  feed no reported metric.
- The full data rebuild from raw was not re-run at the freeze. It was verified at Gate 1
  (9–10 Nov). The freeze rebuild covers everything from the stored predictions on.
- Not reproducible from this repo alone: CERT r4.2 (licence), the model weights (Hugging Face, pinned
  revisions) and the LoRA adapters (lab PC only).
- Windows: `data/processed/test_ids.csv` must not be line-ending converted (`.gitattributes`,
  decisions log 2026-11-11/12). Apple MPS is not used (CPU only on the Mac).

## Public archive (planned: release-2026-11-24, built after the final-check PASS)

Not built yet. Decisions log 2026-11-23.

- **What it is:** `git archive` of the annotated tag `release-2026-11-24`. The tag goes on main
  after `python -m src.final_check` reports PASS on the 24 Nov final compute. It is not the bare
  freeze tag.
- **Statements the release carries:**
  - every reported number is identical to `frozen-2026-11-23` (`f94c5c5`);
  - `results/freeze/MANIFEST.sha256` is unchanged since the freeze;
  - the final-compute PASS report (`results/final_compute/final_check.md` and `.json`) is included.
- **Excluded from the archive** (`export-ignore` in [`.gitattributes`](.gitattributes); the repo
  itself keeps them):
  - `data/processed/test_ids.csv` (the frozen test IDs; tracked);
  - every `*.parquet`. Prediction, feature and processed-data files are not tracked anyway;
    the rule guards against a force-added one.
- **Their SHA-256 values:** all in `results/freeze/MANIFEST.sha256`, which is in the archive:
  233 parquet files, plus `test_ids.csv` (`5ace66c2…818e51`), `heldout_users.csv` and the other
  data/processed and data/step8 files. List them with:

  ```bash
  grep -E "  data/|\.parquet$" results/freeze/MANIFEST.sha256
  ```

### Regenerating the excluded files from your own CERT r4.2 copy

Obtain CERT r4.2 under its own licence and put it in `data/raw/r4.2/` and `data/raw/answers/`.
Then, on the Mac side (see "Run order" above for times):

```bash
python run.py all                      # processed data, features, labels, splits, test_ids.csv, leakage, text features, XGBoost v3 fit, rules, validation evals
python run.py xgb --score --splits valid calib test --allow-test
python run.py rules --score --splits valid calib test --allow-test
python run.py xgb --variant text --fit # text XGBoost (C2) seeds 1-3 from configs/xgb_text_best.yaml (no new search)
python run.py xgb --variant text --score --splits valid calib test --allow-test
python -m src.slm.text_blocks --allow-test
python -m scripts.freeze_manifest verify
```

`run.py all` must reproduce `test_ids.csv` with SHA-256 `5ace66c2…818e51`; if it doesn't, stop.
Gate 1 rebuilt the processed data, splits, test IDs, XGBoost v3 and rules predictions from raw
byte for byte (`results/gate1_report.md`). `freeze_manifest verify` then reports every
regenerated file that differs from the original.

The SLM predictions need GPU training and scoring (step 2 above; [`docs/gpu_setup.md`](docs/gpu_setup.md),
`scripts/run_queue.ps1`, `scripts/run_night2.ps1`), then `python scripts/make_final_copies.py`.
The LoRA adapters are not distributed. Retrained adapters are not expected to reproduce the
original SLM scores byte for byte (GPU kernels, fp16 scoring). Check them against the manifest
hashes only as a reference, or by rank agreement as in `src/final_check.py`.

### Contents and expected size

Measured with the current main and the export-ignore list: **344 files, 8.2 MB tar, 3.8 MB
tar.gz**. The frozen tag without exclusions is 36.2 MB, 28.0 MB of which is `test_ids.csv`.

| Part | Files |
|---|---|
| Code: `src/`, `scripts/`, `run.py`, `Makefile`, `pytest.ini`, `tests/` (synthetic fixture) | 48 + 22 + 3 + 21 |
| Configs (`configs/`) | 13 |
| Docs: `docs/`, `README.md`, `decisions_log.md` | 17 + 2 |
| Environment: `requirements-mac.txt`, `requirements-gpu.txt`, `requirements-gpu.in`, `python-version.txt` | 4 |
| Results, aggregates only (eval JSONs, tables, curves, calibration parameters, `results/freeze/`) | 178 |
| Paper figures (PDF + PNG + captions) and tables (md + tex + provenance) | 34 |
| `.gitattributes`, `.gitignore` | 2 |

At release time this grows by the final-compute report: `final_check.md` and `.json`, its
eval JSON, and the lab's `rationale_diag.json` / `rationales_v2.json`. The re-scored parquet is
excluded. Expect about 350 files and about 8.3 MB (3.9 MB gzipped).

## Repository access and licence

- **This GitHub repository stays private.** Its history contains `test_ids.csv` and commit
  410fef2 (machine hostname in metadata; bug log #3). The public archive has no history, so no
  history rewrite is needed (decisions log 2026-11-23).
- **Licence for this project's code: TODO.** MIT or Apache-2.0, to be confirmed with the
  co-author. There is no LICENSE file until then. Third-party licences:
  [`docs/licence_facts.md`](docs/licence_facts.md).
