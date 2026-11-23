# Licence facts (Step 8.4)

This page reports facts only. Whether to share anything is a project decision that is not taken
here. Checked 2026-11-23.

## CERT Insider Threat r4.2 (data/raw/r4.2/license.txt; identical in data/raw/answers/)

The dataset is "Test Data Files" produced by ExactData's Dynamic Data Generator under SEI agreement
#12-00121-EXP-000 (Carnegie Mellon University and ExactData, LLC), © 2011 ExactData, LLC, all
rights reserved. Its End User Agreement says:

- **Grant of Rights Restrictions (a):** the end user is "prohibited from publishing, distributing
  via the Internet or other public computer based information system". The clause also covers
  creating derivative works (including translations), and transferring, selling, leasing,
  licensing or otherwise making available the Licensed Products, or a copy or portion of them,
  to any unauthorized party.
- **(b):** everyone with authorized access must comply with the agreement.
- **(c):** derivative works based on use of the data, such as reports and research papers, are
  permitted. They may include the data only to the minimal extent needed to describe the
  performance of the systems tested, and should refer to the general qualities of the data rather
  than reproduce it. Any reproduced portion must carry ExactData's copyright notice (exact wording in the
  licence file).
- **Back Up Rights:** copies only for back-up or archival purposes, with all notices kept.

The full text is in `data/raw/r4.2/license.txt`, which is not in git.

## Qwen2.5-1.5B-Instruct (pinned revision 989aa79)

- **Licence:** Apache License 2.0, per the `LICENSE` file and the model card (`license: apache-2.0`)
  in the Hugging Face repo at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The cached copy
  on the Mac holds only the weights, config and tokenizer; the licence file was read from the
  repo at that revision.
- **Apache 2.0 §4 (redistribution of the work or derivative works):**
  - give recipients a copy of the licence;
  - mark modified files as changed;
  - keep the copyright, patent, trademark and attribution notices;
  - pass on any NOTICE file.
- **§3:** patent grant. **§6:** no trademark rights.
- The other two models in `docs/model_record.md` (SmolLM2-1.7B-Instruct, Qwen2.5-0.5B-Instruct)
  are also Apache 2.0. Neither is used in a reported result.

## Main Python packages (versions as pinned)

| Package | Version | Licence | Machine |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib) | Mac |
| pandas | 3.1.0 | BSD-3-Clause | Mac |
| pyarrow | 26.0.1 | Apache-2.0 | Mac |
| polars | 2.1.0 | MIT | Mac |
| duckdb | 2.0.0 | MIT | Mac, GPU |
| scikit-learn | 1.9.1 | BSD-3-Clause | Mac |
| scipy | 1.18.1 | BSD | Mac |
| xgboost | 3.4.1 | Apache-2.0 | Mac |
| optuna | 5.0.0 | MIT | Mac |
| matplotlib | 3.11.2 | matplotlib licence (PSF-based) | Mac |
| PyYAML | 6.0.3 | MIT | Mac |
| pytest | 9.2.0 | MIT | Mac |
| torch | 2.11.0+cu128 | BSD-3-Clause | GPU |
| transformers | 5.22.2 | Apache-2.0 | GPU |
| peft | 0.21.2 | Apache-2.0 | GPU |
| accelerate | 1.15.0 | Apache-2.0 | GPU |
| trl | 1.17.1 | Apache-2.0 | GPU |
| bitsandbytes | 0.50.2 | MIT | GPU (installed; QLoRA not used) |
| datasets | 5.2.0 | Apache-2.0 | GPU |
| huggingface_hub | 2.4.0 | Apache-2.0 | GPU |

Sources:
- Mac packages: the installed package metadata (`.venv`).
- GPU packages: the PyPI metadata for the pinned versions.

All are permissive (BSD, MIT, Apache, PSF-style); none is copyleft. The CUDA runtime wheels
that come with torch are under NVIDIA's own licence and are not redistributed here.

## What could be shared, and which clause bears on it

There is no licence file for this project's own code yet (TODO: MIT or Apache-2.0, see Release decisions).

| Item | What it contains | Clause that allows or blocks it |
|---|---|---|
| Code (`src/`, `scripts/`, `tests/`, `run.py`, `Makefile`) | Project code. The test prompt fixture `tests/fixtures/structured_prompts_03b1641.json` is synthetic (none of its rows appears in features.parquet); no CERT user or PC IDs in any tracked file (checked at the tag) | Not CERT data, so the CERT agreement does not restrict it. Dependencies are permissive. Needs a project licence |
| Configs (`configs/`, incl. `rules.yaml`) | Settings, rule definitions, feature lists | As code. The rules name behaviour categories, not CERT records |
| Split IDs (`data/processed/test_ids.csv`, tracked) | 848,137 `row_id`s = md5(session_id \| score time); no raw values | Derived from CERT sessions. Whether opaque hashes count as a "portion" or "derivative work" under CERT (a), or fall under (c), is open. `heldout_users.csv` and `splits.parquet` (not tracked) hold CERT user IDs, so (a) applies to them |
| Predictions (`results/*/preds_*.parquet`, not tracked) | `row_id`, `y_true` (from the CERT answer key), score, margin | Derived from CERT data and its answer key: CERT (a) (derivative works, portions). Aggregate metrics in a paper are covered by (c) |
| Metrics, tables, figures (`results/**/*.json\|md`, `paper/`) | Aggregates over sessions; no records | CERT (c): reports and research papers that describe system performance |
| Rationale texts (`results/step8/rationales*.parquet`) | Model output for 343 sampled sessions per model; the C2 prompts contain CERT event text. The C2 reason texts contain no URLs, e-mail addresses or CERT user IDs (pattern check) | Generated from CERT-derived prompts: CERT (a) vs (c) is open. Short quoted examples in the paper fall under (c), with the ExactData notice if any data is reproduced |
| LoRA adapters (`runs/*/adapter`, lab PC only, 46 MB each) | Weights fine-tuned from Qwen2.5-1.5B on CERT-derived prompts | Qwen side: Apache 2.0 allows it, with the licence, the notices and a change notice (§4). CERT side: whether trained weights are a "derivative work" under (a) is open |
| Processed features (`data/processed/*.parquet`, `slm_text_blocks.parquet`) | Per-row features, event text, user and PC IDs | CERT (a): portions and derivative works of the Licensed Products; not covered by (c) |
| Raw CERT r4.2 | The dataset | CERT (a); back-up copies only |

## Release decisions (decisions log 2026-11-23)

- The public archive is a `git archive` of `release-2026-11-24`. It excludes
  `data/processed/test_ids.csv` and every `*.parquet` (`export-ignore`); their SHA-256 values are
  in `results/freeze/MANIFEST.sha256`, and README gives the commands to regenerate them from your
  own CERT r4.2 copy.
- The GitHub repository stays private. Its history holds `test_ids.csv` and commit 410fef2 (the
  machine hostname in results metadata, fixed forward in 5deb353 and 4a78b28; bug log #3). The
  archive has no history, so no history rewrite is needed.
- Licence for the project's own code: TODO (MIT or Apache-2.0, to be confirmed with the
  co-author).
