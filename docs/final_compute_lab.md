# Final compute lab session, 24 Nov (GPU lab PC, Windows 11, RTX 4500 Ada)

All code runs from the frozen tag `frozen-2026-11-23` (decisions log 2026-11-23: freeze; after it
only the final compute, the rationale diagnostic and the v2 regeneration run). PowerShell, from the
repo root. Order: setup → final compute, about 30 min (27.7 min on 13 Nov) → diagnostic → v2 regeneration → collect.
Nothing here changes a reported number.

## 1. Setup and checks

```powershell
cd D:\slm_project\SLM-UAE-BANK-PROJ
git fetch --tags
git checkout frozen-2026-11-23
git describe --tags --exact-match
.\.venv\Scripts\Activate.ps1
$env:PIP_CACHE_DIR = "D:\slm_cache\pip"; $env:HF_HOME = "D:\slm_cache\hf"
$env:TEMP = "D:\slm_cache\tmp"; $env:TMP = $env:TEMP
$env:SLM_SCORE_DTYPE = "fp16"; $env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m scripts.freeze_manifest verify
Test-Path runs\c2_qwen15_r8_lr1e-4_s1\adapter\adapter_config.json
Test-Path data\processed\slm_text_blocks.parquet
```

Expected: `git describe` prints `frozen-2026-11-23`;
CUDA `True` and the GPU name; the manifest line `0 mismatched` (most files are not on the lab PC
and count as "not present here"; `data/processed/test_ids.csv`, the text blocks and any results
files that are present must match); `True` twice. Any mismatch: stop. Offline mode (decisions log
2026-11-23) is recorded in every script's log; `rationale_diag.py` and `rationales.py` refuse to
run on CUDA without it and write it into their JSON outputs.

## 2. Final compute: re-score SLM text, C2 seed 1, test logoff rows, about 30 min (27.7 min on 13 Nov)

Same command as the 13 Nov scoring run (`scripts/run_night2.ps1`, score_c2: batch 8, logoff rows,
text prompts, fp16, merged seed-1 adapter), into a new folder so the frozen predictions are never
touched (decisions log 2026-11-23):

```powershell
python -m src.slm.score --device cuda --model primary --models-dir models --adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter --split test --only-final --allow-test --text --batch-size 8 --out results\final_compute\c2_s1_preds_test_final.parquet --log logs\final_compute_c2_s1.score.log
```

Expected in the log: the test SHA-256 `5ace66c2…818e51`, `95570 test rows`, `text prompts`,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, then `wrote results\final_compute\c2_s1_preds_test_final.parquet`.
The PASS check runs on the Mac (`python -m src.final_check`): per-row Spearman ≥ 0.999 against
the frozen `preds_test_final.parquet` and every Table 2 SLM-text seed-1 metric within ±0.002.

## 3. Rationale-path diagnostic (after the final-compute scoring; a few minutes)

```powershell
python -m src.slm.rationale_diag --device cuda --model primary --models-dir models --c1-adapter runs\grid_qwen15_r8_lr1e-4\adapter --c2-adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter 2>&1 | Tee-Object logs\rationale_diag_lab.log
```

One line per row (11 mismatch rows, 7 controls) → `results\step8\rationale_diag.json`. Expected
(decisions log 2026-11-23): step (f-i), the 17 Nov arguments, gives the same label in all 20 repeats
and equals 17 Nov (the 10 C2 rows keep their flipped label); step (f-ii), the corrected arguments
(greedy, repetition_penalty 1.0), agrees with the scoring pass (a), apart from fp16 noise at
|p_yes − 0.5| < 0.01. Labels varying across the 20 repeats would mean sampling: note it and go on.

## 4. Rationale regeneration, v2 (after the diagnostic; about 6 minutes)

Same input sample, same adapters, batch 1; greedy with repetition_penalty=1.0 passed explicitly
(decisions log 2026-11-23). Writes `results\step8\rationales_v2.parquet` and `rationales_v2.json`
(merged generation config, arguments, adapters, offline values); it refuses to write to the 17 Nov
`rationales.parquet`, which is kept for the bug log. Resume-safe: Ctrl+C and rerun the same command.

```powershell
python -m src.slm.rationales --device cuda --model primary --models-dir models --c1-adapter runs\grid_qwen15_r8_lr1e-4\adapter --c2-adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter 2>&1 | Tee-Object logs\rationales_v2_lab.log
```

Expected: C1 343 rows, then C2 343 rows; the log prints each model's merged generation config with
`'repetition_penalty': 1.0` and `'do_sample': False` (it stops otherwise).

## 5. Collect

```powershell
git rev-parse HEAD > results\step8\lab_commit.txt
git rev-parse HEAD > results\final_compute\lab_commit.txt
pip freeze > results\step8\requirements-gpu.lab.txt
Compress-Archive -Force -Path results\step8, results\final_compute, logs\final_compute_c2_s1.score.log, logs\xgb_latency_lab.log, logs\rationale_diag_lab.log, logs\rationales_v2_lab.log -DestinationPath step8_final_results.zip
(Get-FileHash step8_final_results.zip -Algorithm SHA256).Hash.ToLower()
```

Bring back the zip and its hash (record the hash this time). `results\step8` includes
`rationale_diag.json`, `rationales_v2.parquet` and `rationales_v2.json`. On the Mac,
`python -m src.slm.step8_report` then checks v2: 0 label/score disagreements outside fp16 noise,
or it stops; and `python -m src.final_check` gives the final-compute PASS/FAIL
(`results/final_compute/final_check.md`).
