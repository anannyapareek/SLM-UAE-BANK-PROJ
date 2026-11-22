# Final compute lab session, 24 Nov (GPU lab PC, Windows 11, RTX 4500 Ada)

Draft: the setup and final-compute scoring sections are added in S8-6, before the sections below.
PowerShell, from the repo root, after `git pull` to the frozen tag. Setup as `docs/step8_lab.md`
section 1, plus Hugging Face offline mode for every run (decisions log 2026-11-23):

```powershell
$env:SLM_SCORE_DTYPE = "fp16"; $env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
```

Every script's log records both offline values; `rationale_diag.py` and `rationales.py` refuse to
run on CUDA without them and write them into their JSON outputs.

## Rationale-path diagnostic (after the final-compute scoring; a few minutes)

```powershell
python -m src.slm.rationale_diag --device cuda --model primary --models-dir models --c1-adapter runs\grid_qwen15_r8_lr1e-4\adapter --c2-adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter 2>&1 | Tee-Object logs\rationale_diag_lab.log
```

One line per row (11 mismatch rows, 7 controls) → `results\step8\rationale_diag.json`. Expected
(decisions log 2026-11-23): step (f-i), the 17 Nov arguments, gives the same label in all 20 repeats
and equals 17 Nov (the 10 C2 rows keep their flipped label); step (f-ii), the corrected arguments
(greedy, repetition_penalty 1.0), agrees with the scoring pass (a), apart from fp16 noise at
|p_yes − 0.5| < 0.01. Labels varying across the 20 repeats would mean sampling: note it and go on.

## Rationale regeneration, v2 (after the diagnostic; about 6 minutes)

Same input sample, same adapters, batch 1; greedy with repetition_penalty=1.0 passed explicitly
(decisions log 2026-11-23). Writes `results\step8\rationales_v2.parquet` and `rationales_v2.json`
(merged generation config, arguments, adapters, offline values); it refuses to write to the 17 Nov
`rationales.parquet`, which is kept for the bug log. Resume-safe: Ctrl+C and rerun the same command.

```powershell
python -m src.slm.rationales --device cuda --model primary --models-dir models --c1-adapter runs\grid_qwen15_r8_lr1e-4\adapter --c2-adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter 2>&1 | Tee-Object logs\rationales_v2_lab.log
```

Expected: C1 343 rows, then C2 343 rows; the log prints each model's merged generation config with
`'repetition_penalty': 1.0` and `'do_sample': False` (it stops otherwise).

## Collect

```powershell
git rev-parse HEAD > results\step8\lab_commit.txt
pip freeze > results\step8\requirements-gpu.lab.txt
Compress-Archive -Force -Path results\step8, logs\xgb_latency_lab.log, logs\rationale_diag_lab.log, logs\rationales_v2_lab.log -DestinationPath step8_final_results.zip
(Get-FileHash step8_final_results.zip -Algorithm SHA256).Hash.ToLower()
```

Bring back the zip and its hash (record the hash this time). `results\step8` includes
`rationale_diag.json`, `rationales_v2.parquet` and `rationales_v2.json`. On the Mac,
`python -m src.slm.step8_report` then checks v2: 0 label/score disagreements outside fp16 noise,
or it stops.
