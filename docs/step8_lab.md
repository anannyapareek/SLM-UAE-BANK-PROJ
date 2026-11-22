# Step 8.1 lab session (GPU lab PC, Windows 11, RTX 4500 Ada)

Latency (SLM and XGBoost) and rationales. Inputs were built on the Mac by
`python -m src.slm.step8_inputs build` (decisions log 2026-11-15); the lab needs no feature data.
Nothing here computes metrics. PowerShell, from the repo root. Keep the GPU otherwise idle while
`latency` runs.

Bundle: `bundle\step8_bundle.tar`, 7,587,840 bytes (7.2 MB), reproducible (rebuilding gives the same hash):

```
SHA-256 3700222cce4ca4e8ab491f8ad66d2f429fc1a15eedca7d53dd85323bf37e674d
```

## 1. Setup

```powershell
cd D:\slm_project\SLM-UAE-BANK-PROJ
git pull
.\.venv\Scripts\Activate.ps1
$env:PIP_CACHE_DIR = "D:\slm_cache\pip"; $env:HF_HOME = "D:\slm_cache\hf"
$env:TEMP = "D:\slm_cache\tmp"; $env:TMP = $env:TEMP
$env:PYTHONUNBUFFERED = "1"; $env:SLM_SCORE_DTYPE = "fp16"
pip install xgboost==3.4.1
python -c "import torch, xgboost, psutil; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), xgboost.__version__)"
Test-Path runs\grid_qwen15_r8_lr1e-4\adapter\adapter_config.json
Test-Path runs\c2_qwen15_r8_lr1e-4_s1\adapter\adapter_config.json
```

Expected: `True` for CUDA, the GPU name, xgboost 3.4.1, and `True` twice (C1 seed-1 adapter =
the grid adapter; C2 seed-1 adapter). Copy `step8_bundle.tar` into `bundle\` first.

## 2. Checksum and inputs

```powershell
(Get-FileHash bundle\step8_bundle.tar -Algorithm SHA256).Hash.ToLower()
tar -xf bundle\step8_bundle.tar
python -m src.slm.step8_inputs verify
```

Expected: the hash above; then five `OK` file lines, three `OK … 0 outside the frozen test set`
lines, `test_ids.csv SHA-256 5ace66c2…818e51` and `STEP 8 INPUTS OK`. Anything else: stop.

## 3. SLM latency (about 10 minutes)

```powershell
python -m src.slm.latency --device cuda --model primary --models-dir models --c1-adapter runs\grid_qwen15_r8_lr1e-4\adapter --c2-adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter
```

One log line per configuration (a b1, c b1, a b16, b b1, b b16) → `results\step8\latency_gpu.json`
(written after each configuration with `"partial": true`, final without it).

## 4. XGBoost latency (under a minute)

```powershell
python -m src.slm.xgb_latency 2>&1 | Tee-Object logs\xgb_latency_lab.log
```

Expected first line: `check: max |inplace_predict - stored test score| ≤ 1e-6` (expect
~1e-10; 17 Nov: 5.82e-11; it stops above 1e-6) → `results\step8\latency_xgb_lab.json`.

## 5. Rationales (about 15–30 minutes; stop and restart freely)

```powershell
python -m src.slm.rationales --device cuda --model primary --models-dir models --c1-adapter runs\grid_qwen15_r8_lr1e-4\adapter --c2-adapter runs\c2_qwen15_r8_lr1e-4_s1\adapter
```

C1 (343 rows) first, then C2 (343 rows). Ctrl+C at any time and rerun the same command: rows
already in `results\step8\rationales.parquet` are skipped (it is rewritten every 10 rows). For
C1 only: add `--models c1`.

## 6. Collect

```powershell
Compress-Archive -Force -Path results\step8, logs\latency_*.log, logs\rationales_*.log, logs\xgb_latency_lab.log -DestinationPath step8_results.zip
(Get-FileHash step8_results.zip -Algorithm SHA256).Hash.ToLower()
```

Bring back `step8_results.zip` and its hash.

## Carried to the 24 Nov lab session (decisions log 2026-11-22)

Environment for every 24 Nov run (final compute and the diagnostic), in addition to section 1
(decisions log 2026-11-23: Hugging Face offline mode, to support "fully local inference"):

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
```

Every script's log records both values (`src/slm/common.setup_logging`); `rationale_diag.py` also
writes them into its JSON and refuses to run on CUDA without them.

The 17 Nov `step8_results.zip` hash was not recorded and the lab logs were not copied; the outputs
were verified instead by row-ID match with the bundle and the protocol checks in
`src/slm/step8_report.py`. On 24 Nov also collect:

- `logs\xgb_latency_lab.log` (from section 4 above)
- `requirements-gpu.txt` as installed: `pip freeze > requirements-gpu.lab.txt`
- the regenerated rationale sample `results\step8\rationales_v2.parquet` + `.json` and its log
- the rationale-path diagnostic `results\step8\rationale_diag.json` (`src/slm/rationale_diag.py`;
  run after the final-compute scoring; commands go in `docs/final_compute_lab.md`)
