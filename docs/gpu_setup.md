# GPU setup (lab)

Step-by-step for the lab GPU machine. Nothing about the machine is assumed: each step detects and reports. Run everything from the repo root, inside `tmux` (or with `nohup`), because the scripts write logs to `logs/` and keep running if the connection drops.

```bash
tmux new -s slm            # reattach later with: tmux attach -t slm
```

Each `src/slm` script logs to `logs/<script>_<time>.log` as well as the screen. `train_lora.py` writes `<out>/train.log`.

## 1. Repo and Python

```bash
git clone https://github.com/<owner>/SLM-UAE-BANK.git && cd SLM-UAE-BANK
python3 --version          # needs 3.11 or newer (current transformers and torch)
nvidia-smi                 # GPU name, driver, and "CUDA Version: X.Y" (the highest the driver supports)
df -h .                    # free disk: allow ~15 GB (bundle 0.5 GB, models ~8 GB, venv ~6 GB)
```

**No admin rights:** everything below installs into the user's home or the repo folder; no step needs `sudo`.
- **Python missing or older than 3.11:** install uv without admin (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv venv -p 3.12 .venv-gpu` and use `uv pip` in place of `pip` below.
- **No `nvidia-smi`:** the machine has no usable NVIDIA driver. Stop and report.

## 2. Virtual environment

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -U pip
```

## 3. torch: pick the build for the driver's CUDA version

Read "CUDA Version" from `nvidia-smi`. That is the newest CUDA the driver supports. Pick the **highest row at or below it**:

| nvidia-smi CUDA Version | Command | torch (cp312, checked 2026-11-09) |
|---|---|---|
| 13.2 or newer | `pip install torch --index-url https://download.pytorch.org/whl/cu132` | 2.15.0 |
| 13.0 – 13.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu130` | 2.14.1 |
| 12.9 | `pip install torch --index-url https://download.pytorch.org/whl/cu129` | 2.13.0 |
| 12.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` | 2.11.0 |
| 12.6 – 12.7 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` | 2.14.1 |
| 12.4 – 12.5 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` | 2.6.0 (old: pin `transformers<5` if step 4 complains) |
| 11.8 – 12.3 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` | 2.7.1 (old: pin `transformers<5` if step 4 complains) |

Check it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: the version from the table, its CUDA build, `True`, and the GPU name. If the third value is `False`, the build does not match the driver: try the next row down.

## 4. The rest of the packages, then freeze

```bash
pip install -r requirements-gpu.in
pip freeze > requirements-gpu.txt          # the exact set used; bring this file back
```

`bitsandbytes` is needed only for QLoRA on small GPUs. If it fails to install, carry on: `env_check` reports it, and training then uses plain LoRA.

## 5. Environment check

```bash
python -m src.slm.env_check --device cuda
```

This writes `results/gpu_env.json`. Expected output (values vary):
- torch, transformers, peft, accelerate, trl and bitsandbytes versions;
- `torch_cuda_build`, `cuda_available: True`, `cuda_probe: ok`, `cudnn`. The probe runs a small matmul in a child process, so a torch build that does not match the driver is caught here, not mid-run;
- `gpus`: name, `vram_total_gb`, `vram_free_gb`, compute capability;
- `bf16_supported`: True on Ampere (A100, A10, RTX 30xx) or newer, False on older cards, which then train in fp32;
- `disk.free_gb`, `ram_gb`;
- `training_mode`: `lora (bf16)`, or `qlora (4-bit)` if VRAM is below 12 GB.

It exits with code 1 if `--device cuda` is not usable. Every GPU script runs the same probe at start-up and stops with the reason if it fails.

## 6. Data bundle from Google Drive, then verify it first

Download `slm_bundle.tar` (445 MB, uncompressed) and `slm_bundle.tar.sha256` from the shared Drive folder.

- **Browser:** download both files, then copy them to the repo root (`scp` or the lab file share).
- **gdown** (installed in step 4); set the file's sharing to "Anyone with the link":
  ```bash
  gdown --fuzzy "<Drive share link of slm_bundle.tar>" -O slm_bundle.tar
  gdown --fuzzy "<Drive share link of slm_bundle.tar.sha256>" -O slm_bundle.tar.sha256
  ```

Verify **before anything else**:

```bash
sha256sum -c slm_bundle.tar.sha256         # expected: slm_bundle.tar: OK
mkdir -p bundle && tar -xf slm_bundle.tar -C bundle
python -m src.slm.verify_bundle --root bundle --install
```

`verify_bundle` checks every file against `bundle/MANIFEST.sha256` and checks the bundled `test_ids.csv` against the SHA-256 in `decisions_log.md`. It then copies the data files into `data/processed/` and calls `verify_test_checksum()`. Expected:
- one `OK` line per file (14);
- `bundled test_ids.csv matches the logged SHA-256 5ace66c2…818e51`;
- `installed …` for the 4 data files;
- `verify_test_checksum(): 5ace66c2…818e51`;
- `BUNDLE OK`.

Any `MISMATCH`/`MISSING` line, or a non-zero exit, means stop: download the bundle again.

The bundle's configs and docs are a record of the Mac state. The code always uses the repo's own (git) copies, and differences are only reported.

## 7. Models

Straight from Hugging Face, at the revisions pinned in `configs/slm.yaml`. The models are not gated, so `HF_TOKEN` is optional:

```bash
python scripts/download_models.py --out models           # primary, second, fallback: ~7.5 GB
```

Or from a copy fetched from Drive (folder layout `<dir>/<org>__<name>/`):

```bash
python scripts/download_models.py --from-dir /path/to/drive/models --out models
```

Both write `models/download_record.json` (revision, date, SHA-256 of every file). Copy the dates into `docs/model_record.md`. Every script below takes `--models-dir models`.

## 8. Tokenizer and budget check (CPU, quick)

```bash
python -m src.slm.check_tokens --models-dir models
```

Expected: one id each for `yes`, `no`, ` yes` and ` no` per model; 0 boundary mismatches; PASS for the 512-token budget for all three models. It writes `results/token_check.json`.

## 9. Training files

```bash
python -m src.slm.make_train            # data/slm/train.jsonl and valid.jsonl
```

Expected: `train_rows` ≈ 31,000 (2,823 positives + 10 negatives per positive) and `valid_rows` 2,000.

## 10. Smoke test

```bash
python -m src.slm.smoke_test --device cuda --model primary --models-dir models
```

Expected: 10 lines `row … p_yes=… margin=…`, then `SMOKE TEST PASSED`; it writes `results/smoke_test_Qwen2.5-1.5B-Instruct.json`. "generation parses … for k of 10" is informational: the base model is untrained.

## 11. Timing

```bash
python -m src.slm.time_scoring --device cuda --model primary --models-dir models
```

It times 1,000 validation rows at batch sizes 8, 16, 32 and 64, with and without the 256-token text block, and writes `results/gpu_timing.json`. Expected: one line per variant and batch size (seconds, rows/s, peak memory), then two summary lines, hourly and logoff-only, each with the projected GPU-hours for valid + calib + test × 3 seeds × 2 comparisons. A batch size that runs out of memory is recorded as `oom` and skipped.

## 12. Short LoRA run (100 steps)

```bash
python -m src.slm.train_lora --device cuda --model primary --models-dir models \
    --lr 2e-4 --rank 16 --out runs/short_qwen15_r16 --max-steps 100 --save-steps 50 --eval-steps 50
```

Expected: the LoRA settings line (rank 16, alpha 32, dropout 0.05, the 7 projection modules, trainable parameters, precision, QLoRA), loss lines every 10 steps, evals at steps 50 and 100, and `saved adapter to runs/short_qwen15_r16/adapter`.

If the run is interrupted, rerun the same command: it resumes from the latest checkpoint and logs "resume from …/checkpoint-50".

## Bring back

- `requirements-gpu.txt`
- `results/gpu_env.json`, `results/token_check.json`, `results/smoke_test_*.json`, `results/gpu_timing.json`
- `models/download_record.json`
- `runs/*/train_record.json`
- `logs/`
