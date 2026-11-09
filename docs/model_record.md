# Model record

The models are pinned by revision in `configs/slm.yaml`. Every script loads exactly that revision. Details were checked against the Hugging Face API on 2026-11-09.

| Role | Name | Revision (commit hash) | Parameters | Licence | Source | Gated | Quantisation | bf16 weights | Download date |
|---|---|---|---|---|---|---|---|---|---|
| Primary | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 1,543,714,304 | Apache 2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE)) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | no | none (bf16) | 3.09 GB | tokenizer 2026-11-09 (Mac); weights: see below |
| Second | HuggingFaceTB/SmolLM2-1.7B-Instruct | `31b70e2e869a7173562077fd711b654946d38674` | 1,711,376,384 | Apache 2.0 (model card; [licence text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct | no | none (bf16) | 3.42 GB | tokenizer 2026-11-09 (Mac); weights: see below |
| Fallback (Gate 2 "smaller model") | Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 494,032,768 | Apache 2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE)) | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct | no | none (bf16) | 0.99 GB | tokenizer 2026-11-09 (Mac); weights: see below |

- **Parameter counts:** the bf16 safetensors totals reported by Hugging Face. The weight size is the parameter count × 2 bytes.
- **Weights download date:** `scripts/download_models.py` writes `models/download_record.json` with the revision, download date and file hashes. Copy the dates here after the lab download.
- **QLoRA:** quantisation stays "none". QLoRA (4-bit) is a fallback in `train_lora.py`, used only when VRAM is below `lora.qlora_below_gb`. If it is used, record it here.
