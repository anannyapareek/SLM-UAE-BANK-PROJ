"""LoRA fine-tuning (runbook 5.2-5.3), with checkpoints and automatic resume.

Fixed by configs/slm.yaml: alpha = 2 x rank, dropout 0.05, 1 epoch, max length 512, LoRA on
the attention and MLP projections, seed 1. Chosen per run: --lr and --rank (the grid is
lr {1e-4, 2e-4} x rank {8, 16}). The loss is on the completion only ("<yes|no>\\nReason: ...
<eos>"); prompt tokens are masked.

Precision: bf16 on a CUDA GPU that supports it, else fp32 (reported). QLoRA: 4-bit weights
when --qlora on, or --qlora auto (default) and total VRAM < lora.qlora_below_gb and
bitsandbytes imports. MPS and CPU train in fp32 (for testing the pipeline).

Resume: checkpoints go to <out>/checkpoint-<step> every --save-steps; rerunning the same
command resumes from the latest one. The adapter is saved to <out>/adapter.

Run: python -m src.slm.train_lora --device cuda --model primary --lr 2e-4 --rank 16 \\
         --data-dir data/slm --out runs/qwen15_r16_lr2e-4 [--max-steps 100]
"""
import json
import math
import pathlib
import time

from src.slm import common
from src.splits import verify_test_checksum

WARMUP = 0.03          # fraction of training steps with linear warm-up


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def encode(rows, tok, max_length, log, name):
    examples, truncated = [], 0
    for r in rows:
        prompt = tok(r["prompt"], add_special_tokens=True)["input_ids"]
        completion = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids, labels = prompt + completion, [-100] * len(prompt) + completion
        if len(ids) > max_length:
            truncated += 1
            ids, labels = ids[-max_length:], labels[-max_length:]   # keep the completion
        examples.append({"input_ids": ids, "labels": labels})
    log.info("%s: %d examples, %d truncated to %d tokens, longest %d", name, len(examples),
             truncated, max_length, max(len(e["input_ids"]) for e in examples))
    return examples


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, value: seq + [value] * (width - len(seq))
        return {"input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}


def use_qlora(mode, device_name, log):
    import torch
    if mode == "off" or device_name != "cuda":
        return False
    try:
        import bitsandbytes  # noqa: F401
        bnb = True
    except Exception as e:
        bnb = False
        log.warning("bitsandbytes not importable: %s", e)
    if mode == "on":
        if not bnb:
            raise SystemExit("--qlora on but bitsandbytes is not importable")
        return True
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    limit = common.config()["lora"]["qlora_below_gb"]
    log.info("VRAM %.1f GB; QLoRA threshold %s GB", vram, limit)
    if vram < limit and not bnb:
        log.warning("VRAM is small but bitsandbytes is missing: training without QLoRA")
    return vram < limit and bnb


def training_kwargs(**kwargs):
    """TrainingArguments keywords that work across transformers 4.x and 5.x."""
    import inspect
    from transformers import TrainingArguments
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = WARMUP
    else:                                   # transformers 5: a float < 1 is a ratio
        kwargs["warmup_steps"] = WARMUP
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    return kwargs


def latest_checkpoint(out):
    points = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(points[-1]) if points else None


def main():
    cfg = common.config()
    lora = cfg["lora"]
    p = common.parser(__doc__)
    p.add_argument("--model", default="primary")
    p.add_argument("--models-dir")
    p.add_argument("--data-dir", default=str(common.ROOT / "data/slm"))
    p.add_argument("--out", required=True, help="output folder (checkpoints, adapter, record)")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--qlora", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--max-train-rows", type=int, help="cap the training rows (tests)")
    p.add_argument("--max-valid-rows", type=int)
    p.add_argument("--no-resume", action="store_true", help="start fresh even if checkpoints exist")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = common.setup_logging("train_lora", args.log or out / "train.log")
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    common.device(args.device, log)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(lora["seed"])
    tok = common.load_tokenizer(args.model, args.models_dir)
    tok.padding_side = "right"
    data = pathlib.Path(args.data_dir)
    train = encode(read_jsonl(data / "train.jsonl", args.max_train_rows), tok,
                   lora["max_length"], log, "train")
    valid = encode(read_jsonl(data / "valid.jsonl", args.max_valid_rows), tok,
                   lora["max_length"], log, "valid")

    qlora = use_qlora(args.qlora, args.device, log)
    bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    if qlora:
        model = common.load_model(args.model, args.device, args.models_dir, quantize_4bit=True)
        model = prepare_model_for_kbit_training(model)
    else:
        from transformers import AutoModelForCausalLM
        source, revision = common.model_source(args.model, args.models_dir)
        model = AutoModelForCausalLM.from_pretrained(
            source, revision=revision, dtype=torch.bfloat16 if bf16 else torch.float32)
        model.to(args.device)
    if args.device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [m for m in lora["target_modules"] if m in present]
    missing = sorted(set(lora["target_modules"]) - set(targets))
    if missing:
        log.warning("target modules not in this model: %s", missing)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=lora["alpha_per_rank"] * args.rank,
        lora_dropout=lora["dropout"], target_modules=targets, bias="none",
        task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    log.info("LoRA rank %d alpha %d dropout %s on %s; trainable %d of %d parameters; "
             "precision %s; QLoRA %s", args.rank, lora["alpha_per_rank"] * args.rank,
             lora["dropout"], targets, trainable, total, "bf16" if bf16 else "fp32", qlora)

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    targs = TrainingArguments(**training_kwargs(
        output_dir=str(out), num_train_epochs=lora["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", weight_decay=0.0,
        bf16=bf16, logging_steps=max(1, min(10, args.save_steps)),
        eval_steps=args.eval_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=2, seed=lora["seed"],
        data_seed=lora["seed"], report_to=[], dataloader_num_workers=0,
        remove_unused_columns=False, use_cpu=args.device == "cpu"))
    resume = None if args.no_resume else latest_checkpoint(out)
    log.info("%d train examples, %d steps per epoch, max_steps %d; resume from %s",
             len(train), steps_per_epoch, args.max_steps, resume)

    from transformers import TrainerCallback

    class ToLog(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_):
            log.info("step %d: %s", state.global_step, logs)

    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=valid,
                      data_collator=Collator(tok.pad_token_id), callbacks=[ToLog()])
    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    record = {"model": common.model_entry(args.model), "lr": args.lr, "rank": args.rank,
              "alpha": lora["alpha_per_rank"] * args.rank, "dropout": lora["dropout"],
              "target_modules": targets, "max_length": lora["max_length"],
              "precision": "bf16" if bf16 else "fp32", "qlora": qlora,
              "batch_size": args.batch_size, "grad_accum": args.grad_accum,
              "max_steps": args.max_steps, "global_step": trainer.state.global_step,
              "resumed_from": resume, "train_examples": len(train),
              "train_loss": result.training_loss, "eval": metrics,
              "seconds_this_run": round(time.time() - start, 1), "device": args.device,
              "test_ids_sha256": verify_test_checksum()}
    common.write_json(out / "train_record.json", record)
    log.info("saved adapter to %s; record %s", out / "adapter", record)


if __name__ == "__main__":
    main()
