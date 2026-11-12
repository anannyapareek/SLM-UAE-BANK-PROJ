"""The SLM training seed: --seed reaches set_seed and TrainingArguments, and varies LoRA init.

Needs torch, transformers and peft (the GPU/SLM environment); skipped where they are absent,
e.g. in the Mac baseline venv. Uses a tiny randomly initialised Llama: nothing is downloaded.
"""
import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from conftest import ROOT  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VOCAB = 128


def tiny_model():
    """Same base weights every time, without touching the global RNG."""
    cfg = transformers.LlamaConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=2, num_attention_heads=2,
                                   num_key_value_heads=2, max_position_embeddings=64)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return transformers.LlamaForCausalLM(cfg)


class CharTokenizer:
    """Enough of a tokenizer for train_lora: one id per character."""
    eos_token_id = pad_token_id = 1
    padding_side = "left"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [2 + ord(c) % (VOCAB - 2) for c in text]}

    def save_pretrained(self, path):
        pass


def lora_init(seed):
    base = tiny_model()
    transformers.set_seed(seed)
    model = peft.get_peft_model(base, peft.LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_A" in n}


def test_different_seeds_give_different_lora_init():
    a, a_again, b = lora_init(2), lora_init(2), lora_init(3)
    assert a.keys() == b.keys() and len(a) == 2 * len(TARGETS)
    assert all(torch.equal(a[n], a_again[n]) for n in a)          # same seed: same init
    assert not any(torch.equal(a[n], b[n]) for n in a)            # other seed: every A differs


def run_train_lora(tmp_path, monkeypatch, extra):
    from src.slm import common, train_lora
    if not (ROOT / "data/processed/test_ids.csv").exists():
        pytest.skip("test_ids.csv not present")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.jsonl", "valid.jsonl"):
        (data / name).write_text("".join(json.dumps({"row_id": str(i), "prompt": "x\nLabel:",
                                                     "completion": " no\nReason: r."}) + "\n"
                                         for i in range(4)))
    seen = {"set_seed": []}
    real_seed_everything = train_lora.seed_everything

    def spy_seed(seed):
        seen["set_seed"].append(seed)
        return real_seed_everything(seed)

    class FakeTrainer:
        def __init__(self, model=None, args=None, **_):
            seen["args"] = args
            self.state = SimpleNamespace(global_step=0)

        def train(self, resume_from_checkpoint=None):
            return SimpleNamespace(training_loss=0.0)

        def evaluate(self):
            return {}

    # Patch train_lora's own seams: attributes of transformers' lazy module cannot be
    # monkeypatched reliably once resolved.
    monkeypatch.setattr(train_lora, "seed_everything", spy_seed)
    monkeypatch.setattr(train_lora, "make_trainer", lambda **k: FakeTrainer(**k))
    monkeypatch.setattr(train_lora, "load_base_model", lambda *a, **k: tiny_model())
    monkeypatch.setattr(common, "load_tokenizer", lambda *a, **k: CharTokenizer())
    monkeypatch.setattr(common, "model_entry", lambda name: (name, None))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_lora", "--device", "cpu", "--out", str(out),
                                      "--data-dir", str(data), "--lr", "1e-4", "--rank", "8",
                                      "--qlora", "off", *extra])
    train_lora.main()
    return seen, json.loads((out / "train_record.json").read_text())


@pytest.mark.parametrize("extra, expected", [([], 1), (["--seed", "3"], 3)])
def test_seed_reaches_set_seed_and_training_args(tmp_path, monkeypatch, extra, expected):
    seen, record = run_train_lora(tmp_path, monkeypatch, extra)
    assert seen["set_seed"] == [expected, expected]   # at start and right before LoRA init
    assert seen["args"].seed == expected and seen["args"].data_seed == expected
    assert record["seed"] == expected
    assert record["train_data"].endswith("train.jsonl")
