"""Step 8 rationale-path diagnostic (decisions log 2026-11-22): why 11 rationale rows (C1 1, C2 10)
generated a yes/no label on the other side of 0.5 from their score.

For each of the 11 rows (DIAG_ROWS; the mismatch set in results/step8/rationales.parquet, checked by
src.slm.step8_report), with the model's seed-1 adapter merged as for scoring:
  a  scoring pass exactly as src.slm.score.score_texts (batch 1): fp32 " yes"/" no" logits, p_yes
  b  first generation step at the batch size used on 17 Nov: src.slm.rationales generated one prompt
     per call (batch 1, no padding), so b = model.generate(max_new_tokens=1) on the prompt alone,
     same arguments as rationales.generate: generated token id and string, top-5 tokens with
     logits, the " yes"/" no" logits
  c  the same first step at batch 1 (no padding) as a plain forward pass outside generate()
     (model(**enc), full-vocabulary head in the model dtype), same fields as b
  e  the same first step through generate() with the model's 11-row subset as one left-padded batch
     (C2: 10 rows; C1: 1 row, so no padding), same fields as b: the configuration in which a
     padding/position bug could show
  d  for b, c and e: attention mask and position ids (or cache position) seen by the model's first
     forward call, at the last prompt position
Because 17 Nov generation was batch 1, b and c differ only in generate() vs a plain forward; e adds
padding. Reading (to log afterwards): generated id not the " yes"/" no" id -> label from another
token (cause 1); c agrees with a but b or e does not -> generate/padding path (cause 3); b and c
both disagree with a -> prefill precision (cause 2).

Reads only the bundle (data/step8/rationale_sample.parquet, checked against MANIFEST.sha256) and
the frozen test_ids.csv; writes only --out. No log file, no metrics, no regeneration of rationales.

Run (lab, after the final-compute scoring): set SLM_SCORE_DTYPE=fp16, then
  python -m src.slm.rationale_diag --device cuda --model primary --models-dir models \\
      --c1-adapter runs\\grid_qwen15_r8_lr1e-4\\adapter --c2-adapter runs\\c2_qwen15_r8_lr1e-4_s1\\adapter
Writes results/step8/rationale_diag.json.
"""
import argparse
import os
import pathlib
import platform

import numpy as np
import pandas as pd

from src.slm import common
from src.slm.rationales import load_model
from src.slm.score import score_texts
from src.slm.step8_inputs import sha256
from src.splits import verify_test_checksum

INPUTS = common.ROOT / "data/step8"
OUT = common.ROOT / "results/step8/rationale_diag.json"
GEN_BATCH_17NOV = 1          # src.slm.rationales.generate: one prompt per call
TOP = 5
# (model, session_id, row_id): rows of results/step8/rationales.parquet whose generated label is on
# the other side of 0.5 from p_yes (src.slm.step8_report checks this list against the file).
DIAG_ROWS = [
    ("c1", 64092, "270f67d5521946d23826de19e83e2b18"),
    ("c2", 64111, "77b75a5b4ca8f129b4ec1edf0ba60adb"),
    ("c2", 64113, "ba64f5cb16626417a64089fc980ea434"),
    ("c2", 64116, "cf318e0f4aa7ff766fae0d8313f101c2"),
    ("c2", 66857, "3d1529b69d7d6508be66b0c7fea605f1"),
    ("c2", 66873, "22371d0f941b52fd44359501d9c1d2e6"),
    ("c2", 66877, "85fe5e6abb654dc1fba44c4419ebea32"),
    ("c2", 82660, "8d6f9d503ccbc976f4fd6e2d9d8f8050"),
    ("c2", 82682, "a6aa4d136ecdc220a3abd3cb09f2a5ab"),
    ("c2", 221642, "a66e6480392cbc33a81de12bd8aad591"),
    ("c2", 282868, "df0378b9b010723f2b3a0da3aaa70efd"),
]


def load_sample(inputs):
    """The 11 rows' prompts from the bundle, after checking the sample file against the manifest."""
    inputs = pathlib.Path(inputs)
    path = inputs / "rationale_sample.parquet"
    manifest = dict(reversed(l.split("  ", 1)) for l in (inputs / "MANIFEST.sha256").read_text().splitlines())
    expected = manifest["data/step8/rationale_sample.parquet"]
    if sha256(path) != expected:
        raise SystemExit(f"{path} does not match MANIFEST.sha256")
    sample = pd.read_parquet(path)
    keys = pd.DataFrame(DIAG_ROWS, columns=["model", "session_id", "row_id"])
    rows = keys.merge(sample, on=["model", "session_id", "row_id"], how="left", validate="1:1")
    if rows["prompt"].isna().any():
        raise SystemExit("diagnostic rows missing from rationale_sample.parquet")
    return rows, expected


class FirstCall:
    """Forward pre-hook: attention mask and position ids of the first forward call (the prefill)."""

    def __init__(self, model):
        self.seen = None
        self.handle = model.register_forward_pre_hook(self, with_kwargs=True)

    def __call__(self, _module, args, kwargs):
        if self.seen is None:
            get = lambda k: None if kwargs.get(k) is None else kwargs[k].detach().cpu().numpy()
            self.seen = {k: get(k) for k in ("input_ids", "attention_mask", "position_ids", "cache_position")}

    def take(self):
        seen, self.seen = self.seen, None
        return seen

    def remove(self):
        self.handle.remove()


def last_position(seen, i):
    """Mask, position id and cache position at row i's last prompt position (the last column)."""
    pick = lambda a: None if a is None else (int(a[-1]) if a.ndim == 1 else int(a[i if a.shape[0] > 1 else 0, -1]))
    return {"prompt_tokens_padded": int(seen["input_ids"].shape[1]),
            "pad_tokens": None if seen["attention_mask"] is None else int((seen["attention_mask"][i] == 0).sum()),
            "attention_mask_last": pick(seen["attention_mask"]),
            "position_id_last": pick(seen["position_ids"]),
            "cache_position_last": pick(seen["cache_position"])}


def step(logits_row, tok, yes_id, no_id):
    """Fields for one first-step logit vector (full vocabulary)."""
    import torch
    v = logits_row.float()
    top = torch.topk(v, TOP)
    gen = int(top.indices[0])
    pair = v[[yes_id, no_id]]
    return {"generated_id": gen, "generated_token": tok.decode([gen]),
            "generated_is_yes_or_no": gen in (yes_id, no_id),
            "top5": [{"id": int(i), "token": tok.decode([int(i)]), "logit": float(l)}
                     for i, l in zip(top.indices, top.values)],
            "yes_logit": float(pair[0]), "no_logit": float(pair[1]),
            "p_yes_pair": float(torch.softmax(pair, -1)[0]), "yes_rank": int((v > v[yes_id]).sum()),
            "no_rank": int((v > v[no_id]).sum())}


def first_generate_step(model, tok, texts, device_name, hook):
    """generate(max_new_tokens=1) on texts as one (left-padded if > 1) batch, rationales.generate's arguments."""
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device_name)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=1, do_sample=False, num_beams=1, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, output_logits=True, return_dict_in_generate=True)
    return out.logits[0], hook.take()


def plain_forward(model, tok, text, device_name, hook):
    import torch
    enc = tok([text], return_tensors="pt", add_special_tokens=True).to(device_name)
    with torch.inference_mode():
        out = model(**enc)
    return out.logits[:, -1, :], hook.take()


def diagnose(model, tok, rows, device_name, yes_id, no_id):
    hook = FirstCall(model)
    texts = rows["prompt"].tolist()
    _p, _m, pairs = score_texts(model, tok, texts[:1], device_name, 1, yes_id, no_id, return_logits=True)
    hook.take()                                                     # discard (warm-up call)
    padded_logits, padded_seen = first_generate_step(model, tok, texts, device_name, hook)
    out = []
    for i, r in enumerate(rows.itertuples()):
        p_yes, _m, pair = score_texts(model, tok, [r.prompt], device_name, 1, yes_id, no_id, return_logits=True)
        hook.take()
        b_logits, b_seen = first_generate_step(model, tok, [r.prompt], device_name, hook)
        c_logits, c_seen = plain_forward(model, tok, r.prompt, device_name, hook)
        rec = {"model": r.model, "session_id": int(r.session_id), "row_id": r.row_id, "y_true": int(r.y_true),
               "a_scoring": {"yes_logit": float(pair[0, 0]), "no_logit": float(pair[0, 1]),
                             "p_yes": float(p_yes[0])},
               "b_generate_b1": {**step(b_logits[0], tok, yes_id, no_id), "inputs": last_position(b_seen, 0)},
               "c_forward_b1": {**step(c_logits[0], tok, yes_id, no_id), "inputs": last_position(c_seen, 0)},
               "e_generate_padded": {**step(padded_logits[i], tok, yes_id, no_id), "batch_size": len(texts),
                                     "inputs": last_position(padded_seen, i)}}
        a_yes = rec["a_scoring"]["p_yes"] > 0.5
        rec["agrees_with_a"] = {k: (rec[k]["generated_id"] == yes_id) == a_yes and rec[k]["generated_is_yes_or_no"]
                                for k in ("b_generate_b1", "c_forward_b1", "e_generate_padded")}
        out.append(rec)
    hook.remove()
    return out


def environment():
    import torch
    import transformers
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "python": platform.python_version(), "platform": platform.platform(),
           "SLM_SCORE_DTYPE": os.environ.get("SLM_SCORE_DTYPE")}
    try:
        import peft
        env["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        env.update({"torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)})
    return env


def main():
    p = common.parser(__doc__)
    p.add_argument("--model", default="primary")
    p.add_argument("--models-dir")
    p.add_argument("--c1-adapter")
    p.add_argument("--c2-adapter")
    p.add_argument("--models", nargs="+", default=["c1", "c2"], choices=["c1", "c2"])
    p.add_argument("--inputs", default=str(INPUTS))
    p.add_argument("--machine", default="Lab PC (RTX 4500 Ada, Windows 11)")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    checksum = verify_test_checksum()
    print(f"test_ids.csv SHA-256 {checksum}", flush=True)
    if args.device == "cuda" and os.environ.get("SLM_SCORE_DTYPE") != "fp16":
        raise SystemExit("set SLM_SCORE_DTYPE=fp16 (scoring protocol: fp16, merged adapter)")
    common.device(args.device)
    rows, sample_sha = load_sample(args.inputs)
    log = argparse.Namespace(info=lambda fmt, *a: print(fmt % a, flush=True))     # stdout only, no log file
    records, ids = [], {}
    for key in args.models:
        adapter = getattr(args, f"{key}_adapter")
        if not adapter:
            raise SystemExit(f"--{key}-adapter is required for model {key}")
        model, tok, yes_id, no_id = load_model(args, adapter, log)
        ids[key] = {"yes_id": yes_id, "no_id": no_id, "dtype": str(next(model.parameters()).dtype),
                    "adapter": pathlib.PurePath(adapter).as_posix()}
        part = diagnose(model, tok, rows[rows["model"] == key], args.device, yes_id, no_id)
        for rec in part:
            print(f"{key} {rec['session_id']}: a p_yes {rec['a_scoring']['p_yes']:.4f}; "
                  f"b {rec['b_generate_b1']['generated_id']} {rec['b_generate_b1']['generated_token']!r}; "
                  f"c {rec['c_forward_b1']['generated_id']}; e {rec['e_generate_padded']['generated_id']}; "
                  f"agree {rec['agrees_with_a']}", flush=True)
        records += part
        del model
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()
    common.write_json(args.out, {
        "machine": args.machine, "test_ids_sha256": checksum, "device": args.device, "model": args.model,
        "inputs": "data/step8/rationale_sample.parquet", "inputs_sha256": sample_sha,
        "generation_batch_17nov": GEN_BATCH_17NOV, "answer_ids": ids, "environment": environment(),
        "rows": records,
        "summary": {k: int(np.sum([r["agrees_with_a"][k] for r in records]))
                    for k in ("b_generate_b1", "c_forward_b1", "e_generate_padded")} | {"n": len(records)}})
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
