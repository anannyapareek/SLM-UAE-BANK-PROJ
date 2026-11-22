"""Step 8 rationale-path diagnostic (decisions log 2026-11-22, 2026-11-23): why 11 rationale rows
(C1 1, C2 10) generated a yes/no label on the other side of 0.5 from their score.

Rows: DIAG_ROWS (the mismatch set in results/step8/rationales.parquet, checked by
src.slm.step8_report) + CONTROL_ROWS (7 rows that did not flip, chosen as the nearest to the flip
boundary in each prompt-token group; see the list). For each row, with the model's seed-1 adapter
merged as for scoring:
  a  scoring pass exactly as src.slm.score.score_texts (batch 1): fp32 " yes"/" no" logits, p_yes
  b  first generation step at the 17 Nov batch size (1: rationales.py generates one prompt per
     call, no padding) via generate() with the 17 Nov arguments (rationales.GEN_KWARGS_17NOV)
  c  the same first step at batch 1 as a plain forward pass outside generate() (no logits processors)
  e  the same first step through generate() with the model's rows as one left-padded batch (a check
     only: the 17 Nov rows were batch 1, so padding cannot explain them)
  f  (i) generate() with the exact 17 Nov arguments repeated 20 times (label counts);
     (ii) generate() with do_sample=False and repetition_penalty=1.0
  d  for b, c, e: attention mask and position ids (or cache position) seen by the first forward
     call, at the last prompt position
For each generate() step: the generated token (from the output sequence), top-5 raw logits, the
raw and the processed (after logits processors, e.g. repetition penalty) " yes"/" no" scores, and
the merged generation config. Per row: whether the prompt contains the " yes" / " no" token ids
(the repetition penalty acts on every token already in the input).

Reading (decisions log 2026-11-23): f-i labels identical over the 20 repeats and equal to b, and
f-ii agrees with a -> repetition penalty inherited from generation_config.json (deterministic, not
sampling); f-i labels vary over repeats -> sampling; generated id not " yes"/" no" -> label from
another token; c and f-ii both disagree with a -> prefill precision.

Reads only the bundle (data/step8/rationale_sample.parquet, checked against MANIFEST.sha256) and
the frozen test_ids.csv; writes only --out. No log file, no metrics, no regeneration of rationales.

Run (lab, after the final-compute scoring), with HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and
SLM_SCORE_DTYPE=fp16:
  python -m src.slm.rationale_diag --device cuda --model primary --models-dir models \\
      --c1-adapter runs\\grid_qwen15_r8_lr1e-4\\adapter --c2-adapter runs\\c2_qwen15_r8_lr1e-4_s1\\adapter
Writes results/step8/rationale_diag.json.
"""
import argparse
import collections
import os
import pathlib
import platform

import numpy as np
import pandas as pd

from src.slm import common
from src.slm.rationales import GEN_KWARGS, GEN_KWARGS_17NOV, load_model
from src.slm.score import score_texts
from src.slm.step8_inputs import sha256
from src.splits import verify_test_checksum

INPUTS = common.ROOT / "data/step8"
OUT = common.ROOT / "results/step8/rationale_diag.json"
GEN_BATCH_17NOV = 1          # src.slm.rationales.generate: one prompt per call
REPEATS = 20
TOP = 5
OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
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
# Controls (did not flip on 17 Nov), nearest to the boundary in each group (decisions log 2026-11-23).
CONTROL_ROWS = [
    ("c2", 282867, "2a06c0ff2322679c28a6e5d24f2b3af8"),  # " no" in prompt, p_yes 0.049 (highest below 0.5 that kept "no")
    ("c2", 15379, "58ad7103e986356eb4318f08564cb281"),   # " no" in prompt, p_yes 0.029
    ("c2", 104551, "4b04efb26502ce6173923d7694c1f9d4"),  # " no" in prompt, p_yes 0.519 (lowest above 0.5)
    ("c2", 144782, "89e80b458f164bbc6a78f6b533826075"),  # neither token, p_yes 0.500 (below 0.5)
    ("c2", 172732, "01456827e398bff4d3dd5a49db2c72be"),  # neither token, p_yes 0.501
    ("c1", 144766, "3b53c33a94469afffe334871eb35d55a"),  # neither token, p_yes 0.500 (below 0.5)
    ("c1", 144771, "5b6894a7164de5055327f2a64cd0d59b"),  # neither token, p_yes 0.502
]
COMPARED = ("b_generate_b1", "c_forward_b1", "e_generate_padded", "f_ii_greedy_no_penalty")


def load_sample(inputs):
    """The diagnostic and control rows' prompts from the bundle, after the manifest check."""
    inputs = pathlib.Path(inputs)
    path = inputs / "rationale_sample.parquet"
    manifest = dict(reversed(l.split("  ", 1)) for l in (inputs / "MANIFEST.sha256").read_text().splitlines())
    expected = manifest["data/step8/rationale_sample.parquet"]
    if sha256(path) != expected:
        raise SystemExit(f"{path} does not match MANIFEST.sha256")
    sample = pd.read_parquet(path)
    keys = pd.DataFrame([k + ("mismatch",) for k in DIAG_ROWS] + [k + ("control",) for k in CONTROL_ROWS],
                        columns=["model", "session_id", "row_id", "role"])
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


def step(raw, tok, yes_id, no_id, generated=None, processed=None):
    """Fields for one first step: raw logits (full vocabulary), optional processed scores and the
    token generate() actually emitted (default: argmax of the raw logits, for the plain forward)."""
    import torch
    v = raw.float()
    top = torch.topk(v, TOP)
    gen = int(top.indices[0]) if generated is None else int(generated)
    pair = v[[yes_id, no_id]]
    out = {"generated_id": gen, "generated_token": tok.decode([gen]),
           "generated_is_yes_or_no": gen in (yes_id, no_id),
           "top5_raw": [{"id": int(i), "token": tok.decode([int(i)]), "logit": float(l)}
                        for i, l in zip(top.indices, top.values)],
           "yes_logit": float(pair[0]), "no_logit": float(pair[1]),
           "p_yes_pair": float(torch.softmax(pair, -1)[0])}
    if processed is not None:
        q = processed.float()
        out.update({"yes_processed": float(q[yes_id]), "no_processed": float(q[no_id]),
                    "argmax_processed": int(q.argmax())})
    return out


def merged_config(model, kwargs):
    """The generation config generate() would use for these kwargs (model config merged with kwargs)."""
    gc, _ = model._prepare_generation_config(None, **kwargs)
    return {k: getattr(gc, k, None) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
                                              "num_beams", "eos_token_id", "pad_token_id", "max_new_tokens")}


def first_generate_step(model, tok, texts, device_name, hook, kwargs):
    """generate(max_new_tokens=1) on texts as one (left-padded if > 1) batch."""
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device_name)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=1, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             output_logits=True, output_scores=True, return_dict_in_generate=True, **kwargs)
    return out.logits[0], out.scores[0], out.sequences[:, -1], hook.take()


def plain_forward(model, tok, text, device_name, hook):
    import torch
    enc = tok([text], return_tensors="pt", add_special_tokens=True).to(device_name)
    with torch.inference_mode():
        out = model(**enc)
    return out.logits[:, -1, :], hook.take()


NO_PENALTY = GEN_KWARGS          # (f-ii): the corrected protocol, greedy with repetition_penalty 1.0


def diagnose(model, tok, rows, device_name, yes_id, no_id):
    hook = FirstCall(model)
    texts = rows["prompt"].tolist()
    score_texts(model, tok, texts[:1], device_name, 1, yes_id, no_id)          # warm-up call
    hook.take()
    pad_raw, pad_proc, pad_gen, pad_seen = first_generate_step(model, tok, texts, device_name, hook, GEN_KWARGS_17NOV)
    out = []
    for i, r in enumerate(rows.itertuples()):
        p_yes, _m, pair = score_texts(model, tok, [r.prompt], device_name, 1, yes_id, no_id, return_logits=True)
        hook.take()
        ids = set(tok(r.prompt, add_special_tokens=True)["input_ids"])
        b_raw, b_proc, b_gen, b_seen = first_generate_step(model, tok, [r.prompt], device_name, hook, GEN_KWARGS_17NOV)
        c_raw, c_seen = plain_forward(model, tok, r.prompt, device_name, hook)
        f_raw, f_proc, f_gen, _ = first_generate_step(model, tok, [r.prompt], device_name, hook, NO_PENALTY)
        repeats = collections.Counter()
        for _ in range(REPEATS):
            *_x, g, _s = first_generate_step(model, tok, [r.prompt], device_name, hook, GEN_KWARGS_17NOV)
            repeats[tok.decode([int(g[0])])] += 1
        rec = {"model": r.model, "session_id": int(r.session_id), "row_id": r.row_id, "role": r.role,
               "y_true": int(r.y_true), "prompt_has_yes_token": yes_id in ids, "prompt_has_no_token": no_id in ids,
               "a_scoring": {"yes_logit": float(pair[0, 0]), "no_logit": float(pair[0, 1]), "p_yes": float(p_yes[0])},
               "b_generate_b1": {**step(b_raw[0], tok, yes_id, no_id, b_gen[0], b_proc[0]),
                                 "inputs": last_position(b_seen, 0)},
               "c_forward_b1": {**step(c_raw[0], tok, yes_id, no_id), "inputs": last_position(c_seen, 0)},
               "e_generate_padded": {**step(pad_raw[i], tok, yes_id, no_id, pad_gen[i], pad_proc[i]),
                                     "batch_size": len(texts), "inputs": last_position(pad_seen, i)},
               "f_i_repeats_17nov_kwargs": {"repeats": REPEATS, "labels": dict(repeats)},
               "f_ii_greedy_no_penalty": step(f_raw[0], tok, yes_id, no_id, f_gen[0], f_proc[0])}
        a_yes = rec["a_scoring"]["p_yes"] > 0.5
        rec["agrees_with_a"] = {k: rec[k]["generated_is_yes_or_no"] and (rec[k]["generated_id"] == yes_id) == a_yes
                                for k in COMPARED}
        out.append(rec)
    hook.remove()
    return out


def offline_env():
    return {k: os.environ.get(k) for k in OFFLINE_VARS}


def environment():
    import torch
    import transformers
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "python": platform.python_version(), "platform": platform.platform(),
           "SLM_SCORE_DTYPE": os.environ.get("SLM_SCORE_DTYPE"), **offline_env()}
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
    print(f"test_ids.csv SHA-256 {checksum}; offline {offline_env()}", flush=True)
    if args.device == "cuda" and os.environ.get("SLM_SCORE_DTYPE") != "fp16":
        raise SystemExit("set SLM_SCORE_DTYPE=fp16 (scoring protocol: fp16, merged adapter)")
    if args.device == "cuda" and any(os.environ.get(k) != "1" for k in OFFLINE_VARS):
        raise SystemExit("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 (decisions log 2026-11-23)")
    common.device(args.device)
    rows, sample_sha = load_sample(args.inputs)
    log = argparse.Namespace(info=lambda fmt, *a: print(fmt % a, flush=True))     # stdout only, no log file
    records, models = [], {}
    for key in args.models:
        adapter = getattr(args, f"{key}_adapter")
        if not adapter:
            raise SystemExit(f"--{key}-adapter is required for model {key}")
        model, tok, yes_id, no_id = load_model(args, adapter, log)
        models[key] = {"yes_id": yes_id, "no_id": no_id, "dtype": str(next(model.parameters()).dtype),
                       "adapter": pathlib.PurePath(adapter).as_posix(),
                       "model_generation_config": model.generation_config.to_diff_dict(),
                       "merged_17nov": merged_config(model, {**GEN_KWARGS_17NOV, "max_new_tokens": 64}),
                       "merged_no_penalty": merged_config(model, {**NO_PENALTY, "max_new_tokens": 1})}
        part = diagnose(model, tok, rows[rows["model"] == key], args.device, yes_id, no_id)
        for rec in part:
            print(f"{key} {rec['session_id']} ({rec['role']}): a p_yes {rec['a_scoring']['p_yes']:.4f}; "
                  f"b {rec['b_generate_b1']['generated_token']!r}; f-i {rec['f_i_repeats_17nov_kwargs']['labels']}; "
                  f"f-ii {rec['f_ii_greedy_no_penalty']['generated_token']!r}; agree {rec['agrees_with_a']}", flush=True)
        records += part
        del model
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()
    common.write_json(args.out, {
        "machine": args.machine, "test_ids_sha256": checksum, "device": args.device, "model": args.model,
        "inputs": "data/step8/rationale_sample.parquet", "inputs_sha256": sample_sha,
        "generation_batch_17nov": GEN_BATCH_17NOV, "gen_kwargs_17nov": GEN_KWARGS_17NOV, "gen_kwargs_v2": GEN_KWARGS, "models": models,
        "environment": environment(), "rows": records,
        "summary": {"n": len(records), "agrees_with_a": {k: int(np.sum([r["agrees_with_a"][k] for r in records]))
                                                         for k in COMPARED},
                    "f_i_rows_with_varying_labels": int(sum(len(r["f_i_repeats_17nov_kwargs"]["labels"]) > 1
                                                            for r in records))}})
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
