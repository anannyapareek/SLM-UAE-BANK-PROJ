"""Score rows with an SLM (runbook 5.4): P(yes) for the token after "Label:".

For each row the input is serialise.scoring_prompt(row), which ends in "\\nLabel:". Batches
are left-padded, so the last position of every row is its "Label:" token. The score is the
softmax over the " yes" and " no" logits at that position: p_yes = softmax([yes, no])[0];
margin = logit(yes) - logit(no). The two logits are computed in fp32 from the final hidden
state (bf16 logits are too coarse and would tie many scores); only the last position is
projected (logits_to_keep=1).

Run: python -m src.slm.score --device cuda --model primary --split valid \\
         [--adapter DIR] [--rows N] [--only-final] --out results/<run>/preds_valid.parquet
Writes preds.parquet with row_id, y_true, score, margin.
"""
import pathlib
import time

import numpy as np
import pandas as pd

from src.slm import common, serialise
from src.splits import verify_test_checksum


def score_texts(model, tok, texts, device_name, batch_size, yes_id, no_id, log=None):
    """p_yes and margin for each text, in input order. Texts are batched by length."""
    import torch
    order = np.argsort([len(t) for t in texts], kind="stable")
    p_yes, margin = np.empty(len(texts)), np.empty(len(texts))
    model.eval()
    head = model.get_output_embeddings()
    weight = head.weight[[yes_id, no_id]].detach().float()
    bias = None if getattr(head, "bias", None) is None else head.bias[[yes_id, no_id]].float()
    captured = {}
    hook = head.register_forward_hook(
        lambda _m, inputs, _out: captured.__setitem__("h", inputs[0][:, -1, :]))
    start = time.time()
    with torch.inference_mode():
        for b in range(0, len(texts), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device_name)
            # explicit positions so left padding never shifts a row's positions
            positions = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            model(**enc, position_ids=positions, logits_to_keep=1)
            pair = captured["h"].float() @ weight.T
            if bias is not None:
                pair = pair + bias
            p_yes[idx] = torch.softmax(pair, dim=-1)[:, 0].cpu().numpy()
            margin[idx] = (pair[:, 0] - pair[:, 1]).cpu().numpy()
            if log and (b // batch_size) % 50 == 0:
                log.info("scored %d / %d rows (%.1f s)", min(b + batch_size, len(texts)),
                         len(texts), time.time() - start)
    hook.remove()
    return p_yes, margin


def load_for_scoring(args, log):
    tok = common.load_tokenizer(args.model, args.models_dir)
    if tok.padding_side != "left":
        raise SystemExit("tokenizer must pad on the left")
    model = common.load_model(args.model, args.device, args.models_dir)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to(args.device)
        model = model.merge_and_unload().eval()  # merge LoRA for scoring speed; same model
        log.info("adapter %s", args.adapter)
    yes_id, no_id = common.answer_ids(tok)
    log.info("model %s; answer ids yes=%d no=%d; dtype %s", args.model, yes_id, no_id,
             next(model.parameters()).dtype)
    return model, tok, yes_id, no_id


def add_model_args(p):
    p.add_argument("--model", default="primary",
                   help="role in configs/slm.yaml (primary, second, fallback) or a model id/path")
    p.add_argument("--models-dir", help="local model copies (scripts/download_models.py)")
    p.add_argument("--adapter", help="LoRA adapter folder from train_lora.py")
    p.add_argument("--batch-size", type=int, default=common.config()["scoring"]["batch_size"])


def main():
    p = common.parser(__doc__)
    add_model_args(p)
    p.add_argument("--split", choices=common.SPLITS, required=True)
    p.add_argument("--rows", type=int, help="random sample of this many rows (seed 1)")
    p.add_argument("--only-final", action="store_true", help="logoff (final) rows only")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--text", action="store_true",
                   help="Comparison 2: add each row's text block (from --text-blocks)")
    p.add_argument("--text-blocks", default=str(common.ROOT / common.config()["text"]["blocks"]))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    log = common.setup_logging("score", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    common.device(args.device, log)

    rows = common.load_rows(args.split, n=args.rows, allow_test=args.allow_test,
                            only_final=args.only_final)
    log.info("%d %s rows", len(rows), args.split)
    if args.text:                    # check the blocks before loading the model
        from src.slm.make_train import load_blocks
        blocks = load_blocks(args.text_blocks, list(rows["row_id"]))
    model, tok, yes_id, no_id = load_for_scoring(args, log)
    if args.text:
        texts = [serialise.text_scoring_prompt(r, blocks[r["row_id"]])
                 for r in rows.to_dict("records")]
        log.info("text prompts: blocks from %s", args.text_blocks)
    else:
        texts = [serialise.scoring_prompt(r) for r in rows.to_dict("records")]
    start = time.time()
    p_yes, margin = score_texts(model, tok, texts, args.device, args.batch_size, yes_id, no_id,
                                log)
    log.info("scored %d rows in %.1f s", len(texts), time.time() - start)
    out = pd.DataFrame({"row_id": rows["row_id"], "y_true": rows["y"].astype(int),
                        "score": p_yes, "margin": margin})
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
