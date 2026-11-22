"""Step 8 consistency check for the rationale label/score mismatches (decisions log 2026-11-23).
Not a metric: counts and arithmetic on the stored p_yes. Tokenizer only, no model, no GPU.

17 Nov generation was greedy (do_sample=False), so sampling predicts 0 label flips. But
generate() also applied repetition_penalty 1.1 from Qwen2.5-1.5B-Instruct's generation_config.json
(rationales.GEN_KWARGS did not override it). That penalty divides a positive logit by 1.1 for every
token already in the input, so the first step is deterministic and flips the label only when exactly
one of " yes" / " no" occurs in the prompt and the score's margin is small enough:
  " no" in prompt only, p_yes < 0.5:  flips to "yes" iff  margin > -(1 - 1/1.1) * no_logit
  " yes" in prompt only, p_yes > 0.5: flips to "no"  iff  margin <  (1 - 1/1.1) * yes_logit
  both or neither: no change (both or neither penalised).
Absolute logits are not stored (only p_yes), so the check is whether one threshold per group
separates the flipped from the unflipped rows, and what logit that implies. The 24 Nov diagnostic
(src.slm.rationale_diag, step f) measures it directly.

Reads results/step8/rationales.parquet and data/step8/rationale_sample.parquet; writes
results/step8/rationale_penalty_check.json. Run (Mac, SLM venv):
  python -m src.slm.rationale_penalty_check
"""
import json
import math

import numpy as np
import pandas as pd

from src.slm import common
from src.splits import verify_test_checksum

D = common.ROOT / "results/step8"
PENALTY = 1.1
SHRINK = 1 - 1 / PENALTY


def poisson_ci(k, level=0.95):
    """Exact (Garwood) interval for a Poisson count k."""
    from scipy.stats import chi2
    a = 1 - level
    return (0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2, chi2.ppf(1 - a / 2, 2 * k + 2) / 2)


def main():
    checksum = verify_test_checksum()
    tok = common.load_tokenizer("primary")
    yes_id, no_id = common.answer_ids(tok)
    rat = pd.read_parquet(D / "rationales.parquet")
    sample = pd.read_parquet(common.ROOT / "data/step8/rationale_sample.parquet")
    x = rat.merge(sample[["model", "row_id", "prompt"]], on=["model", "row_id"], validate="1:1")
    ids = [set(tok(p, add_special_tokens=True)["input_ids"]) for p in x["prompt"]]
    x["has_yes"], x["has_no"] = [yes_id in i for i in ids], [no_id in i for i in ids]
    x["flip"] = (x["label_text"] == "yes") != (x["p_yes"] > 0.5)
    x["margin"] = np.log(x["p_yes"] / (1 - x["p_yes"]))
    x["group"] = np.select([x.has_yes & x.has_no, x.has_no, x.has_yes], ["both", "no_only", "yes_only"], "neither")
    out = {"test_ids_sha256": checksum, "answer_ids": {"yes": yes_id, "no": no_id}, "repetition_penalty": PENALTY,
           "sampling_expected_flips": 0, "models": {}}
    for key, g in x.groupby("model"):
        groups = {}
        for name, h in g.groupby("group"):
            entry = {"rows": len(h), "flips": int(h.flip.sum())}
            if name == "no_only":
                cand = h[h.p_yes < 0.5]
                f, k = cand[cand.flip], cand[~cand.flip]
                entry.update({"score_no_rows": len(cand), "flipped_margin_min": float(f.margin.min()) if len(f) else None,
                              "kept_margin_max": float(k.margin.max()) if len(k) else None})
                if len(f) and len(k):
                    entry["separable"] = bool(f.margin.min() > k.margin.max())
                    entry["implied_no_logit_range"] = [-f.margin.min() / SHRINK, -k.margin.max() / SHRINK]
            if name == "yes_only":
                cand = h[h.p_yes > 0.5]
                f, k = cand[cand.flip], cand[~cand.flip]
                entry.update({"score_yes_rows": len(cand), "flipped_margin_max": float(f.margin.max()) if len(f) else None,
                              "kept_margin_min": float(k.margin.min()) if len(k) else None})
                if len(f):
                    entry["implied_yes_logit_min"] = f.margin.max() / SHRINK
            entry["flips_off_penalised_side"] = int(h[h.flip & ~(((h.group == "no_only") & (h.p_yes < 0.5)) |
                                                                  ((h.group == "yes_only") & (h.p_yes > 0.5)))].shape[0])
            groups[name] = entry
        flips = int(g.flip.sum())
        lo, hi = poisson_ci(flips)
        out["models"][key] = {"rows": len(g), "observed_flips": flips, "poisson_95": [lo, hi],
                              "flips_explained_by_penalty": int(g.flip.sum() - sum(e["flips_off_penalised_side"]
                                                                                 for e in groups.values())),
                              "groups": groups}
    common.write_json(D / "rationale_penalty_check.json", out)
    for key, m in out["models"].items():
        print(key, json.dumps(m, default=float))


if __name__ == "__main__":
    main()
