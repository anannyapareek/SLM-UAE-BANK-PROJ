"""Step 8.1 report (Mac): checks the 17 Nov lab outputs against the 2026-11-15 protocol and writes the
operations table, the rationale counts and the plain-language operations note.

Reads results/step8/latency_gpu.json, latency_xgb_lab.json, rationales.parquet (lab), the bundle
(data/step8/rationale_sample.parquet), the stored seed-1 test scores of C1 and C2, and
results/step7/table3/compute.json (routing shares and the Step 8.1 compute line written by
scripts/step7_table3.py). Writes results/step8/:
  operations.md / .csv     one row per configuration (XGBoost, SLM structured, SLM text,
                           SLM + rationale, hybrid): batch-1 median/p95, throughput, memory, versions
  rationales.md            counts, label/score mismatches, token lengths (no examples yet: S8-2)
  operations_for_L.md      one-page plain-language operations note
  run.csv                  run-log rows for the three lab runs and this report
Computes no metrics (src/eval.py does); every number is a timing, a count or arithmetic on them.
Run from the repo root: python -m src.slm.step8_report
"""
import datetime
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

from src.slm.rationale_diag import DIAG_ROWS
from src.splits import verify_test_checksum

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "results/step8"
LAB = "Lab PC (RTX 4500 Ada, Windows 11)"
MAC = "Mac (arm64, Darwin)"
STORED = {"c1": ROOT / "results/c1_qwen15_r8_lr1e-4_s1/preds_test.parquet",
          "c2": ROOT / "results/c2_qwen15_r8_lr1e-4_s1/preds_test_final.parquet"}
STORED_TOL = 0.02               # |lab p_yes - stored score|: fp16 lab scoring vs the stored run
NEAR_HALF = (0.45, 0.55)        # v1: mismatches inside this band are read as fp16 noise
V2_NOISE = 0.01                 # v2: every mismatch must have |p_yes - 0.5| < 0.01 (decisions log 2026-11-23)
FILES = {"v1": ("rationales.parquet", "17 Nov, repetition_penalty 1.1 inherited from the model's generation config"),
         "v2": ("rationales_v2.parquet", "24 Nov, greedy with repetition_penalty 1.0 (logged protocol)")}
CONFIGS = ["a_score_structured_b1", "b_score_text_b1", "c_score_rationale_structured_b1",
           "a_score_structured_b16", "b_score_text_b16"]
WEEKDAY_SESSIONS = 963.9        # results/step7/review_budget/capacity_facts.md (test, per weekday)


def load(name):
    return json.loads((D / name).read_text())


def check_protocol(gpu, xgb, checksum):
    """Raise on any departure from the 2026-11-15 protocol."""
    bad = []
    for name, d in (("latency_gpu", gpu), ("latency_xgb_lab", xgb)):
        if d["machine"] != LAB or d["test_ids_sha256"] != checksum:
            bad.append(f"{name}: machine or test SHA-256")
        if pathlib.PurePath(d["inputs"]).is_absolute() or ":" in d["inputs"] or "\\" in d["inputs"]:
            bad.append(f"{name}: inputs path not repo-relative")
    p = gpu["protocol"]
    if gpu.get("partial") or (p["warmup_calls"], p["batch"], p["throughput_batch_size"]) != (20, 1, 16):
        bad.append("latency_gpu: partial or protocol")
    if p["rationale_max_new_tokens"] != 48 or p["decoding"] != "greedy" or gpu["rows"] != 1000:
        bad.append("latency_gpu: rationale protocol or rows")
    if sorted(gpu["configs"]) != sorted(CONFIGS):
        bad.append(f"latency_gpu: configs {sorted(gpu['configs'])}")
    n = {k: gpu["configs"][k].get("latency", {}).get("n", gpu["configs"][k].get("rows")) for k in CONFIGS}
    if n != {"a_score_structured_b1": 1000, "b_score_text_b1": 1000, "c_score_rationale_structured_b1": 150,
             "a_score_structured_b16": 1000, "b_score_text_b16": 1000}:
        bad.append(f"latency_gpu: rows per config {n}")
    if xgb["check_max_abs_diff"] > 1e-6 or xgb["rows"] != 1000 or xgb["batch1"]["warmup_calls"] != 20:
        bad.append("latency_xgb_lab: check, rows or warm-up")
    if xgb["full_batch_tiled"]["rows"] != 100_000:
        bad.append("latency_xgb_lab: tiled rows")
    if bad:
        raise SystemExit("protocol check failed: " + "; ".join(bad))


def mismatches(rat):
    return rat[(rat["label_text"] == "yes") != (rat["p_yes"] > 0.5)]


def check_rationales(rat, version="v1"):
    """Row-ID match with the bundle sample; lab p_yes vs stored scores. v1 (17 Nov): mismatch set =
    DIAG_ROWS. v2 (24 Nov, no penalty): every mismatch within fp16 noise, |p_yes - 0.5| < V2_NOISE."""
    sample = pd.read_parquet(ROOT / "data/step8/rationale_sample.parquet")
    keys = ["session_id", "row_id", "model", "y_true"]
    merged = rat[keys].merge(sample[keys], how="outer", indicator=True)
    if len(rat) != len(sample) or (merged["_merge"] != "both").any() or rat.duplicated(["model", "row_id"]).any():
        raise SystemExit("rationales.parquet does not match the bundle's rationale_sample.parquet")
    diffs = {}
    for key, path in STORED.items():
        stored = pd.read_parquet(path, columns=["row_id", "score"])
        x = rat[rat["model"] == key].merge(stored, on="row_id", how="left", validate="1:1")
        if x["score"].isna().any():
            raise SystemExit(f"{key}: rationale rows missing from {path}")
        diffs[key] = float((x["p_yes"] - x["score"]).abs().max())
        if diffs[key] > STORED_TOL:
            raise SystemExit(f"{key}: lab p_yes differs from stored test scores by {diffs[key]:.4f}")
    mism = mismatches(rat)
    if version == "v1":
        found = set(zip(mism["model"], mism["session_id"], mism["row_id"]))
        if found != set(DIAG_ROWS):
            raise SystemExit("label/score mismatch set differs from src.slm.rationale_diag.DIAG_ROWS")
    else:
        far = mism[(mism["p_yes"] - 0.5).abs() >= V2_NOISE]
        if len(far):
            raise SystemExit(f"{version}: {len(far)} label/score mismatches outside fp16 noise "
                             f"(|p_yes - 0.5| >= {V2_NOISE}): {far[['model', 'session_id', 'p_yes']].to_dict('records')}")
    return diffs, mism


def versions(gpu, xgb):
    e, x = gpu["environment"], xgb["environment"]
    return {"gpu": e["gpu_name"], "driver": e["driver"], "cuda": e["torch_cuda"], "torch": e["torch"],
            "transformers": e["transformers"], "peft": e["peft"], "cudnn": e["cudnn"], "xgboost": x["xgboost"],
            "python": e["python"], "os": e["platform"], "cpu": f"{e['cpu']} ({e['cpu_count']} logical)"}


def operations(gpu, xgb, compute):
    c = gpu["configs"]
    v = versions(gpu, xgb)
    slm_sw = f"torch {v['torch']}, CUDA {v['cuda']}, driver {v['driver']}, transformers {v['transformers']}, peft {v['peft']}"
    hw_gpu = f"{v['gpu']} (fp16, merged LoRA)"

    def slm(name, b1, b16, prompt):
        lat, mem = c[b1]["latency"], c[b1]["memory"]
        tp = c[b16] if b16 else None
        vram = max(c[k]["memory"]["gpu_max_memory_allocated_gb"] for k in (b1, b16) if k)
        rss = max(c[k]["memory"]["rss_gb"] for k in (b1, b16) if k)
        return {"config": name, "hardware": hw_gpu, "software": slm_sw, "prompt_tokens_mean": c[b1]["prompt_tokens_mean"],
                "batch1_median_ms": lat["median_ms"], "batch1_p95_ms": lat["p95_ms"], "batch1_n": lat["n"],
                "throughput_rows_per_s": tp["rows_per_second"] if tp else None,
                "throughput_batch": tp["batch_size"] if tp else None,
                "peak_vram_allocated_gb": vram, "rss_gb": rss, "note": prompt}

    rows = [
        {"config": "XGBoost v3 seed 1", "hardware": f"lab PC CPU, {v['cpu']}", "software": f"xgboost {v['xgboost']}, nthread 0 (all)",
         "prompt_tokens_mean": None, "batch1_median_ms": xgb["batch1"]["median_ms"], "batch1_p95_ms": xgb["batch1"]["p95_ms"],
         "batch1_n": xgb["batch1"]["n"], "throughput_rows_per_s": xgb["full_batch_tiled"]["rows_per_second"],
         "throughput_batch": xgb["full_batch_tiled"]["rows"], "peak_vram_allocated_gb": None, "rss_gb": None,
         "note": f"44 features in memory; {xgb['full_batch']['rows_per_second']:,.0f} rows/s at a 1,000-row batch; RSS not recorded"},
        slm("SLM structured (C1 seed 1)", "a_score_structured_b1", "a_score_structured_b16", "score only"),
        slm("SLM text (C2 seed 1)", "b_score_text_b1", "b_score_text_b16", "score only"),
        slm("SLM structured + rationale (C1 seed 1)", "c_score_rationale_structured_b1", None,
            f"score + greedy rationale, max 48 new tokens ({c['c_score_rationale_structured_b1']['new_tokens_mean_first20']:.1f} "
            "mean over the first 20 rows); batch 1 only"),
    ]
    share, sess = compute["routing"]["row_share"][0], compute["routing"]["session_share"][0]
    a, x = rows[1], rows[0]
    rows.append({"config": "Hybrid (XGBoost; SLM C1 in band)", "hardware": "both of the above", "software": "both of the above",
                 "prompt_tokens_mean": None,
                 "batch1_median_ms": share * a["batch1_median_ms"] + x["batch1_median_ms"], "batch1_p95_ms": None,
                 "batch1_n": None, "throughput_rows_per_s": compute["compute"]["implied_rows_per_second"],
                 "throughput_batch": None, "peak_vram_allocated_gb": a["peak_vram_allocated_gb"], "rss_gb": None,
                 "note": (f"effective per-row latency = {share:.1%} × SLM batch-1 median + XGBoost batch-1 median "
                          f"(expected value, not a measured distribution; no p95); a routed row costs "
                          f"{a['batch1_median_ms'] + x['batch1_median_ms']:.1f} ms, an unrouted row {x['batch1_median_ms']:.1f} ms; "
                          f"{sess:.1%} of test sessions send at least one row to the SLM; throughput = Table 3 compute line")})
    return pd.DataFrame(rows), v, share, sess


def fmt(v, spec):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def write_operations(ops, v, gpu):
    ops.to_csv(D / "operations.csv", index=False)
    md = ["# Step 8.1 operations table", "",
          f"All timings from one lab session on 17 Nov 2026 ({LAB}), protocol fixed before the session "
          "(decisions log 2026-11-15): batch 1 = 20 warm-up calls then one call per row, device synchronised "
          "before and after each call; SLM = Qwen2.5-1.5B + seed-1 LoRA adapter merged, fp16; a call = tokenise, "
          "forward, fp32 yes/no head, copy to host. Inputs: 1,000 test logoff rows (numpy default_rng(0)); "
          "rationale config on the first 150. Sources: `latency_gpu.json`, `latency_xgb_lab.json`. "
          "Timings, not metrics.", "",
          "| Configuration | Batch-1 median (ms) | Batch-1 p95 (ms) | Throughput (rows/s) | Peak VRAM allocated (GB) | RSS (GB) | Prompt tokens (mean) |",
          "|---|---|---|---|---|---|---|"]
    for r in ops.itertuples():
        tp = fmt(r.throughput_rows_per_s, ",.1f" if (r.throughput_rows_per_s or 0) < 1e4 else ",.0f") + ("" if r.throughput_batch is None or pd.isna(r.throughput_batch)
                                                      else f" (batch {int(r.throughput_batch):,})")
        md.append(f"| {r.config} | {fmt(r.batch1_median_ms, '.2f')} | {fmt(r.batch1_p95_ms, '.2f')} | {tp} | "
                  f"{fmt(r.peak_vram_allocated_gb, '.2f')} | {fmt(r.rss_gb, '.2f')} | {fmt(r.prompt_tokens_mean, '.0f')} |")
    md += ["", "Notes:", ""] + [f"- **{r.config}**: {r.note}." for r in ops.itertuples()]
    md += ["", "Peak VRAM = `torch.cuda.max_memory_allocated`, reset per configuration, weights included "
           "(reserved: up to "
           f"{max(c['memory']['gpu_max_memory_reserved_gb'] for c in gpu['configs'].values()):.2f} GB). "
           "RSS = psutil after each configuration.", "",
           "## Hardware and software", "", "| Item | Value |", "|---|---|"]
    md += [f"| {k} | {val} |" for k, val in v.items()]
    md += ["", f"SLM latency run time {gpu['runtime_seconds']:.0f} s. Test set SHA-256 "
           f"`{gpu['test_ids_sha256']}` in both files.", ""]
    (D / "operations.md").write_text("\n".join(md))


def agreement(runs):
    """Label/score agreement per file and model (counts, not a metric)."""
    md = ["## Label/score agreement", "",
          "Generated label equal to the score's side of 0.5 (p_yes > 0.5 ↔ \"yes\"); the decision in every "
          "reported result is the score.", "",
          "| File | Model | Rows | Agree | Disagree | Disagree with \\|p_yes − 0.5\\| ≥ 0.01 | Max \\|lab p_yes − stored score\\| |",
          "|---|---|---|---|---|---|---|"]
    for version, (rat, diffs, mism) in runs.items():
        for key, g in rat.groupby("model"):
            m = mism[mism.model == key]
            md.append(f"| {FILES[version][0]} ({version}: {FILES[version][1]}) | {key.upper()} | {len(g)} | {len(g) - len(m)} | "
                      f"{len(m)} | {int(((m.p_yes - 0.5).abs() >= V2_NOISE).sum())} | {diffs[key]:.4f} |")
    if "v2" not in runs:
        md += ["", "`rationales_v2.parquet`: not yet generated (24 Nov lab session, `docs/final_compute_lab.md`).",
               "When present, it must show 0 disagreements outside fp16 noise (|p_yes − 0.5| < 0.01) or this report stops."]
    return md + [""]


def counts(rat):
    md = ["| Model | Sessions | Positive | Negative | Generated \"yes\" | Generated \"no\" | Other label | Empty reason |",
          "|---|---|---|---|---|---|---|---|"]
    for key, g in rat.groupby("model"):
        md.append(f"| {key.upper()} | {len(g)} | {(g.y_true == 1).sum()} | {(g.y_true == 0).sum()} | "
                  f"{(g.label_text == 'yes').sum()} | {(g.label_text == 'no').sum()} | "
                  f"{(~g.label_text.isin(['yes', 'no'])).sum()} | {(g.reason_text == '').sum()} |")
    md += ["", "| Model | New tokens mean | Median | Min | Max | Seconds per row (median, scoring + generation) |",
           "|---|---|---|---|---|---|"]
    for key, g in rat.groupby("model"):
        t = g.n_tokens
        md.append(f"| {key.upper()} | {t.mean():.1f} | {t.median():.0f} | {t.min()} | {t.max()} | {g.seconds.median():.3f} |")
    return md


def write_rationales(runs):
    rat, diffs, mism = runs["v1"]
    md = ["# Step 8.1 rationales: counts", "",
          "Sample (decisions log 2026-11-15): all 243 positive test sessions + 100 negative (numpy "
          "default_rng(0)); C1 = each session's max-score row under the C1 seed-1 test scores, C2 = its "
          "logoff row; greedy generation, max 64 new tokens, from the scoring prompt; batch 1 on the lab GPU "
          "(fp16, merged seed-1 adapter). `rationales.parquet` (untracked). Examples and the faithfulness "
          "check follow in S8-2 on `rationales_v2.parquet` (decisions log 2026-11-23).", ""]
    md += agreement(runs)
    if "v2" in runs:
        md += [f"## {FILES['v2'][0]} ({FILES['v2'][1]})", ""] + counts(runs["v2"][0]) + [""]
    md += [f"## {FILES['v1'][0]} ({FILES['v1'][1]}; kept for the bug log)", ""] + counts(rat)
    lo, hi = NEAR_HALF
    near = mism[(mism.p_yes >= lo) & (mism.p_yes <= hi)]
    per = mism.groupby("model").size().reindex(["c1", "c2"], fill_value=0)
    n = rat.groupby("model").size()
    md += ["", "## Generated label vs score (v1, 17 Nov)", "",
           f"In {len(mism)} of {len(rat)} rows the generated label is on the other side of 0.5 from the score "
           f"p_yes: C1 {per['c1']}/{n['c1']}, C2 {per['c2']}/{n['c2']}.", "",
           "| Model | Session | y_true | p_yes (rationale run) | Stored test score | Generated label |", "|---|---|---|---|---|---|"]
    stored = {k: pd.read_parquet(p, columns=["row_id", "score"]).set_index("row_id")["score"] for k, p in STORED.items()}
    for r in mism.sort_values(["model", "p_yes"]).itertuples():
        md.append(f"| {r.model.upper()} | {r.session_id} | {r.y_true} | {r.p_yes:.3f} | {stored[r.model][r.row_id]:.3f} | {r.label_text} |")
    far = mism.drop(near.index)
    md += ["",
           f"- **Scoring is unaffected.** The lab p_yes matches the stored test scores within "
           f"{diffs['c1']:.3f} (C1) and {diffs['c2']:.3f} (C2) over all {n['c1']} + {n['c2']} rows.",
           f"- {len(near)} {'row sits' if len(near) == 1 else 'rows sit'} within p_yes [{lo}, {hi}]: "
           + ", ".join(f"{r.model.upper()} at {r.p_yes:.3f}" for r in near.itertuples())
           + ": fp16 noise between the scoring pass and greedy generation.",
           f"- {len(far)} rows disagree outside [{lo}, {hi}] ("
           + ", ".join(f"{m.upper()} {k}" for m, k in far.groupby("model").size().items())
           + f"; p_yes {far.p_yes.min():.3f}–{far.p_yes.max():.3f}). Scoring and generation use the same "
           "prompt string.",
           "- In all reported results the decision is the score (p_yes), never the generated label. The "
           "rationale sample is not regenerated whatever the diagnostic finds (decisions log 2026-11-22).", ""]
    md += penalty_section()
    (D / "rationales.md").write_text("\n".join(md))


def penalty_section():
    """Cause analysis (decisions log 2026-11-23) from results/step8/rationale_penalty_check.json."""
    pc = json.loads((D / "rationale_penalty_check.json").read_text())
    c1, c2 = pc["models"]["c1"], pc["models"]["c2"]
    no, yes = c2["groups"]["no_only"], c2["groups"]["yes_only"]
    fmt_ci = lambda m: f"{m['observed_flips']} (Poisson 95% {m['poisson_95'][0]:.1f}–{m['poisson_95'][1]:.1f})"
    return [
        "## Likely cause: repetition penalty from the model's generation config (consistency check, not a metric)", "",
        "`rationales.py` called `generate()` with `do_sample=False, num_beams=1, temperature=None, top_p=None, "
        "top_k=None` (and pad/eos ids); no seed is needed or set, since decoding is greedy. Everything else came from "
        "Qwen2.5-1.5B-Instruct's `generation_config.json` at the pinned revision (`do_sample: true, temperature 0.7, "
        f"top_p 0.8, top_k 20, repetition_penalty {pc['repetition_penalty']}`). The merged config is greedy, so "
        "sampling predicts **0** flips; but `repetition_penalty 1.1` is not overridden and stays active "
        "(`RepetitionPenaltyLogitsProcessor` is the only logits processor). It divides the positive logit of every "
        "token already in the input by 1.1, so the label flips deterministically when exactly one of \" yes\" "
        f"(id {pc['answer_ids']['yes']}) / \" no\" (id {pc['answer_ids']['no']}) appears in the prompt and the margin is small.", "",
        "| Model | Rows | Observed flips | Prompts with \" no\" only | with \" yes\" only | with neither | Flips on the penalised side |",
        "|---|---|---|---|---|---|---|",
        f"| C1 | {c1['rows']} | {fmt_ci(c1)} | {c1['groups'].get('no_only', {}).get('rows', 0)} | "
        f"{c1['groups'].get('yes_only', {}).get('rows', 0)} | {c1['groups'].get('neither', {}).get('rows', 0)} | "
        f"{c1['flips_explained_by_penalty']} |",
        f"| C2 | {c2['rows']} | {fmt_ci(c2)} | {no['rows']} | {yes['rows']} | {c2['groups']['neither']['rows']} | "
        f"{c2['flips_explained_by_penalty']} |", "",
        f"- C2, \" no\" in the prompt and p_yes < 0.5 ({no['score_no_rows']} rows): the {no['flips']} flips are exactly "
        f"the rows with margin ≥ {no['flipped_margin_min']:.2f}; every row at ≤ {no['kept_margin_max']:.2f} kept \"no\" "
        f"(one threshold separates them). That implies a \" no\" logit of {no['implied_no_logit_range'][0]:.1f}–"
        f"{no['implied_no_logit_range'][1]:.1f}, to be measured on 24 Nov.",
        f"- C2, \" yes\" in the prompt and p_yes > 0.5: {yes['score_yes_rows']} row, flipped to \"no\" (margin "
        f"{yes['flipped_margin_max']:.2f}).",
        "- No flip on a row where the penalty cannot act (C2: 0 of "
        f"{c2['groups']['neither']['rows']} neither-token rows). C1 prompts contain neither token; its one flip (p_yes "
        "0.501) is fp16 noise.",
        "- Observed C2 flips are inconsistent with greedy decoding without a penalty (expected 0, observed 10) and "
        "consistent with the penalty in every group. Confirmation: 24 Nov diagnostic step (f) (`src/slm/rationale_diag.py`).",
        "",
        "## Interpretation (decisions log 2026-11-23)", "",
        "- Padding (cause 3) is ruled out for the 17 Nov rows: they were generated at batch 1 without padding; "
        "diagnostic step (e) stays as a check only.",
        "- Cause accepted (2026-11-23): greedy decoding with repetition_penalty 1.1 inherited from Qwen's "
        "generation config, not plain greedy as logged on 15 Nov. The penalty acts on every generated token, not "
        "only the label: reasons cite feature names and values that are in the prompt, so all 686 v1 reason texts "
        "(C1 and C2) were generated under it. Scores are unaffected (score_texts does not use generate()).",
        "- Fix: the rationale sample is regenerated on 24 Nov with the logged protocol (greedy, repetition_penalty=1.0 "
        "passed explicitly) to `rationales_v2.parquet`; the 17 Nov file is kept for the bug log. No reported metric "
        "changes. The v2 file must show 0 label/score disagreements outside fp16 noise (|p_yes − 0.5| < 0.01).",
        "- Diagnostic step (f) still runs on 24 Nov to confirm the cause in the lab: (f-i) the 17 Nov arguments, 20 "
        "repeats, should reproduce the 17 Nov labels every time; (f-ii) the corrected arguments should agree with the "
        "scoring pass. If (f-i) varied across repeats, the labels would have been sampled instead.",
        "- S8-2 (examples, faithfulness check) runs on `rationales_v2.parquet`.", ""]


def write_note(ops, v, share, sess, compute):
    x, a, b, c, h = (ops.iloc[i] for i in range(5))
    per_hour = lambda rps: rps * 3600
    rows_per_session = compute["routing"]["rows"] / compute["routing"]["per_seed"][0]["sessions"]
    day_rows = WEEKDAY_SESSIONS * rows_per_session
    peak_hour_rows = WEEKDAY_SESSIONS                                   # every session active in one hour
    slm_routed_peak_s = share * peak_hour_rows * a.batch1_median_ms / 1e3
    xgb_peak_s = peak_hour_rows * x.batch1_median_ms / 1e3
    md = ["# Running the detector: what it needs (Step 8.1)", "",
          f"Measured on one lab PC ({v['gpu']}, 24 GB; {v['cpu']}), 17 Nov 2026. Plain-language summary of "
          "`operations.md`; every assumption is listed at the end.", "",
          "**Everything runs locally.** Both models run on the one machine: the language model (Qwen2.5-1.5B with "
          "a small LoRA adapter, about 3 GB on the GPU) and the gradient-boosted trees (XGBoost, CPU). Scoring "
          "calls no external service or API; the lab runs loaded the model weights from local copies.", "",
          "**Time per scored row** (one row = one user session at one hourly scoring point):", "",
          f"- XGBoost alone: {x.batch1_median_ms:.1f} ms.",
          f"- Language model, structured features: {a.batch1_median_ms:.0f} ms (95% of calls under {a.batch1_p95_ms:.0f} ms); "
          f"with the activity-text prompt: {b.batch1_median_ms:.0f} ms.",
          f"- Language model with a written reason: {c.batch1_median_ms:.0f} ms (95% under {c.batch1_p95_ms:.0f} ms).",
          f"- Hybrid (XGBoost on every row, the language model on the {share:.1%} of rows in the uncertain band): "
          f"{h.batch1_median_ms:.1f} ms per row on average; {sess:.1%} of sessions reach the language model at least once.", "",
          "**Rows per hour on one RTX 4500 Ada** (batched):", "",
          f"- Language model, structured: ~{round(per_hour(a.throughput_rows_per_s), -3):,.0f} rows/hour; text prompt: "
          f"~{round(per_hour(b.throughput_rows_per_s), -3):,.0f}.",
          f"- Hybrid: ~{round(per_hour(h.throughput_rows_per_s), -3):,.0f} rows/hour.",
          f"- XGBoost alone: over {per_hour(x.throughput_rows_per_s) / 1e9:.0f} billion rows/hour (not a constraint).", "",
          f"**GPU memory:** {a.peak_vram_allocated_gb:.1f}–{b.peak_vram_allocated_gb:.1f} GB at the peak (of 24 GB), so "
          "the card has room for other work.", "",
          f"**Hourly scoring for a 1,000-user organisation** (~{WEEKDAY_SESSIONS:.0f} sessions per weekday): at most "
          f"~{peak_hour_rows:.0f} rows in any hour and ~{day_rows:,.0f} rows per weekday. With the hybrid, the "
          f"busiest hour needs about {slm_routed_peak_s + xgb_peak_s:.0f} s of compute "
          f"({slm_routed_peak_s:.0f} s GPU, {xgb_peak_s:.0f} s CPU) one row at a time; even the language model "
          f"on every row needs {peak_hour_rows * a.batch1_median_ms / 1e3:.0f} s. One RTX 4500 Ada is busy well under "
          "1% of each hour; a smaller GPU with ~4 GB free would very likely do (not measured).", "",
          "## Assumptions", "",
          "1. Timings are from one machine, one session and one 1,000-row sample of test rows; other hardware, "
          "drivers or load will differ.",
          "2. Per-row latency is a single request (batch 1) with the model already loaded and warmed up; "
          "model loading (seconds) and feature extraction from the logs are not included.",
          f"3. Hybrid latency is an average: {share:.1%} (the test share routed to the language model) × the language "
          "model's median + XGBoost's median; the routed share may shift with the data (it was 19.0% on validation).",
          f"4. ~{WEEKDAY_SESSIONS:.0f} sessions per weekday and {rows_per_session:.2f} scoring points per session are the "
          "test-split averages (CERT r4.2, 1,000 users); real organisations differ.",
          "5. \"At most ~964 rows in an hour\" assumes every session of the day is active in the same hour (an upper bound).",
          "6. \"Fully local\": the scoring code calls no external service, and the lab runs were pointed at local "
          "copies of the weights (`--models-dir models`); network access was not blocked or monitored during the runs.",
          "7. 24 GB is the RTX 4500 Ada's specified memory, not a measurement.",
          "8. Written reasons are optional and cost ~15× the score alone; they are not needed for the risk score.", ""]
    (D / "operations_for_L.md").write_text("\n".join(md))


def run_log(gpu, xgb, rat, ops):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    lab = {"date": "2026-11-17", "git_commit": "6514afb (pulled on lab PC 17 Nov; not captured in outputs)", "seed": 1, "machine": LAB}
    rows = [
        lab | {"run_id": "latency_gpu", "step": "8.1 latency", "model": "slm c1/c2",
               "config": json.dumps(gpu["protocol"] | {"configs": CONFIGS}),
               "metrics": json.dumps({k: c.get("latency", {k2: c[k2] for k2 in ("rows_per_second",) if k2 in c})
                                      for k, c in gpu["configs"].items()}),
               "runtime_seconds": gpu["runtime_seconds"], "outputs": "results/step8/latency_gpu.json"},
        lab | {"run_id": "latency_xgb_lab", "step": "8.1 latency", "model": "xgb_struct_v3",
               "config": json.dumps({"method": xgb["method"], "iteration_range": xgb["iteration_range"]}),
               "metrics": json.dumps({"batch1": xgb["batch1"], "rows_per_second_1000": xgb["full_batch"]["rows_per_second"],
                                      "rows_per_second_100000": xgb["full_batch_tiled"]["rows_per_second"],
                                      "check_max_abs_diff": xgb["check_max_abs_diff"]}),
               "runtime_seconds": None, "outputs": "results/step8/latency_xgb_lab.json"},
        lab | {"run_id": "rationales", "step": "8.1 rationales", "model": "slm c1/c2",
               "config": json.dumps({"rows": len(rat), "max_new_tokens": 64, "decoding": "greedy", "batch": 1}),
               "metrics": json.dumps({"label_score_mismatch": rat.assign(m=(rat.label_text == "yes") != (rat.p_yes > 0.5))
                                      .groupby("model")["m"].sum().astype(int).to_dict()}),
               "runtime_seconds": float(rat["seconds"].sum()), "outputs": "results/step8/rationales.parquet"},
        {"date": datetime.date.today().isoformat(), "git_commit": commit, "seed": None, "machine": MAC,
         "run_id": "step8_report", "step": "8.1 report", "model": "-", "config": "{}",
         "metrics": json.dumps(dict(zip(ops["config"], ops["batch1_median_ms"].round(3)))),
         "runtime_seconds": None, "outputs": "results/step8/operations.md, operations.csv, rationales.md, operations_for_L.md"},
        {"date": datetime.date.today().isoformat(), "git_commit": commit, "seed": None, "machine": MAC,
         "run_id": "table3_compute_line", "step": "8.1 table3 compute line", "model": "hybrid",
         "config": json.dumps({"slm": "latency_gpu.json a_score_structured_b16", "xgb": "latency_xgb_lab.json full_batch_tiled"}),
         "metrics": "{}", "runtime_seconds": None, "outputs": "results/step7/table3/table3.md, compute.json"},
    ]
    cols = ["date", "git_commit", "run_id", "step", "model", "config", "seed", "machine", "metrics", "runtime_seconds", "outputs"]
    pd.DataFrame(rows)[cols].to_csv(D / "run.csv", index=False)


def main():
    checksum = verify_test_checksum()
    print(f"test_ids.csv SHA-256 {checksum}")
    gpu, xgb = load("latency_gpu.json"), load("latency_xgb_lab.json")
    check_protocol(gpu, xgb, checksum)
    runs = {}
    for version, (name, _) in FILES.items():
        if (D / name).exists():
            r = pd.read_parquet(D / name)
            runs[version] = (r, *check_rationales(r, version))
    if "v1" not in runs:
        raise SystemExit("results/step8/rationales.parquet (17 Nov) is required")
    rat, diffs, mism = runs["v1"]
    compute = json.loads((ROOT / "results/step7/table3/compute.json").read_text())
    if compute["compute"].get("machine") != LAB:
        raise SystemExit("Table 3 compute line is not the Step 8.1 single-machine line; run scripts.step7_table3 first")
    ops, v, share, sess = operations(gpu, xgb, compute)
    write_operations(ops, v, gpu)
    write_rationales(runs)
    write_note(ops, v, share, sess, compute)
    run_log(gpu, xgb, rat, ops)
    print(f"protocol OK; rationales match the bundle; stored-score max diff C1 {diffs['c1']:.4f}, C2 {diffs['c2']:.4f}; "
          f"mismatches {len(mism)}; wrote {D}/operations.md, operations.csv, rationales.md, operations_for_L.md, run.csv")


if __name__ == "__main__":
    main()
