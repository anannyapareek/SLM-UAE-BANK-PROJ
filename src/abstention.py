"""Step 7.5 abstention (runbook 7.5; decisions log 2026-11-14), session level.

For each model and seed: the threshold is the seed's session-level 5%-FPR threshold on
VALIDATION (src/eval.py recall_at_fpr, max over scoring points; C2 logoff rows). On test, the
sessions closest to it, |logit(p) - logit(threshold)|, are deferred to analysts, for deferral
shares 0% to 50%; precision, recall and FPR are computed on the sessions left for automatic
decision (src/eval.py abstention). Also: system recall if analysts catch every deferred
positive (an upper bound) and the deferral workload in sessions per test day. At the operating
point (10% deferred, and 0% for reference) the 95% user-level bootstrap CI uses eval.py's
draws (SeedSequence(0), 1,000 stratified resamples). Rules and rules + SLM are not
calibrated (rank only), so they have no logit distance and are not included.

Run: python -m src.abstention
"""
import json
import pathlib
import time

import duckdb
import numpy as np
import pandas as pd
import yaml

from src import eval as ev

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["abstention"]


def eval_cfg():
    return {k: v for k, v in yaml.safe_load(ev.CONFIG.read_text()).items() if k != "baselines"}


def logits(model, seed, split, meta):
    """Calibrated logit per row, aligned to meta, labels checked."""
    folder = pathlib.Path(model["folder"].format(seed=seed))
    d = pd.read_parquet(folder / f"preds_{split}{model['suffix']}.parquet",
                        columns=["row_id", "y_true", "margin"]).set_index("row_id")
    if d.index.duplicated().any() or set(d.index) != set(meta["row_id"]):
        raise RuntimeError(f"{folder} {split}: rows do not match the split")
    d = d.reindex(meta["row_id"])
    if not np.array_equal(d["y_true"].to_numpy(int), meta["y"].to_numpy()):
        raise RuntimeError(f"{folder} {split}: labels differ")
    m = d["margin"].to_numpy(float)
    if model["kind"] == "logit":
        return m
    p = json.loads((folder / "params.json").read_text())["params"]
    return p["a"] * m + p["b"]


def units_for(con, model, split, cfg):
    meta = ev.load_meta(con, split, "final" if model["suffix"] == "_final" else "all")
    return ev.build_units(meta, [logits(model, s, split, meta) for s in CFG["seeds"]], "session", cfg)


def shares():
    s = CFG["shares"]
    return np.round(np.arange(s["start"], s["stop"] + s["step"] / 2, s["step"]), 10)


def point_summary(frame, boot, q, ci):
    rows = frame[np.isclose(frame["share"], q)]
    cols = ev.ABSTENTION_METRICS + ["deferred_per_day"]
    return {"per_seed": rows.drop(columns=["model", "figure"]).to_dict("records"),
            "mean": rows[cols].mean().to_dict(), "std": rows[cols].std(ddof=1).to_dict(),
            "ci": {k: ev.interval(v, ci)[0] for k, v in boot[q].items()},
            "boot_skipped": {k: ev.interval(v, ci)[1] for k, v in boot[q].items()}}


def seed_mean(values):
    return np.nan if all(np.isnan(v) for v in values) else float(np.nanmean(values))


def main():
    checksum = ev.verify_test_checksum()
    if CFG["threshold_split"] == "test":
        raise SystemExit("thresholds never come from test")
    started = time.perf_counter()
    cfg = eval_cfg()
    con = duckdb.connect(config={"threads": 1})
    out = pathlib.Path(CFG["out"])
    out.mkdir(parents=True, exist_ok=True)
    grid = shares()
    points = sorted({0.0, CFG["operating_point"], *CFG["extra_points"]})
    boot_seeds = np.random.SeedSequence(cfg["bootstrap"]["seed"]).spawn(cfg["bootstrap"]["resamples"])
    curves, summary, state = [], {}, {}
    for model in CFG["models"]:
        valid = units_for(con, model, CFG["threshold_split"], cfg)
        thresholds = [ev.recall_at_fpr(s, np.ones(len(s.score)), CFG["fpr"])[1] for s in valid["scored"]]
        test = units_for(con, model, "test", cfg)
        n_days = ev.counts(test)["n_days"]
        orders = [ev.deferral_order(s, t, test["key"]) for s, t in zip(test["scored"], thresholds)]
        state[model["key"]] = (test, thresholds, orders)
        ones = np.ones(len(test["units"]))
        for seed, s, t, o in zip(CFG["seeds"], test["scored"], thresholds, orders):
            for q in grid:
                m = ev.abstention(s, ones, t, o, q)
                curves.append({"model": model["key"], "figure": model["figure"], "seed": seed, "share": q,
                               "threshold_logit": t, **m, "deferred_per_day": m["deferred_units"] / n_days})
        strata = ev.user_strata(test)
        boot = {q: {k: [] for k in ev.ABSTENTION_METRICS} for q in points}
        for bs in boot_seeds:
            w = ev.draw_weights(np.random.default_rng(bs), strata, test)
            for q in points:
                per = [ev.abstention(s, w, t, o, q) for s, t, o in zip(test["scored"], thresholds, orders)]
                for k in ev.ABSTENTION_METRICS:
                    boot[q][k].append(seed_mean([p[k] for p in per]))
        frame = pd.DataFrame([c for c in curves if c["model"] == model["key"]])
        summary[model["key"]] = {
            "figure": model["figure"], "thresholds_logit_valid": thresholds,
            "test_sessions": int(len(test["units"])), "test_positive_sessions": int(test["units"]["y"].sum()),
            "test_days": n_days,
            "points": {f"{q:.2f}": point_summary(frame, boot, q, cfg["bootstrap"]["ci"]) for q in points}}
        op = summary[model["key"]]["points"][f"{CFG['operating_point']:.2f}"]["mean"]
        print(f"{time.strftime('%H:%M:%S')} {model['key']}: at {CFG['operating_point']:.0%} deferred "
              f"precision_auto {op['precision_auto']:.4f}, recall_auto {op['recall_auto']:.4f}, "
              f"system recall (upper) {op['system_recall_upper']:.4f}, {op['deferred_per_day']:.0f} sessions/day",
              flush=True)
    paired = {}
    for pair in CFG["exploratory_paired"]:
        a, b = pair["a"], pair["b"]
        (ua, ta, oa), (ub, tb, ob) = state[a], state[b]
        if not ua["units"]["session_id"].equals(ub["units"]["session_id"]):
            raise RuntimeError(f"{a} and {b} are not on the same sessions")
        strata, ones = ev.user_strata(ua), np.ones(len(ua["units"]))
        result = {}
        for q in pair["points"]:
            point = {n: [ev.abstention(s, ones, t, o, q) for s, t, o in zip(u["scored"], th, od)]
                     for n, (u, th, od) in ((a, state[a]), (b, state[b]))}
            diffs = {k: [] for k in pair["metrics"]}
            for bs in boot_seeds:                      # the same user draws for both models
                w = ev.draw_weights(np.random.default_rng(bs), strata, ua)
                pa = [ev.abstention(s, w, t, o, q) for s, t, o in zip(ua["scored"], ta, oa)]
                pb = [ev.abstention(s, w, t, o, q) for s, t, o in zip(ub["scored"], tb, ob)]
                for k in pair["metrics"]:
                    diffs[k].append(seed_mean([x[k] for x in pa]) - seed_mean([x[k] for x in pb]))
            result[f"{q:.2f}"] = {}
            for k in pair["metrics"]:
                va, vb = seed_mean([x[k] for x in point[a]]), seed_mean([x[k] for x in point[b]])
                ci, skipped = ev.interval(diffs[k], cfg["bootstrap"]["ci"])
                result[f"{q:.2f}"][k] = {"a": va, "b": vb, "difference": va - vb, "ci": ci,
                                         "significant": bool(ci[0] > 0 or ci[1] < 0), "boot_skipped": skipped}
        paired[f"{a}_vs_{b}"] = {"a": a, "b": b, "exploratory": True, "points": result}
        print(f"exploratory paired {a} - {b}: {json.dumps(result)}", flush=True)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False, float_format="%.8g")
    (out / "abstention.json").write_text(json.dumps(
        {"test_ids_sha256": checksum, "config": CFG, "models": summary, "exploratory_paired": paired,
         "runtime_seconds": round(time.perf_counter() - started, 1)}, indent=2, default=float) + "\n")
    print(f"wrote {out}/curves.csv and abstention.json in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
