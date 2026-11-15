"""Step 7.6 review budget (runbook 7.6; decisions log 2026-11-15), session level, test.

Capacity = configs/eval.yaml alerts_per_day (1 analyst x 20 alerts per calendar day, D4), set
before any curve. Each test day alerts on its top k sessions by score (src/eval.py
top_k_per_day, max over scoring points; C2 logoff rows); recall and precision for k = 1..k_max,
per model and seed. Weekday-only variant (sensitivity): weekends get no alerts and their
positives count as missed. At k = 10, 20 (primary), 50: mean ± std over seeds and 95%
user-level bootstrap CI with eval.py's draws (SeedSequence(0), 1,000 stratified resamples).
Paired SLM-text − XGBoost-text at the capacity (pre-declared), the same draws for both.

Run: python -m src.review_budget          (writes curves.csv, review_budget.json)
     python -m src.review_budget report   (review_budget.md, figures, run.csv rows)
"""
import datetime
import json
import pathlib
import platform
import subprocess
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import yaml

from src import eval as ev

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["review_budget"]
OUT = pathlib.Path(CFG["out"])
RUN_LOG = pathlib.Path("results/step7/run.csv")
VARIANTS = ("calendar", "weekday")
LABELS = {"xgb": "XGBoost (structured v3)", "slm": "SLM C1", "hybrid": "Hybrid", "rules": "Rules",
          "rules_slm": "Rules + SLM", "xgb_text": "XGBoost text (C2)", "slm_text": "SLM text (C2)"}
COLOURS = {"xgb": "#0072B2", "slm": "#D55E00", "hybrid": "#009E73", "rules": "#E69F00",   # Okabe-Ito
           "rules_slm": "#CC79A7", "xgb_text": "#0072B2", "slm_text": "#D55E00"}


def eval_cfg():
    return {k: v for k, v in yaml.safe_load(ev.CONFIG.read_text()).items() if k != "baselines"}


def capacities(cfg):
    return sorted({cfg["alerts_per_day"], *CFG["sensitivity"]})


def paths(model):
    f = model["folder"]
    seeds = CFG["seeds"] if "{seed}" in f else [None]
    suffix = "_final" if model["rows"] == "final" else ""
    return [str(pathlib.Path(f.format(seed=s)) / f"preds_test{suffix}.parquet") for s in seeds]


def weekday(u):
    """Bool per unit: its day (eval.py's unit day, days since epoch) is Monday-Friday."""
    return ((u["day"] + 3) % 7) < 5           # 1970-01-01 was a Thursday


def metrics(u, w, k, variant):
    open_unit = weekday(u) if variant == "weekday" else None
    per = [ev.top_k_per_day(s, w, k, open_unit) for s in u["scored"]]
    return {"precision": float(np.mean([p[0] for p in per])), "recall": float(np.mean([p[1] for p in per]))}, per


def main():
    checksum = ev.verify_test_checksum()
    started = time.perf_counter()
    cfg = eval_cfg()
    caps, ci = capacities(cfg), cfg["bootstrap"]["ci"]
    con = duckdb.connect(config={"threads": 1})
    metas = {r: ev.load_meta(con, "test", r) for r in ("all", "final")}
    units, curves, summary = {}, [], {}
    for model in CFG["models"]:
        meta = metas[model["rows"]]
        u = ev.build_units(meta, ev.load_preds(paths(model), meta), "session", cfg)
        units[model["key"]] = u
        seeds = CFG["seeds"] if len(u["scored"]) > 1 else [None]
        ones = np.ones(len(u["units"]))
        for variant in VARIANTS:
            open_unit = weekday(u) if variant == "weekday" else None
            for k in range(1, CFG["k_max"] + 1):
                for seed, s in zip(seeds, u["scored"]):
                    p, r = ev.top_k_per_day(s, ones, k, open_unit)
                    curves.append({"model": model["key"], "figure": model["figure"], "variant": variant,
                                   "seed": seed, "k": k, "precision": p, "recall": r})
        print(f"{time.strftime('%H:%M:%S')} {model['key']}: curves done", flush=True)
    keys = [m["key"] for m in CFG["models"]]
    ref = units[keys[0]]
    for key in keys:
        if not units[key]["units"]["session_id"].equals(ref["units"]["session_id"]):
            raise RuntimeError(f"{key} is not on the same test sessions as {keys[0]}")
    strata = ev.user_strata(ref)
    boot = {(key, v, k): {"precision": [], "recall": []} for key in keys for v in VARIANTS for k in caps}
    pairs = {(p["a"], p["b"]): {(v, k): {"precision": [], "recall": []} for v in VARIANTS for k in caps}
             for p in CFG["paired"]}
    for i, bs in enumerate(np.random.SeedSequence(cfg["bootstrap"]["seed"]).spawn(cfg["bootstrap"]["resamples"])):
        w = ev.draw_weights(np.random.default_rng(bs), strata, ref)       # one draw for every model
        point = {(key, v, k): metrics(units[key], w, k, v)[0] for key in keys for v in VARIANTS for k in caps}
        for x, m in point.items():
            for name in ("precision", "recall"):
                boot[x][name].append(m[name])
        for (a, b), store in pairs.items():
            for (v, k), d in store.items():
                for name in ("precision", "recall"):
                    d[name].append(point[(a, v, k)][name] - point[(b, v, k)][name])
        if (i + 1) % 100 == 0:
            print(f"{time.strftime('%H:%M:%S')} bootstrap {i + 1}", flush=True)
    frame = pd.DataFrame(curves)
    for key in keys:
        u, ones = units[key], np.ones(len(units[key]["units"]))
        summary[key] = {"figure": next(m["figure"] for m in CFG["models"] if m["key"] == key),
                        "preds": paths(next(m for m in CFG["models"] if m["key"] == key)),
                        "seeded": len(u["scored"]) > 1, "points": {}}
        for v in VARIANTS:
            for k in caps:
                mean, per = metrics(u, ones, k, v)
                entry = {"per_seed": [{"precision": p, "recall": r} for p, r in per], "mean": mean}
                if len(per) > 1:
                    entry["std"] = {n: float(np.std([p[j] for p in per], ddof=1)) for j, n in enumerate(("precision", "recall"))}
                entry["ci"] = {n: ev.interval(boot[(key, v, k)][n], ci)[0] for n in ("precision", "recall")}
                entry["boot_skipped"] = {n: ev.interval(boot[(key, v, k)][n], ci)[1] for n in ("precision", "recall")}
                summary[key]["points"][f"{v}_{k}"] = entry
    paired = {}
    for (a, b), store in pairs.items():
        res = {}
        for (v, k), d in store.items():
            res[f"{v}_{k}"] = {}
            for n in ("precision", "recall"):
                va, vb = summary[a]["points"][f"{v}_{k}"]["mean"][n], summary[b]["points"][f"{v}_{k}"]["mean"][n]
                lo_hi, skipped = ev.interval(d[n], ci)
                res[f"{v}_{k}"][n] = {"a": va, "b": vb, "difference": va - vb, "ci": lo_hi,
                                      "significant": bool(lo_hi[0] > 0 or lo_hi[1] < 0), "boot_skipped": skipped,
                                      "pre_declared": bool(v == "calendar" and k == cfg["alerts_per_day"])}
        paired[f"{a}_vs_{b}"] = {"a": a, "b": b, "points": res}
    c = ev.counts(ref)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "curves.csv", index=False, float_format="%.8g")
    (OUT / "review_budget.json").write_text(json.dumps(
        {"test_ids_sha256": checksum, "config": CFG, "capacity": cfg["alerts_per_day"], "capacities": caps,
         "test": c | {"weekday_days": int(len(np.unique(ref["day"][weekday(ref)]))),
                      "weekend_positive_sessions": int(ref["units"]["y"].to_numpy()[~weekday(ref)].sum())},
         "models": summary, "paired": paired, "runtime_seconds": round(time.perf_counter() - started, 1)},
        indent=2, default=float) + "\n")
    print(f"wrote {OUT}/curves.csv and review_budget.json in {time.perf_counter() - started:.0f} s")


# Report

def cell(p, n):
    std = p.get("std", {}).get(n)
    return f"{p['mean'][n]:.3f}{'' if std is None else f' ± {std:.3f}'} [{p['ci'][n][0]:.3f}, {p['ci'][n][1]:.3f}]"


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads((OUT / "review_budget.json").read_text())
    curves = pd.read_csv(OUT / "curves.csv")
    cap, caps, t = r["capacity"], r["capacities"], r["test"]
    md = ["# Review budget (Step 7.6, test, session level)", "",
          f"Capacity: **1 analyst × {cap} alerts per calendar day** (configs/eval.yaml alerts_per_day; "
          "decisions log 2026-11-15), set before any curve; 10 and 50 per day are sensitivity values. "
          "Each test day alerts on its top k sessions by score (src/eval.py top_k_per_day; session "
          "score = max over scoring points, C2 = logoff point; ties by eval.py's hash key). Mean ± std "
          "over seeds 1–3 (rules: one deterministic run) and 95% user-level bootstrap CI (1,000 "
          "stratified resamples, seed 0, the same draws as src/eval.py and for every model).", "",
          f"Test: {t['n_units']:,} sessions, {t['n_pos']} positive, {t['n_days']} days "
          f"({t['weekday_days']} weekdays). Calendar-day top-k (primary, D4) alerts on weekends too; "
          f"weekday-only (sensitivity) gives weekends no alerts, so their {t['weekend_positive_sessions']} "
          "positive session(s) count as missed. Precision@k ceilings are in capacity_facts.md.", ""]
    for v, title in (("calendar", "Calendar-day top-k (primary)"), ("weekday", "Weekday-only top-k (sensitivity)")):
        md += [f"## {title}", "", "| Model | " + " | ".join(f"Recall @ {k}/day" for k in caps) +
               f" | Precision @ {cap}/day |", "|---" * (len(caps) + 2) + "|"]
        for key, m in r["models"].items():
            md.append(f"| {LABELS[key]} | " + " | ".join(
                (f"**{cell(m['points'][f'{v}_{k}'], 'recall')}**" if k == cap else cell(m["points"][f"{v}_{k}"], "recall"))
                for k in caps) + f" | {cell(m['points'][f'{v}_{cap}'], 'precision')} |")
        md.append("")
    for name, pair in r["paired"].items():
        md += [f"## Paired {LABELS[pair['a']]} − {LABELS[pair['b']]}", "",
               f"Pre-declared (runbook 7.6 project decision) at the chosen capacity, calendar-day top-k, "
               f"{cap}/day; the other rows are sensitivity. Same user draws for both models; difference of "
               "seed-averaged metrics; significant = 95% CI excludes 0.", "",
               f"| Top-k | Alerts/day | Metric | {LABELS[pair['a']]} | {LABELS[pair['b']]} | Difference | 95% CI | Significant |",
               "|---|---|---|---|---|---|---|---|"]
        order = sorted(pair["points"], key=lambda q: (q.split("_")[0] != "calendar", int(q.split("_")[1]) != cap, int(q.split("_")[1])))
        for q in order:
            v, k = q.split("_")
            for n, x in pair["points"][q].items():
                tag = " (pre-declared)" if x["pre_declared"] else ""
                md.append(f"| {v}{tag} | {k} | {n} | {x['a']:.3f} | {x['b']:.3f} | {x['difference']:+.4f} | "
                          f"[{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {'yes' if x['significant'] else 'no'} |")
        md.append("")
    md += ["Figures: `review_budget_main.png/.pdf` (C1 models, hybrid, rules, rules + SLM), "
           "`review_budget_c2.png/.pdf` (C2 pair), `review_budget_weekday.png/.pdf` (weekday-only, "
           f"both groups). Recall vs alerts per analyst-day (k = 1–{CFG['k_max']}); thick = mean over "
           f"seeds, thin = each seed; solid line = capacity {cap}/day, dashed = 10 and 50/day. "
           "Per-k values for every seed are in curves.csv.", ""]
    (OUT / "review_budget.md").write_text("\n".join(md))

    def panel(ax, variant, figure, title):
        for m in CFG["models"]:
            if m["figure"] != figure:
                continue
            c = curves[(curves["model"] == m["key"]) & (curves["variant"] == variant)]
            if c["seed"].notna().any():
                for _, g in c.groupby("seed"):
                    ax.plot(g["k"], g["recall"], color=COLOURS[m["key"]], lw=0.8, alpha=0.35)
            mean = c.groupby("k")["recall"].mean()
            ax.plot(mean.index, mean.to_numpy(), color=COLOURS[m["key"]], lw=2, label=LABELS[m["key"]],
                    ls=(0, (4, 2)) if m["key"] == "rules_slm" else "-")
        for k in caps:
            ax.axvline(k, color="#444444" if k == cap else "#888888", lw=1.2 if k == cap else 0.9,
                       ls="-" if k == cap else "--")
            ax.text(k + 1, 0.98, f"{k}/day" + (" (capacity)" if k == cap else ""), transform=ax.get_xaxis_transform(),
                    va="top", fontsize=8, color="#444444")
        ax.set_xlim(0, CFG["k_max"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Alerts per analyst-day (top k sessions per day, 1 analyst)")
        ax.set_ylabel("Recall (test positive sessions)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for figure, name, title in (("main", "review_budget_main", "Comparison 1 models, hybrid and rules"),
                                ("c2", "review_budget_c2", "Comparison 2: XGBoost text vs SLM text")):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        panel(ax, "calendar", figure, title + " (calendar-day top-k)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for ax, (figure, title) in zip(axes, (("main", "C1 models, hybrid and rules"), ("c2", "C2: XGBoost text vs SLM text"))):
        panel(ax, "weekday", figure, title + " (weekday-only top-k)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "review_budget_weekday")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for key, m in r["models"].items():
        seeds = CFG["seeds"] if m["seeded"] else [None]
        for i, seed in enumerate(seeds):
            rows.append({"date": datetime.date.today().isoformat(), "git_commit": commit,
                         "run_id": f"review_budget_{key}", "step": "7.6 review_budget", "model": key,
                         "config": json.dumps({"capacity": cap, "capacities": caps, "preds": m["preds"][i]}),
                         "seed": seed, "machine": f"Mac ({platform.machine()}, {platform.system()})",
                         "metrics": json.dumps({q: p["per_seed"][i] for q, p in m["points"].items()}),
                         "runtime_seconds": r["runtime_seconds"], "outputs": str(OUT)})
    frame = pd.DataFrame(rows)
    if RUN_LOG.exists():
        old = pd.read_csv(RUN_LOG)
        frame = pd.concat([old[old["step"] != "7.6 review_budget"], frame], ignore_index=True)
    frame.to_csv(RUN_LOG, index=False)
    print("\n".join(md))


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
