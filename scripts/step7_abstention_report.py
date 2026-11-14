"""Step 7.5 abstention report from results/step7/abstention/{abstention.json, curves.csv}.

Writes, in results/step7/abstention/:
  abstention.md          operating point (10% deferred) and 0% reference: precision, recall and
                         FPR on auto-decided sessions, system recall (upper bound), deferred
                         positives, workload; mean ± std over seeds and 95% user-bootstrap CI
  abstention_main.png/pdf      C1 models + hybrid (main figure)
  abstention_appendix.png/pdf  C2 models (appendix)
and appends rows to results/step7/run.csv (step "7.5 abstention"). Reads no data or labels.
Run from the repo root: python -m scripts.step7_abstention_report
"""
import datetime
import json
import pathlib
import platform
import subprocess

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["abstention"]
D = pathlib.Path(CFG["out"])
RUN_LOG = pathlib.Path("results/step7/run.csv")
LABELS = {"xgb": "XGBoost (structured v3)", "slm": "SLM C1", "hybrid": "Hybrid",
          "xgb_text": "XGBoost text (C2)", "slm_text": "SLM text (C2)"}
COLOURS = {"xgb": "#0072B2", "slm": "#D55E00", "hybrid": "#009E73",      # Okabe-Ito
           "xgb_text": "#0072B2", "slm_text": "#D55E00"}
METRICS = [("precision_auto", "Precision (auto-decided)"), ("recall_auto", "Recall (auto-decided)"),
           ("fpr_auto", "FPR (auto-decided)"), ("system_recall_upper", "System recall (upper bound)"),
           ("deferred_positive_share", "Positives deferred")]
CAPTION = ("Precision and recall on auto-decided sessions become unstable once few alerts or positives "
           "remain among them (e.g. the SLM curves at high deferral shares, where their kept alerts run "
           "out); counts per point and seed are in curves.csv (alerts_auto, positives_auto).")
PANELS = [("precision_auto", "Precision on auto-decided sessions"),
          ("recall_auto", "Recall on auto-decided sessions"),
          ("system_recall_upper", "System recall, upper bound\n(analysts catch every deferred positive)")]


def cell(p, k):
    std = p["std"][k]
    return f"{p['mean'][k]:.3f}{'' if std != std else f' ± {std:.3f}'} [{p['ci'][k][0]:.3f}, {p['ci'][k][1]:.3f}]"


def table(result):
    op = f"{CFG['operating_point']:.2f}"
    first = next(iter(result["models"].values()))
    md = ["# Abstention (Step 7.5, test, session level)", "",
          f"Each model and seed defers the test sessions closest to its validation session-level "
          f"{CFG['fpr']:.0%}-FPR threshold, |logit(p) − logit(threshold)|, to analysts; alert = score ≥ "
          "threshold on the rest. Mean ± std over seeds 1–3 and 95% user-level bootstrap CI (1,000 "
          "stratified resamples, seed 0, the same draws as src/eval.py), from src/eval.py "
          "abstention(). System recall assumes analysts catch every deferred positive (an upper "
          "bound). Scores: calibrated logits (Platt session fit; hybrid: its Platt row-fit logit); "
          "any affine logit map gives the same thresholds and deferral order. Rules and rules + SLM "
          "are not calibrated (rank only) and are not included.", "",
          f"Test: {first['test_sessions']:,} sessions, {first['test_positive_sessions']} positive, "
          f"{first['test_days']} days. Deferring {CFG['operating_point']:.0%} = "
          f"{first['points'][op]['mean']['deferred_units']:,.0f} sessions = "
          f"{first['points'][op]['mean']['deferred_per_day']:.1f} sessions per test day for analysts.", ""]
    for figure, title in (("main", "Comparison 1 models and the hybrid (main)"), ("appendix", "Comparison 2 models (appendix)")):
        md += [f"## {title}", "", "| Model | Deferred | " + " | ".join(t for _, t in METRICS) + " |",
               "|---" * (len(METRICS) + 2) + "|"]
        for key, m in result["models"].items():
            if m["figure"] != figure:
                continue
            for q in ("0.00", op):
                p = m["points"][q]
                md.append(f"| {LABELS[key]} | {float(q):.0%}{' (operating point)' if q == op else ''} | "
                          + " | ".join(cell(p, k) for k, _ in METRICS) + " |")
        md.append("")
    work = sorted(f"{q:.2f}" for q in CFG["extra_points"]) + [op]
    md += ["## Workload view: low deferral shares", "",
           "Deferral shares an analyst team can absorb. Sessions per test day are the same for every "
           "model (a share of all test sessions). Mean ± std over seeds and 95% user-bootstrap CI.", "",
           "| Deferred | Sessions/day | Model | Positives deferred | System recall (upper bound) |",
           "|---|---|---|---|---|"]
    for q in work:
        for key, m in result["models"].items():
            p = m["points"][q]
            md.append(f"| {float(q):.0%} | {p['mean']['deferred_per_day']:.1f} | {LABELS[key]} | "
                      f"{cell(p, 'deferred_positive_share')} | {cell(p, 'system_recall_upper')} |")
    md += ["| P7 capacity share: to be filled after P7 (Step 7.6) | – | – | – | – |", ""]
    for name, pair in result.get("exploratory_paired", {}).items():
        md += [f"## Exploratory (not pre-declared): paired {LABELS[pair['a']]} − {LABELS[pair['b']]}", "",
               "Not pre-declared; reported as exploratory (decisions log 2026-11-14). Same user draws for "
               "both models (1,000 stratified resamples, seed 0); difference of seed-averaged metrics; "
               "significant = 95% CI excludes 0.", "",
               "| Deferred | Metric | " + LABELS[pair["a"]] + " | " + LABELS[pair["b"]] +
               " | Difference | 95% CI | Significant |", "|---|---|---|---|---|---|---|"]
        names = dict(METRICS)
        for q, res in pair["points"].items():
            for k, x in res.items():
                md.append(f"| {float(q):.0%} | {names[k]} | {x['a']:.3f} | {x['b']:.3f} | {x['difference']:+.4f} | "
                          f"[{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {'yes' if x['significant'] else 'no'} |")
        md.append("")
    md += ["Figures: `abstention_main.png/.pdf` (C1 + hybrid) and `abstention_appendix.png/.pdf` (C2); "
           "deferral share 0–50% in steps of 1%, mean over seeds (line) and each seed (thin), the "
           f"{CFG['operating_point']:.0%} operating point marked. " + CAPTION, ""]
    (D / "abstention.md").write_text("\n".join(md))


def figure(curves, figure_name, path):
    models = [m["key"] for m in CFG["models"] if m["figure"] == figure_name]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (k, title) in zip(axes, PANELS):
        for key in models:
            c = curves[curves["model"] == key]
            for _, g in c.groupby("seed"):
                ax.plot(g["share"] * 100, g[k], color=COLOURS[key], lw=0.8, alpha=0.35)
            mean = c.groupby("share")[k].mean()
            ax.plot(mean.index * 100, mean.to_numpy(), color=COLOURS[key], lw=2, label=LABELS[key])
        ax.axvline(CFG["operating_point"] * 100, color="#666666", lw=1, ls="--")
        ax.text(CFG["operating_point"] * 100 + 0.8, 0.98, f"{CFG['operating_point']:.0%} deferred",
                transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="#444444")
        ax.set_title(title, fontsize=10, wrap=True)
        ax.set_xlabel("Sessions deferred to analysts (%)")
        ax.set_xlim(0, CFG["shares"]["stop"] * 100)
        ax.set_ylim(0, None if k == "precision_auto" else 1)
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Abstention on test sessions (thresholds: validation 5% FPR; thick = mean, thin = seeds)")
    fig.text(0.5, -0.13, CAPTION, ha="center", va="top", fontsize=8, wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(D / f"{path}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_log(result):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for key, m in result["models"].items():
        for i, seed in enumerate(CFG["seeds"]):
            per = {q: p["per_seed"][i] for q, p in m["points"].items()}
            rows.append({"date": datetime.date.today().isoformat(), "git_commit": commit,
                         "run_id": f"abstention_{key}", "step": "7.5 abstention", "model": key,
                         "config": json.dumps({"threshold_split": CFG["threshold_split"], "fpr": CFG["fpr"],
                                               "threshold_logit": m["thresholds_logit_valid"][i],
                                               "folder": next(x["folder"] for x in CFG["models"] if x["key"] == key)}),
                         "seed": seed, "machine": f"Mac ({platform.machine()}, {platform.system()})",
                         "metrics": json.dumps({q: {k: v for k, v in p.items() if k not in ("seed", "share")}
                                                for q, p in per.items()}),
                         "runtime_seconds": result["runtime_seconds"], "outputs": str(D)})
    frame = pd.DataFrame(rows)
    if RUN_LOG.exists():
        old = pd.read_csv(RUN_LOG)
        frame = pd.concat([old[old["step"] != "7.5 abstention"], frame], ignore_index=True)
    frame.to_csv(RUN_LOG, index=False)


if __name__ == "__main__":
    result = json.loads((D / "abstention.json").read_text())
    curves = pd.read_csv(D / "curves.csv")
    table(result)
    figure(curves, "main", "abstention_main")
    figure(curves, "appendix", "abstention_appendix")
    run_log(result)
    print("wrote abstention.md, figures, run log")
