"""Paper figures (Step 8.2) from results/ only: no data/, no models, no configs.

Reads, under --results (default results/):
  step7/calibration/<model>_{uncalibrated,sessionfit_temperature,sessionfit_platt}/metrics_test.json
  step7/abstention/{abstention.json, curves.csv}
  step7/review_budget/{review_budget.json, curves.csv}
Writes, under --out (default paper/figures/): one PDF (vector) and one PNG (300 dpi) per figure,
and captions.md. Every model has one colour and one marker in every figure (Okabe-Ito palette);
change the venue sizes and fonts in FIGURE_STYLE only.
Run from the repo root: python -m src.figures [--results DIR] [--out DIR]
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIGURE_STYLE = {
    "venue": "TODO",
    "single_column_in": 3.5,
    "double_column_in": 7.0,
    "font_family": "DejaVu Sans",
    "font_size": 8,              # ticks, legends, annotations
    "label_size": 9,             # axis labels and panel titles
    "dpi": 300,                  # PNG
    "line_width": 1.4,           # seed-mean curves
    "seed_line_width": 0.6,      # per-seed curves
    "seed_alpha": 0.35,
    "marker_size": 3.5,
    "grid_colour": "#dddddd",
    "reference_colour": "#444444",
}
# One colour and one marker per model, the same in every figure (Okabe-Ito; yellow left out for contrast).
MODELS = {
    "xgb": ("XGBoost (C1)", "#0072B2", "s"),
    "slm": ("SLM (C1)", "#D55E00", "o"),
    "hybrid": ("Hybrid (C1)", "#009E73", "D"),
    "rules": ("Rules", "#000000", "x"),
    "rules_slm": ("Rules + SLM", "#CC79A7", "P"),
    "xgb_text": ("XGBoost text (C2)", "#56B4E9", "^"),
    "slm_text": ("SLM text (C2)", "#E69F00", "v"),
}
CALIBRATION_RUNS = {"xgb": "xgb_struct", "slm": "slm_struct", "xgb_text": "xgb_text", "slm_text": "slm_text"}
METHODS = [("uncalibrated", "Uncalibrated", ":", "none"), ("sessionfit_temperature", "Temperature", "--", "none"),
           ("sessionfit_platt", "Platt", "-", "full")]
FLOOR = 1e-5
ABSTENTION_POINTS = [(0.03, "3%"), (0.10, "10%")]
ABSTENTION_PANELS = [("system_recall_upper", "System recall (upper bound)"),
                     ("precision_auto", "Precision, auto-decided"), ("recall_auto", "Recall, auto-decided")]


def style():
    s = FIGURE_STYLE
    plt.rcParams.update({
        "font.family": s["font_family"], "font.size": s["font_size"], "axes.labelsize": s["label_size"],
        "axes.titlesize": s["label_size"], "legend.fontsize": s["font_size"], "xtick.labelsize": s["font_size"],
        "ytick.labelsize": s["font_size"], "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": s["grid_colour"], "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(out / f"{name}.png", dpi=FIGURE_STYLE["dpi"], metadata={"Software": None})
    plt.close(fig)


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def reliability(R, out):
    s = FIGURE_STYLE
    fig, axes = plt.subplots(2, 2, figsize=(s["double_column_in"], s["double_column_in"] * 0.82),
                             constrained_layout=True)
    counts = {}
    for ax, (key, run) in zip(axes.ravel(), CALIBRATION_RUNS.items()):
        label, colour, marker = MODELS[key]
        ax.plot([FLOOR, 1], [FLOOR, 1], color=s["reference_colour"], lw=0.7, ls=(0, (1, 2)), label="Perfect calibration")
        for method, name, ls, fill in METHODS:
            m = read_json(R / "step7/calibration" / f"{run}_{method}" / "metrics_test.json")
            counts[key] = m["session"]["counts"]
            for i, seed in enumerate(m["session"]["per_seed"]):
                r = seed["reliability_equal_mass"]
                pts = [(c, a) for c, a, n in zip(r["mean_confidence"], r["accuracy"], r["count"]) if n and c and c > 0]
                ax.plot([p[0] for p in pts], [max(p[1], FLOOR) for p in pts], color=colour, ls=ls, marker=marker,
                        ms=s["marker_size"], mfc=colour if fill == "full" else "white", mec=colour,
                        lw=s["seed_line_width"] + 0.4, alpha=0.9, label=name if i == 0 else None)
        ax.set(xscale="log", yscale="log", xlim=(FLOOR, 1), ylim=(FLOOR * 0.8, 1), title=f"{label}, test sessions",
               xlabel="Mean predicted probability", ylabel="Observed positive rate")
        ax.axhline(FLOOR, color=s["reference_colour"], lw=0.5)
        ax.text(0.7, 1.3 * FLOOR, "floor 1e-5 = observed 0", fontsize=s["font_size"] - 1,
                color=s["reference_colour"], va="bottom", ha="right")
        ax.legend(loc="upper left")
    save(fig, out, "reliability_test")
    return counts


def abstention_figure(curves, keys, out, name):
    s = FIGURE_STYLE
    fig, axes = plt.subplots(1, 3, figsize=(s["double_column_in"], 2.7), constrained_layout=True)
    for ax, (metric, title) in zip(axes, ABSTENTION_PANELS):
        for key in keys:
            label, colour, marker = MODELS[key]
            c = curves[curves["model"] == key]
            for _, g in c.groupby("seed"):
                ax.plot(g["share"] * 100, g[metric], color=colour, lw=s["seed_line_width"], alpha=s["seed_alpha"])
            mean = c.groupby("share")[metric].mean()
            ax.plot(mean.index * 100, mean.to_numpy(), color=colour, lw=s["line_width"], marker=marker,
                    ms=s["marker_size"], markevery=5, label=label)
        for share, text in ABSTENTION_POINTS:
            ax.axvline(share * 100, color=s["reference_colour"], lw=0.8, ls="-" if share == 0.03 else "--")
            ax.text(share * 100, 1.01, text, transform=ax.get_xaxis_transform(), va="bottom",
                    ha="right" if share == 0.03 else "left", fontsize=s["font_size"] - 1, color=s["reference_colour"])
        ax.set(xlim=(0, 50), ylim=(0, None if metric == "precision_auto" else 1), xlabel="Sessions deferred (%)")
        ax.set_title(title, pad=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(keys))
    save(fig, out, name)


def review_panel(ax, curves, variant, keys, caps, cap):
    s = FIGURE_STYLE
    for key in keys:
        label, colour, marker = MODELS[key]
        c = curves[(curves["model"] == key) & (curves["variant"] == variant)]
        if c["seed"].notna().any():
            for _, g in c.groupby("seed"):
                ax.plot(g["k"], g["recall"], color=colour, lw=s["seed_line_width"], alpha=s["seed_alpha"])
        mean = c.groupby("k")["recall"].mean()
        ax.plot(mean.index, mean.to_numpy(), color=colour, lw=s["line_width"], marker=marker, ms=s["marker_size"],
                markevery=10, label=label)
    for k in caps:
        ax.axvline(k, color=s["reference_colour"], lw=0.9 if k == cap else 0.7, ls="-" if k == cap else "--")
        ax.text(k + 1, 0.99, f"{k}/day" + (" (capacity)" if k == cap else ""), transform=ax.get_xaxis_transform(),
                va="top", fontsize=s["font_size"] - 1, color=s["reference_colour"], rotation=90)
    ax.set(xlim=(0, 100), ylim=(0, 1), xlabel="Alerts per analyst-day (top k per day)", ylabel="Recall, test sessions")


def review_budget(R, out):
    s = FIGURE_STYLE
    rb = read_json(R / "step7/review_budget/review_budget.json")
    curves = pd.read_csv(R / "step7/review_budget/curves.csv")
    groups = {g: [m["key"] for m in rb["config"]["models"] if m["figure"] == g] for g in ("main", "c2")}
    caps, cap = rb["capacities"], rb["capacity"]
    for group, name in (("main", "review_budget_main"), ("c2", "review_budget_c2")):
        fig, ax = plt.subplots(figsize=(s["single_column_in"], 3.0), constrained_layout=True)
        review_panel(ax, curves, "calendar", groups[group], caps, cap)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3 if group == "main" else 2)
        save(fig, out, name)
    fig, axes = plt.subplots(1, 2, figsize=(s["double_column_in"], 2.7), constrained_layout=True)
    handles = []
    for ax, group, title in zip(axes, ("main", "c2"), ("C1 models, hybrid, rules", "C2 models")):
        review_panel(ax, curves, "weekday", groups[group], caps, cap)
        ax.set_title(f"{title}, weekday-only top-k")
        handles += list(zip(*ax.get_legend_handles_labels()))
    fig.legend(*zip(*handles), loc="outside lower center", ncol=4)
    save(fig, out, "review_budget_weekday")
    return rb, groups


def captions(out, cal_counts, ab, rb, groups):
    n = rb["test"]
    names = lambda keys: ", ".join(MODELS[k][0] for k in keys)
    test = (f"Test split (frozen), {n['n_units']:,} sessions, {n['n_pos']} positive, {n['n_users']} users, "
            f"{n['n_days']} days")
    day3 = ab["models"]["xgb"]["points"]["0.03"]["mean"]["deferred_per_day"]
    day10 = ab["models"]["xgb"]["points"]["0.10"]["mean"]["deferred_per_day"]
    c = cal_counts["xgb"]
    md = ["# Figure captions", "", f"Venue: {FIGURE_STYLE['venue']}. Colours and markers: one per model "
          "(Okabe-Ito) in every figure. Thin lines = each seed (1–3); thick = mean over seeds; no confidence "
          "intervals are drawn: 95% user-level bootstrap CIs (1,000 resamples, stratified, seed 0, src/eval.py) "
          "are given at the marked points in the paper tables.", "",
          "## reliability_test", "",
          f"Reliability diagrams on the test split, unit = session (C1: max over scoring points; C2: the logoff "
          f"row), n = {c['n_units']:,} sessions ({c['n_pos']} positive, {c['n_users']} users) per model and seed. "
          "One panel per model (C1 XGBoost, C1 SLM, C2 XGBoost text, C2 SLM text); uncalibrated (dotted, open "
          "markers), temperature scaling (dashed, open; pre-declared primary) and Platt scaling (solid, filled; "
          "sensitivity), both fitted on the calib split at the session level. 15 equal-mass bins, one line per "
          "seed; log–log axes; bins with no observed positive are drawn at the 1e-5 floor. No CIs drawn "
          "(calibration table: ECE, Brier, BSS with 95% CIs).", "",
          "## abstention_main / abstention_appendix", "",
          f"Abstention on the test split, unit = session, n = {n['n_units']:,} ({n['n_pos']} positive). Each "
          "model and seed defers the sessions closest to its validation 5%-FPR threshold (calibrated logit "
          "distance) to analysts and alerts on the rest. Main: " + names(["xgb", "slm", "hybrid"]) + "; appendix: "
          + names(["xgb_text", "slm_text"]) + f". Vertical lines: 3% deferred (≈ {day3:.1f} sessions/day, close "
          f"to the capacity of {rb['capacity']} alerts/day; solid line) and 10% (operating point, {day10:.1f} "
          "sessions/day; dashed). System recall assumes analysts catch every deferred positive (upper bound). "
          "Precision and recall on auto-decided sessions become unstable once few alerts or positives remain "
          "(e.g. the SLM beyond ~25% deferred, where its kept alerts run out; counts per point and seed in "
          "results/step7/abstention/curves.csv). "
          "Curves: 0–50% in steps of 1%; no CIs drawn (abstention table: 2%, 3%, 10% with 95% CIs).", "",
          "## review_budget_main / review_budget_c2", "",
          f"Recall vs alerts per analyst-day on the test split, unit = session, n = {n['n_units']:,} "
          f"({n['n_pos']} positive), calendar-day top-k over {n['n_days']} days (weekends included). Main: "
          + names(groups["main"]) + "; C2: " + names(groups["c2"]) + f". Solid line: capacity {rb['capacity']}/day "
          "(1 analyst); dashed: 10 and 50/day (sensitivity). Rules are deterministic (one run). No CIs drawn "
          "(review-budget table at 20/day with 95% CIs).", "",
          "## review_budget_weekday (appendix)", "",
          f"As review_budget_main and _c2, with weekday-only top-k ({n['weekday_days']} weekdays; weekends get no "
          f"alerts, so their {n['weekend_positive_sessions']} positive session counts as missed). {test}.", ""]
    (out / "captions.md").write_text("\n".join(md))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="paper/figures")
    args = p.parse_args(argv)
    R, out = pathlib.Path(args.results), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    cal_counts = reliability(R, out)
    ab = read_json(R / "step7/abstention/abstention.json")
    curves = pd.read_csv(R / "step7/abstention/curves.csv")
    abstention_figure(curves, [k for k, m in ab["models"].items() if m["figure"] == "main"], out, "abstention_main")
    abstention_figure(curves, [k for k, m in ab["models"].items() if m["figure"] == "appendix"], out,
                      "abstention_appendix")
    rb, groups = review_budget(R, out)
    captions(out, cal_counts, ab, rb, groups)
    print(f"wrote figures and captions to {out}")


if __name__ == "__main__":
    main()
