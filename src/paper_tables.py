"""Paper tables (Step 8.2) from results/ only: no data/, no models, no configs.

Every number is copied from a results file (src/eval.py outputs and the Step 7/8 reports); nothing
is recomputed. Each cell records its sources (file relative to --results, key path, value) in
provenance.json, which tests/test_paper_outputs.py re-reads and checks against the source files.
Format as in the summaries: mean ± std over seeds [95% user-level bootstrap CI]; deterministic
models (rules) have no std.

Writes, under --out (default paper/tables/): <name>.md and <name>.tex (booktabs) for table1,
table2, text_gain, table3, calibration, abstention, review_budget, fairness_c1, fairness_c2,
operations; and provenance.json.
Run from the repo root: python -m src.paper_tables [--results DIR] [--out DIR]
"""
import argparse
import csv
import json
import math
import pathlib

METRICS = [("pr_auc", "PR-AUC"), ("recall_at_fpr_0.01", "Recall @ 1% FPR"),
           ("recall_at_fpr_0.05", "Recall @ 5% FPR"), ("precision_at_k", "Precision @ k (20/day)")]
UNITS = [("max", "session", "Session, max over scoring points (primary)"),
         ("final", "session", "Session, final point (secondary)"),
         ("max", "row", "Row (secondary)")]
LABELS = {"xgb": "XGBoost (C1)", "slm": "SLM (C1)", "hybrid": "Hybrid (C1)", "rules": "Rules",
          "rules_slm": "Rules + SLM", "xgb_text": "XGBoost text (C2)", "slm_text": "SLM text (C2)"}
CI_NOTE = ("Mean ± std over seeds 1–3 (rules: deterministic, one run, no std) and 95% user-level bootstrap CI "
           "(1,000 stratified resamples, seed 0), src/eval.py. Test split (frozen).")


class Source:
    """Reads results files once and records what each cell took from them."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.cache = {}

    def load(self, file):
        if file not in self.cache:
            path = self.root / file
            if file.endswith(".json"):
                self.cache[file] = json.loads(path.read_text())
            else:
                with open(path, newline="") as f:
                    self.cache[file] = list(csv.DictReader(f))
        return self.cache[file]

    def get(self, file, path):
        x = self.load(file)
        for k in path:
            if isinstance(x, list) and isinstance(k, dict):         # csv row selected by key fields
                x = next(r for r in x if all(r[c] == v for c, v in k.items()))
            else:
                x = x[k]
        if isinstance(x, str):
            x = None if x == "" else float(x)
        return x

    def try_get(self, file, path):
        try:
            return self.get(file, path)
        except (KeyError, StopIteration):
            return None


def render(kind, values, dp):
    """Text of a cell from its source values (the test calls this too)."""
    f = lambda v: f"{v:.{dp}f}"
    if kind == "msci":                    # mean, std, ci_low, ci_high
        mean, std, lo, hi = values
        return f"{f(mean)}{'' if std is None else f' ± {f(std)}'} [{f(lo)}, {f(hi)}]"
    if kind == "ms":                      # mean, std
        mean, std = values
        return f(mean) + ("" if std is None else f" ± {f(std)}")
    if kind == "diff":                    # difference, ci_low, ci_high, significant
        d, lo, hi, sig = values
        return f"{d:+.{dp}f} [{lo:+.{dp}f}, {hi:+.{dp}f}]{' *' if sig else ''}"
    if kind == "pct_ms":                  # share mean, std as percentages
        mean, std = values
        return f"{mean * 100:.{dp}f}% ± {std * 100:.{dp}f}%"
    if kind == "num":
        (v,) = values
        return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{dp}f}"
    if kind == "int":
        (v,) = values
        return f"{int(v):,}"
    raise ValueError(kind)


class Table:
    def __init__(self, name, title, columns, note):
        self.name, self.title, self.columns, self.note = name, title, columns, note
        self.rows = []                 # ("section", text) or ("row", label, [cells])

    def section(self, text):
        self.rows.append(("section", text))

    def row(self, label, cells):
        self.rows.append(("row", label, cells))


class Builder:
    def __init__(self, src):
        self.src = src
        self.provenance = []
        self.table = None

    def cell(self, row, col, kind, dp, refs):
        """refs: list of (file, path); values looked up and recorded."""
        values = [self.src.try_get(f, p) for f, p in refs]
        text = render(kind, values, dp)
        self.provenance.append({"table": self.table.name, "row": row, "column": col, "kind": kind, "dp": dp,
                                "text": text, "sources": [{"file": f, "path": p, "value": v}
                                                          for (f, p), v in zip(refs, values)]})
        return text

    def stat(self, row, col, file, base, metric, dp=3, ci=True):
        refs = [(file, base + ["mean", metric]), (file, base + ["std", metric])]
        if ci:
            refs += [(file, base + ["ci", metric, 0]), (file, base + ["ci", metric, 1])]
        return self.cell(row, col, "msci" if ci else "ms", dp, refs)

    def diff(self, row, col, file, base, dp=3):
        return self.cell(row, col, "diff", dp, [(file, base + ["difference"]), (file, base + ["ci", 0]),
                                                (file, base + ["ci", 1]), (file, base + ["significant"])])


def metric_table(b, name, title, folder, models, paired, extra_note):
    t = b.table = Table(name, title, ["Model"] + [m for _, m in METRICS], CI_NOTE + " " + extra_note)
    for agg, unit, heading in UNITS:
        t.section(heading)
        for key in models:
            file = f"{folder}/{key}_test_{agg}.json"
            t.row(LABELS[key], [b.stat(f"{LABELS[key]} / {heading}", m, file, [unit], k) for k, m in METRICS])
        for a, bb, pfile in paired:
            label = f"{LABELS[a]} − {LABELS[bb]}"
            file = pfile.format(agg=agg)
            t.row(label, [b.diff(f"{label} / {heading}", m, file, [unit, k]) for k, m in METRICS])
    return t


def table2(b):
    t = b.table = Table("table2", "Table 2: Comparison 2 on the test split (text added; logoff rows, session = logoff row)",
                        ["Model"] + [m for _, m in METRICS], CI_NOTE + " Paired rows: SLM text − XGBoost text, "
                        "same user draws; * = 95% CI excludes 0.")
    for key in ("slm_text", "xgb_text", "rules"):
        file = f"comparison2/{key}_test_final.json"
        t.row(LABELS[key] if key != "rules" else "Rules (reference)",
              [b.stat(LABELS[key], m, file, ["session"], k) for k, m in METRICS])
    label = "SLM text − XGBoost text"
    t.row(label, [b.diff(label, m, "comparison2/paired_slm_text_vs_xgb_text.json", ["session", k]) for k, m in METRICS])
    return t


def text_gain(b):
    t = b.table = Table("text_gain", "Text gain on the test split (logoff rows): C2 text − C1 structured, and DiD",
                        ["Comparison"] + [m for _, m in METRICS],
                        "Change = text (C2) − structured (C1, final point) for the same model; difference-in-"
                        "differences = SLM gain − XGBoost gain. Paired user bootstrap, the same 1,000 user draws "
                        "(seed 0); 95% CI; * = CI excludes 0. src/text_gain.py reuses src/eval.py.")
    for pair, label in (("slm_text_vs_slm_struct", "SLM: text − structured"),
                        ("xgb_text_vs_xgb_struct", "XGBoost: text − structured")):
        t.row(label, [b.diff(label, m, f"comparison2/paired_{pair}.json", ["session", k]) for k, m in METRICS])
    label = "Difference-in-differences (SLM gain − XGBoost gain)"
    t.row(label, [b.cell(label, m, "diff", 3, [("comparison2/text_gain_did.json", ["did", k, "difference_in_differences"]),
                                               ("comparison2/text_gain_did.json", ["did", k, "ci", 0]),
                                               ("comparison2/text_gain_did.json", ["did", k, "ci", 1]),
                                               ("comparison2/text_gain_did.json", ["did", k, "significant"])])
                  for k, m in METRICS])
    return t


def table3(b):
    t = metric_table(b, "table3", "Table 3: Comparison 3 (hybrid) on the test split", "step7/table3/eval",
                     ["xgb", "slm", "hybrid", "rules", "rules_slm"],
                     [("hybrid", "xgb", "step7/table3/eval/paired_hybrid_vs_xgb_test_{agg}.json")],
                     "XGBoost and SLM: the Platt row-fit files the hybrid is built from (ranking metrics equal "
                     "Table 1's). Hybrid − XGBoost is the pre-declared key test (paired, same user draws; * = 95% CI "
                     "excludes 0). Band chosen on validation, not re-selected after test.")
    t.section("Hybrid routing (test)")
    t.row("Rows sent to the SLM", [b.cell("routing", "rows", "pct_ms", 1, [("step7/table3/compute.json", ["routing", "row_share", 0]),
                                                                          ("step7/table3/compute.json", ["routing", "row_share", 1])])]
          + [""] * (len(METRICS) - 1))
    t.row("Sessions with ≥ 1 SLM row", [b.cell("routing", "sessions", "pct_ms", 1,
                                               [("step7/table3/compute.json", ["routing", "session_share", 0]),
                                                ("step7/table3/compute.json", ["routing", "session_share", 1])])]
          + [""] * (len(METRICS) - 1))
    return t


def calibration(b):
    cols = [("ece_equal_mass", "ECE (equal-mass, 15 bins)"), ("brier", "Brier"), ("bss_reference", "BSS (calib base rate)")]
    t = b.table = Table("calibration", "Calibration on the test split (session level)", ["Model / method"] + [c for _, c in cols],
                        CI_NOTE + " Unit = session (C1: max over scoring points; C2: logoff row). Temperature (pre-"
                        "declared primary) and Platt (sensitivity) fitted on calib at the session level. BSS = 1 − "
                        "Brier / (p(1 − p)), p = calib session positive rate.")
    methods = [("uncalibrated", "Uncalibrated"), ("sessionfit_temperature", "Temperature (primary)"),
               ("sessionfit_platt", "Platt (sensitivity)")]
    for run, key in (("xgb_struct", "xgb"), ("slm_struct", "slm"), ("xgb_text", "xgb_text"), ("slm_text", "slm_text")):
        t.section(LABELS[key])
        for method, mlabel in methods:
            file = f"step7/calibration/{run}_{method}/metrics_test.json"
            t.row(mlabel, [b.stat(f"{LABELS[key]} / {mlabel}", c, file, ["session"], k, dp=4) for k, c in cols])
    return t


def abstention(b):
    cols = [("system_recall_upper", "System recall (upper bound)"), ("precision_auto", "Precision (auto-decided)"),
            ("recall_auto", "Recall (auto-decided)"), ("deferred_positive_share", "Positives deferred")]
    t = b.table = Table("abstention", "Abstention on the test split (session level)",
                        ["Deferred / model", "Sessions/day"] + [c for _, c in cols],
                        CI_NOTE + " Each model and seed defers the sessions closest to its validation 5%-FPR threshold "
                        "(calibrated logit). 3% ≈ the 20 alerts/day capacity; 10% = operating point. System recall "
                        "assumes analysts catch every deferred positive (upper bound).")
    file = "step7/abstention/abstention.json"
    for q in ("0.02", "0.03", "0.10"):
        t.section(f"{float(q):.0%} deferred")
        for key in ("xgb", "slm", "hybrid", "xgb_text", "slm_text"):
            base = ["models", key, "points", q]
            t.row(LABELS[key], [b.cell(f"{q} / {LABELS[key]}", "Sessions/day", "num", 1, [(file, base + ["mean", "deferred_per_day"])])]
                  + [b.stat(f"{q} / {LABELS[key]}", c, file, base, k) for k, c in cols])
    return t


def review_budget(b):
    t = b.table = Table("review_budget", "Review budget at 20 alerts per analyst-day (test, session level)",
                        ["Model", "Precision @ 20/day", "Recall @ 20/day"],
                        CI_NOTE + " Top 20 sessions per day, 1 analyst. Calendar-day top-k (primary, weekends "
                        "included) and weekday-only (sensitivity).")
    file = "step7/review_budget/review_budget.json"
    for variant, heading in (("calendar_20", "Calendar-day top-k (primary)"), ("weekday_20", "Weekday-only (sensitivity)")):
        t.section(heading)
        for key in ("xgb", "slm", "hybrid", "rules", "rules_slm", "xgb_text", "slm_text"):
            base = ["models", key, "points", variant]
            t.row(LABELS[key], [b.stat(f"{heading} / {LABELS[key]}", m, file, base, k) for k, m in
                                (("precision", "Precision @ 20/day"), ("recall", "Recall @ 20/day"))])
    return t


def fairness(b, name, title, pairs):
    (ka, fa), (kb, fb) = pairs
    t = b.table = Table(name, title, ["Attribute / group", "Sessions", "Positives", f"FPR {LABELS[ka]}", f"FPR {LABELS[kb]}",
                                      f"Recall {LABELS[ka]}", f"Recall {LABELS[kb]}"],
                        "Test split, session level; alert at each seed's validation 5%-FPR threshold; mean ± std over "
                        "3 seeds. Groups with at least 5 positive sessions (recall defined), as in the fairness reports; "
                        "the full lists are in results/comparison*/fairness_*_test.csv.")
    rows = sorted(b.src.load(fa), key=lambda r: r["attribute"] != "all")        # "all" first, file order otherwise
    for r in rows:
        if r["recall_mean"] == "":
            continue
        key = {"attribute": r["attribute"], "group": r["group"]}
        label = f"{r['attribute']} / {r['group']}"
        t.row(label, [b.cell(label, "Sessions", "int", 0, [(fa, [key, "sessions"])]),
                      b.cell(label, "Positives", "int", 0, [(fa, [key, "positives"])]),
                      b.cell(label, "FPR a", "ms", 4, [(fa, [key, "fpr_mean"]), (fa, [key, "fpr_std"])]),
                      b.cell(label, "FPR b", "ms", 4, [(fb, [key, "fpr_mean"]), (fb, [key, "fpr_std"])]),
                      b.cell(label, "Recall a", "ms", 3, [(fa, [key, "recall_mean"]), (fa, [key, "recall_std"])]),
                      b.cell(label, "Recall b", "ms", 3, [(fb, [key, "recall_mean"]), (fb, [key, "recall_std"])])])
    return t


def operations(b):
    file = "step8/operations.csv"
    t = b.table = Table("operations", "Operations (Step 8.1, one lab PC: RTX 4500 Ada, Windows 11)",
                        ["Configuration", "Batch-1 median (ms)", "Batch-1 p95 (ms)", "Throughput (rows/s)",
                         "Peak VRAM (GB)", "RSS (GB)"],
                        "Timings, not metrics (results/step8/operations.md): batch 1 after 20 warm-up calls, device "
                        "synchronised per call, 1,000 test logoff rows (rationale: first 150); SLM fp16 with the merged "
                        "seed-1 adapter; throughput at batch 16 (SLM) or 100,000 rows (XGBoost). Hybrid latency = 21.5% "
                        "× SLM + XGBoost batch-1 medians (expected value); hybrid throughput = Table 3 compute line.")
    for r in b.src.load(file):
        key = {"config": r["config"]}
        label = r["config"]
        tp = float(r["throughput_rows_per_s"]) if r["throughput_rows_per_s"] else None
        t.row(label, [b.cell(label, "median", "num", 2, [(file, [key, "batch1_median_ms"])]),
                      b.cell(label, "p95", "num", 2, [(file, [key, "batch1_p95_ms"])]),
                      b.cell(label, "throughput", "num", 1 if tp is not None and tp < 1e4 else 0,
                             [(file, [key, "throughput_rows_per_s"])]),
                      b.cell(label, "vram", "num", 2, [(file, [key, "peak_vram_allocated_gb"])]),
                      b.cell(label, "rss", "num", 2, [(file, [key, "rss_gb"])])])
    return t


TEX_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
              "±": r"$\pm$", "−": r"$-$", "≥": r"$\geq$", "≈": r"$\approx$", "×": r"$\times$", "–": "--",
              "↔": r"$\leftrightarrow$"}


def tex_escape(text):
    return "".join(TEX_ESCAPE.get(c, c) for c in str(text))


def to_md(t):
    n = len(t.columns)
    md = [f"# {t.title}", "", "| " + " | ".join(t.columns) + " |", "|" + "---|" * n]
    for r in t.rows:
        md.append(f"| **{r[1]}** |" + " |" * (n - 1) if r[0] == "section" else "| " + " | ".join([r[1]] + r[2]) + " |")
    return "\n".join(md + ["", t.note, ""])


def to_tex(t):
    n = len(t.columns)
    out = ["% Generated by src/paper_tables.py from results/ (do not edit by hand)", r"\begin{table*}[t]", r"\centering",
           r"\scriptsize", f"\\caption{{{tex_escape(t.title)}. {tex_escape(t.note)}}}", f"\\label{{tab:{t.name}}}",
           r"\begin{tabular}{l" + "r" * (n - 1) + "}", r"\toprule",
           " & ".join(tex_escape(c) for c in t.columns) + r" \\", r"\midrule"]
    first = True
    for r in t.rows:
        if r[0] == "section":
            if not first:
                out.append(r"\midrule")
            out.append(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{tex_escape(r[1])}}}}} \\\\")
        else:
            out.append(" & ".join(tex_escape(c) for c in [r[1]] + r[2]) + r" \\")
        first = False
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="paper/tables")
    args = p.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b = Builder(Source(args.results))
    tables = [
        metric_table(b, "table1", "Table 1: Comparison 1 on the test split (structured features)", "comparison1/eval",
                     ["slm", "xgb", "rules"], [("slm", "xgb", "comparison1/eval/paired_slm_vs_xgb_test_{agg}.json")],
                     "Paired row: SLM − XGBoost, same user draws; * = 95% CI excludes 0. Precision @ k: top 20 units per day."),
        table2(b), text_gain(b), table3(b), calibration(b), abstention(b), review_budget(b),
        fairness(b, "fairness_c1", "Fairness, Comparison 1 (test, session level)",
                 [("xgb", "comparison1/fairness_xgb_test.csv"), ("slm", "comparison1/fairness_slm_test.csv")]),
        fairness(b, "fairness_c2", "Fairness, Comparison 2 (test, session level)",
                 [("xgb_text", "comparison2/fairness_xgb_text_test.csv"), ("slm_text", "comparison2/fairness_slm_text_test.csv")]),
        operations(b),
    ]
    for t in tables:
        (out / f"{t.name}.md").write_text(to_md(t))
        (out / f"{t.name}.tex").write_text(to_tex(t))
    (out / "provenance.json").write_text(json.dumps(b.provenance, indent=1, default=str) + "\n")
    print(f"wrote {len(tables)} tables ({len(b.provenance)} cells) to {out}")


if __name__ == "__main__":
    main()
