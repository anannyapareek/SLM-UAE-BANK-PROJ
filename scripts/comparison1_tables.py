"""Comparison 1 summary tables from the eval, time-to-detection and fairness outputs.

Reads results/comparison1/{eval/*.json, ttd_*_test.json, fairness_*_test.csv} (no data, no
labels) and writes results/comparison1/:
  table1.md, table1.csv            test metrics, both session aggregations and rows
  paired_slm_vs_xgb.md/.csv        paired user-bootstrap differences, SLM - XGBoost
  time_to_detection.md             at each model's validation 5%-FPR threshold
  fairness.md                      FPR and recall per group at that threshold
Run from the repo root: python scripts/comparison1_tables.py
"""
import json
import pathlib

import pandas as pd

D = pathlib.Path("results/comparison1")
MODELS = [("slm", "SLM (Qwen2.5-1.5B LoRA)"), ("xgb", "XGBoost (structured v3)"), ("rules", "Rules")]
UNITS = [("max", "session", "Session, max over scoring points (primary)"),
         ("final", "session", "Session, final point (secondary)"),
         ("max", "row", "Row (secondary)")]
METRICS = [("pr_auc", "PR-AUC"), ("recall_at_fpr_0.01", "Recall @ 1% FPR"),
           ("recall_at_fpr_0.05", "Recall @ 5% FPR"), ("precision_at_k", "Precision @ k (20/day)")]


def load(name):
    return json.loads((D / "eval" / name).read_text())


def cell(m, k):
    mean, std, (lo, hi) = m["mean"][k], m["std"][k], m["ci"][k]
    sd = "" if std is None else f" ± {std:.3f}"
    return f"{mean:.3f}{sd} [{lo:.3f}, {hi:.3f}]"


def table1():
    rows, md = [], ["# Table 1: Comparison 1 on the test split", "",
                    "Mean ± std over seeds (SLM and XGBoost: 3 seeds; rules: deterministic, 1 run) "
                    "and 95% user-level bootstrap CI (1,000 resamples). Recall at FPR is the "
                    "pre-declared curve metric of src/eval.py (highest recall with FPR at or below "
                    "the level on this split). Precision at k: top 20 units per day.", ""]
    for agg, unit, title in UNITS:
        first = load(f"slm_test_{agg}.json")[unit]["counts"]
        md += [f"## {title}", "",
               f"{first['n_units']:,} units, {first['n_pos']:,} positive "
               f"({first['positive_rate']:.4%}), {first['n_users']:,} users; precision-at-k "
               f"ceiling {load(f'slm_test_{agg}.json')[unit]['precision_at_k_ceiling']:.3f}.", "",
               "| Model | " + " | ".join(t for _, t in METRICS) + " |",
               "|---" * (len(METRICS) + 1) + "|"]
        for key, label in MODELS:
            m = load(f"{key}_test_{agg}.json")[unit]
            md.append(f"| {label} | " + " | ".join(cell(m, k) for k, _ in METRICS) + " |")
            for k, t in METRICS:
                rows.append({"unit": unit, "session_agg": agg if unit == "session" else "n/a",
                             "model": key, "metric": k, "mean": m["mean"][k], "std": m["std"][k],
                             "ci_low": m["ci"][k][0], "ci_high": m["ci"][k][1],
                             "n_units": m["counts"]["n_units"], "n_pos": m["counts"]["n_pos"]})
        md.append("")
    (D / "table1.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(D / "table1.csv", index=False, float_format="%.6f")


def paired():
    rows, md = [], ["# Paired test: SLM - XGBoost on the test split", "",
                    "Same users in every bootstrap resample (1,000, stratified); difference of "
                    "seed-averaged metrics, 95% CI. Significant = CI excludes 0.", ""]
    for agg, unit, title in UNITS:
        d = load(f"paired_slm_vs_xgb_test_{agg}.json")[unit]
        md += [f"## {title}", "", "| Metric | SLM | XGBoost | Difference | 95% CI | Significant |",
               "|---|---|---|---|---|---|"]
        for k, t in METRICS:
            r = d[k]
            md.append(f"| {t} | {r['a']:.3f} | {r['b']:.3f} | {r['difference']:+.3f} | "
                      f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {'yes' if r['significant'] else 'no'} |")
            rows.append({"unit": unit, "session_agg": agg if unit == "session" else "n/a",
                         "metric": k, "slm": r["a"], "xgb": r["b"], "difference": r["difference"],
                         "ci_low": r["ci"][0], "ci_high": r["ci"][1], "significant": r["significant"]})
        md.append("")
    (D / "paired_slm_vs_xgb.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(D / "paired_slm_vs_xgb.csv", index=False, float_format="%.6f")


def ttd():
    md = ["# Time to detection on the test split (hourly scoring)", "",
          "Alert = first scoring point with score >= the seed's session-level 5%-FPR threshold "
          "set on VALIDATION (max aggregation). Hours pooled over the 3 seeds (median, IQR).", "",
          "| Model | Thresholds (valid) | Test session FPR at them | Detected | Detected before logoff | "
          "Start to alert (h) | First malicious event to alert (h) | Alert before logoff (h) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, label in MODELS[:2]:
        d = json.loads((D / f"ttd_{key}_test.json").read_text())
        seeds, pooled = d["per_seed"], d["pooled_over_seeds"]
        h = lambda x: f"{x['median']:.2f} ({x['q1']:.2f} to {x['q3']:.2f})"
        md.append(f"| {label} | {', '.join(f'{s['threshold']:.4g}' for s in seeds)} | "
                  f"{', '.join(f'{s['session_fpr_at_threshold']:.3f}' for s in seeds)} | "
                  f"{pooled['detection_rate']:.1%} | {pooled['share_of_positives_detected_before_logoff']:.1%} "
                  f"of positives ({pooled['share_of_detected_before_logoff']:.1%} of detected) | "
                  f"{h(pooled['hours_start_to_alert'])} | {h(pooled['hours_first_event_to_alert'])} | "
                  f"{h(pooled['hours_alert_before_logoff'])} |")
    d = json.loads((D / "ttd_slm_test.json").read_text())
    md += ["", f"{d['positive_sessions']} positive test sessions; first malicious event a median "
           f"{d['hours_start_to_first_malicious_event']['median']:.2f} h after session start."]
    (D / "time_to_detection.md").write_text("\n".join(md) + "\n")


SMALL = 500        # sessions: extremes from groups below this are flagged as small-sample


def extremes(g, column):
    """Group names at the lowest and highest value of `column` (ties included)."""
    g = g.dropna(subset=[column])
    if g.empty:
        return set(), set()
    return (set(g.loc[g[column] == g[column].min(), "group"]),
            set(g.loc[g[column] == g[column].max(), "group"]))


FAIRNESS_INTRO = ("Session units, max aggregation; alert at each seed's validation 5%-FPR "
                  "threshold. ")


def fairness(directory=D, models=None, title="# Fairness on the test split (runbook 6.4)",
             intro=FAIRNESS_INTRO, csv="fairness_{key}_test.csv"):
    """Fairness report from src/fairness.py CSVs; also used for Comparison 2."""
    models = models or MODELS[:2]
    md = [title, "",
          intro +
          "Mean ± std over 3 seeds. Groups: the pre-declared fairness proxies role, department "
          "and functional_unit (tenure was dropped on 2026-11-04 and is not available). Shown "
          "only for groups with >= 200 sessions; recall only where a group has >= 5 positive "
          "sessions. Flags: the lowest and highest FPR (and recall) in each attribute; "
          f"**small sample** marks such an extreme from a group with fewer than {SMALL} sessions.", ""]
    for key, label in models:
        f = pd.read_csv(directory / csv.format(key=key))
        md += [f"## {label}", ""]
        for attribute in ["all", "role", "department", "functional_unit"]:
            g = f[f["attribute"] == attribute].sort_values("sessions", ascending=False)
            fpr_lo, fpr_hi = extremes(g, "fpr_mean")
            rec_lo, rec_hi = extremes(g, "recall_mean")
            md += [f"### {attribute}", ""]
            if attribute != "all":
                def where(groups, col):
                    rows = g[g["group"].isin(groups)]
                    return "; ".join(f"{r.group} {getattr(r, col):.3f} (n = {r.sessions:,}"
                                     f"{', small sample' if r.sessions < SMALL else ''})"
                                     for r in rows.itertuples(index=False))
                md.append(f"FPR range: lowest {where(fpr_lo, 'fpr_mean')}; highest "
                          f"{where(fpr_hi, 'fpr_mean')}.")
                if rec_lo:
                    md.append(f"Recall range: lowest {where(rec_lo, 'recall_mean')}; highest "
                              f"{where(rec_hi, 'recall_mean')}.")
                md.append("")
            md += ["| Group | Sessions | Positives | FPR | Recall | Flag |",
                   "|---|---|---|---|---|---|"]
            for r in g.itertuples(index=False):
                recall = "n/a (< 5 positives)" if pd.isna(r.recall_mean) else f"{r.recall_mean:.3f} ± {r.recall_std:.3f}"
                flags = []
                if attribute != "all":
                    flags += ["lowest FPR"] * (r.group in fpr_lo) + ["highest FPR"] * (r.group in fpr_hi)
                    flags += ["lowest recall"] * (r.group in rec_lo) + ["highest recall"] * (r.group in rec_hi)
                    if flags and r.sessions < SMALL:
                        flags.append("**small sample** (n < 500)")
                md.append(f"| {r.group} | {r.sessions:,} | {r.positives} | "
                          f"{r.fpr_mean:.4f} ± {r.fpr_std:.4f} | {recall} | {', '.join(flags)} |")
            md.append("")
    (directory / "fairness.md").write_text("\n".join(md))


if __name__ == "__main__":
    table1()
    paired()
    ttd()
    fairness()
    print("wrote", ", ".join(sorted(p.name for p in D.glob("*.md"))))
