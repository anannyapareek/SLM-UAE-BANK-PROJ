"""Step 7.4 Table 3 (Comparison 3) from the eval.py outputs in results/step7/table3/eval/.

Writes results/step7/table3/:
  table3.md / .csv     XGBoost, SLM, hybrid, rules, rules + SLM on test; both session
                       aggregations and rows; hybrid routing shares (rows sent to the SLM,
                       sessions with >= 1 SLM row) and the compute line
  paired.md / .csv     hybrid vs XGBoost (key test), hybrid vs SLM, rules + SLM vs rules
  table1_check.md      XGBoost, SLM and rules here vs Table 1 (must agree within 1e-6)
and appends one row per run and seed to results/step7/run.csv (step "7.4 table3").
Reads no data and no labels. Run from the repo root: python -m scripts.step7_table3
"""
import datetime
import json
import pathlib
import platform
import subprocess

import pandas as pd
import yaml

ALL = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())
CFG, HYB = ALL["table3"], ALL["hybrid"]
D = pathlib.Path(CFG["out"])
E = D / "eval"
T1 = pathlib.Path("results/comparison1/eval")
RUN_LOG = pathlib.Path("results/step7/run.csv")
LABELS = {"xgb": "XGBoost (structured v3)", "slm": "SLM (Qwen2.5-1.5B LoRA)",
          "hybrid": "Hybrid (XGBoost; SLM in band)", "rules": "Rules", "rules_slm": "Rules + SLM"}
PAIR_LABELS = {("hybrid", "xgb"): "Hybrid − XGBoost (key test)", ("hybrid", "slm"): "Hybrid − SLM",
               ("rules_slm", "rules"): "Rules + SLM − Rules"}
UNITS = [("max", "session", "Session, max over scoring points (primary)"),
         ("final", "session", "Session, final point (secondary)"),
         ("max", "row", "Row (secondary)")]
METRICS = [("pr_auc", "PR-AUC"), ("recall_at_fpr_0.01", "Recall @ 1% FPR"),
           ("recall_at_fpr_0.05", "Recall @ 5% FPR"), ("precision_at_k", "Precision @ k (20/day)")]
TOLERANCE = 1e-6


def load(name):
    return json.loads((E / name).read_text())


def cell(m, k):
    mean, std, (lo, hi) = m["mean"][k], m["std"][k], m["ci"][k]
    return f"{mean:.3f}{'' if std is None else f' ± {std:.3f}'} [{lo:.3f}, {hi:.3f}]"


def routing():
    shares = pd.DataFrame(json.loads((pathlib.Path(HYB["out"]) / "build.json").read_text())["shares"])
    t = shares[shares["split"] == "test"]
    return {"row_share": (t["row_share"].mean(), t["row_share"].std(ddof=1)),
            "session_share": (t["session_share"].mean(), t["session_share"].std(ddof=1)),
            "rows": int(t["rows"].iloc[0]), "rows_to_slm_mean": float(t["rows_to_slm"].mean()),
            "per_seed": t.to_dict("records")}


def compute_line(r):
    xgb = json.loads(pathlib.Path(CFG["xgb_timing"]).read_text())
    slm_rps, xgb_rps, n = CFG["slm_rows_per_second"], xgb["rows_per_second_median"], r["rows"]
    share = r["row_share"][0]
    slm_s, xgb_s = share * n / slm_rps, n / xgb_rps
    implied = n / (slm_s + xgb_s)
    return {"slm_rows_per_second": slm_rps, "xgb_rows_per_second": xgb_rps, "xgb_machine": xgb["machine"],
            "test_rows": n, "share": share, "slm_seconds": slm_s, "xgb_seconds": xgb_s,
            "implied_rows_per_second": implied, "slm_only_seconds": n / slm_rps,
            "speedup_vs_slm_on_all_rows": implied / slm_rps}


def table3():
    r, c = routing(), None
    c = compute_line(r)
    pct = lambda v: f"{v[0]:.1%} ± {v[1]:.1%}"
    rows, md = [], ["# Table 3: Comparison 3 on the test split", "",
                    "Mean ± std over seeds (XGBoost, SLM, hybrid, rules + SLM: 3 seeds, seed i of the "
                    "hybrid and of rules + SLM using SLM seed i; rules: deterministic, 1 run) and 95% "
                    "user-level bootstrap CI (1,000 resamples, stratified, seed 0), src/eval.py. "
                    "XGBoost and SLM are the Platt row-fit files the hybrid is built from; their "
                    "ranking metrics equal Table 1's (table1_check.md). The hybrid band [a, b] = "
                    f"[{json.loads((pathlib.Path(HYB['out']) / 'build.json').read_text())['band']['a']:.6g}, "
                    f"{json.loads((pathlib.Path(HYB['out']) / 'build.json').read_text())['band']['b']:.6g}] "
                    "was chosen on validation (X = 20%) and not re-selected after test. Precision at k: "
                    "top 20 units per day.", ""]
    for agg, unit, title in UNITS:
        first = load(f"xgb_test_{agg}.json")[unit]
        md += [f"## {title}", "",
               f"{first['counts']['n_units']:,} units, {first['counts']['n_pos']:,} positive "
               f"({first['counts']['positive_rate']:.4%}), {first['counts']['n_users']:,} users; "
               f"precision-at-k ceiling {first['precision_at_k_ceiling']:.3f}.", "",
               "| Model | " + " | ".join(t for _, t in METRICS) + " | Test rows sent to SLM | "
               "Test sessions with ≥ 1 SLM row |", "|---" * (len(METRICS) + 3) + "|"]
        for key in CFG["models"]:
            m = load(f"{key}_test_{agg}.json")[unit]
            extra = (f" | {pct(r['row_share'])} | {pct(r['session_share'])} |" if key == "hybrid"
                     else " | – | – |")
            md.append(f"| {LABELS[key]} | " + " | ".join(cell(m, k) for k, _ in METRICS) + extra)
            for k, _ in METRICS:
                rows.append({"unit": unit, "session_agg": agg if unit == "session" else "n/a", "model": key,
                             "metric": k, "mean": m["mean"][k], "std": m["std"][k], "ci_low": m["ci"][k][0],
                             "ci_high": m["ci"][k][1], "n_units": m["counts"]["n_units"],
                             "n_pos": m["counts"]["n_pos"]})
        md.append("")
    for key, v in (("row_share", r["row_share"]), ("session_share", r["session_share"])):
        rows.append({"unit": "routing", "session_agg": "n/a", "model": "hybrid", "metric": key,
                     "mean": v[0], "std": v[1]})
    md += ["## Hybrid routing and compute", "",
           f"Test rows sent to the SLM: {pct(r['row_share'])} (per seed "
           f"{', '.join(f'{s['row_share']:.1%}' for s in r['per_seed'])}); above the 20% cap set on "
           "validation (19.0% there), from the validation-to-test shift in the XGBoost score "
           "distribution; reported as-is, band not changed (decisions log 2026-11-14). Test "
           f"sessions with at least one SLM row: {pct(r['session_share'])}.", "",
           f"Compute line: the SLM scores {c['share']:.1%} of {c['test_rows']:,} test rows at "
           f"{c['slm_rows_per_second']} rows/s (RTX 4500 Ada, fp16 merged adapter) = "
           f"{c['slm_seconds']:,.0f} s; XGBoost scores all rows at {c['xgb_rows_per_second']:,.0f} "
           f"rows/s ({c['xgb_machine']}, results/step7/xgb_timing.json) = {c['xgb_seconds']:.1f} s. "
           f"Implied hybrid throughput {c['implied_rows_per_second']:,.0f} rows/s, "
           f"{c['speedup_vs_slm_on_all_rows']:.2f}× the SLM scoring every row "
           f"({c['slm_only_seconds']:,.0f} s for the test split). XGBoost was timed on a different "
           "machine (CPU) from the SLM (GPU); its share of the time is negligible either way.", ""]
    (D / "table3.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(D / "table3.csv", index=False, float_format="%.6f")
    (D / "compute.json").write_text(json.dumps({"routing": r, "compute": c}, indent=2, default=float) + "\n")
    return r, c


def paired():
    rows, md = [], ["# Paired tests on the test split (Comparison 3)", "",
                    "Same users in every bootstrap resample (1,000, stratified, seed 0); difference "
                    "of seed-averaged metrics, 95% CI. Significant = CI excludes 0. Hybrid − XGBoost "
                    "is the key test (runbook 7.4).", ""]
    for a, b in CFG["paired"]:
        md += [f"## {PAIR_LABELS[(a, b)]}", ""]
        for agg, unit, title in UNITS:
            d = load(f"paired_{a}_vs_{b}_test_{agg}.json")[unit]
            md += [f"### {title}", "", f"| Metric | {LABELS[a]} | {LABELS[b]} | Difference | 95% CI | Significant |",
                   "|---|---|---|---|---|---|"]
            for k, t in METRICS:
                x = d[k]
                md.append(f"| {t} | {x['a']:.3f} | {x['b']:.3f} | {x['difference']:+.4f} | "
                          f"[{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {'yes' if x['significant'] else 'no'} |")
                rows.append({"a": a, "b": b, "unit": unit, "session_agg": agg if unit == "session" else "n/a",
                             "metric": k, "value_a": x["a"], "value_b": x["b"], "difference": x["difference"],
                             "ci_low": x["ci"][0], "ci_high": x["ci"][1], "significant": x["significant"]})
            md.append("")
    (D / "paired.md").write_text("\n".join(md))
    pd.DataFrame(rows).to_csv(D / "paired.csv", index=False, float_format="%.6f")


def table1_check():
    md = ["# Table 3 vs Table 1: XGBoost, SLM and rules", "",
          f"Means and CI bounds of the ranking metrics must agree within {TOLERANCE:g} (Platt row "
          "fits are strictly increasing; XGBoost float32 ties, see calibration/ranking_check.md).", "",
          "| Model | Aggregation | Unit | Largest abs. difference | Pass |", "|---|---|---|---|---|"]
    bad = []
    for key in ("xgb", "slm", "rules"):
        for agg in CFG["session_aggs"]:
            a, b = load(f"{key}_test_{agg}.json"), json.loads((T1 / f"{key}_test_{agg}.json").read_text())
            for unit in ("session", "row"):
                diff = max(max(abs(a[unit]["mean"][k] - b[unit]["mean"][k]),
                               *[abs(x - y) for x, y in zip(a[unit]["ci"][k], b[unit]["ci"][k])])
                           for k, _ in METRICS)
                ok = diff <= TOLERANCE
                bad += [] if ok else [f"{key} {agg} {unit} ({diff:.1e})"]
                md.append(f"| {key} | {agg} | {unit} | {diff:.1e} | {'yes' if ok else '**NO**'} |")
    md += ["", f"Failures: {', '.join(bad) if bad else 'none'}.", ""]
    (D / "table1_check.md").write_text("\n".join(md))
    return bad


def run_log(r, c):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for key in CFG["models"]:
        for agg in CFG["session_aggs"]:
            m = load(f"{key}_test_{agg}.json")
            seeds = HYB["seeds"] if key in CFG["seeded"] else [None]
            for i, seed in enumerate(seeds):
                config = {"model": key, "session_agg": agg, "preds": CFG["models"][key]}
                if key == "hybrid":
                    config["band"] = json.loads((pathlib.Path(HYB["out"]) / "build.json").read_text())["band"]
                    config["test_row_share"] = r["per_seed"][i]["row_share"]
                    config["test_session_share"] = r["per_seed"][i]["session_share"]
                rows.append({"date": datetime.date.today().isoformat(), "git_commit": commit,
                             "run_id": f"table3_{key}_{agg}", "step": "7.4 table3", "model": key,
                             "config": json.dumps(config), "seed": seed,
                             "machine": f"Mac ({platform.machine()}, {platform.system()})",
                             "metrics": json.dumps({u: {k: m[u]["per_seed"][i].get(k) for k, _ in METRICS}
                                                    for u in ("session", "row")}),
                             "runtime_seconds": m.get("runtime_seconds"), "outputs": str(E)})
    frame = pd.DataFrame(rows)
    if RUN_LOG.exists():
        old = pd.read_csv(RUN_LOG)
        frame = pd.concat([old[old["step"] != "7.4 table3"], frame], ignore_index=True)
    frame.to_csv(RUN_LOG, index=False)


if __name__ == "__main__":
    r, c = table3()
    paired()
    bad = table1_check()
    run_log(r, c)
    print(f"wrote Table 3, paired tests, Table 1 check ({len(bad)} mismatches), run log")
