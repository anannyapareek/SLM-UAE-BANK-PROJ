"""Step 7.1 summary: calibration table, ranking check, run log and reliability diagrams.

Reads only the eval.py outputs in results/step7/calibration/<run_id>/metrics_<split>.json and
the fit records (params.json). Writes, in results/step7/calibration/:
  table_calibration.md / .csv   Brier, ECE (equal-mass, equal-width), mean predicted vs
                                observed rate, before and after, calib and test; mean ± std
                                over seeds and 95% user-bootstrap CI
  ranking_check.md              every fit's ranking metrics vs the uncalibrated ones (must be equal)
  reliability_test.png / .pdf   reliability diagrams, test, primary unit, before and after
and appends one row per run and seed to results/step7/run.csv.
Run from the repo root: python -m scripts.step7_calibration_tables
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

from src import eval as ev  # noqa: E402

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["calibration"]
ROOT = pathlib.Path(CFG["out"])
RUN_LOG = pathlib.Path("results/step7/run.csv")
LABELS = {"xgb_struct": "XGBoost v3 (C1)", "slm_struct": "SLM C1", "xgb_text": "XGBoost text (C2)",
          "slm_text": "SLM text (C2)"}
METHODS = {"uncalibrated": "Uncalibrated", "temperature": "Temperature (primary)",
           "platt": "Platt (sensitivity)"}
COLOURS = {"uncalibrated": "#999999", "temperature": "#0072B2", "platt": "#D55E00"}   # Okabe-Ito
MARKERS = {"uncalibrated": "o", "temperature": "s", "platt": "^"}
CAL = ["brier", "ece_equal_mass", "ece_equal_width"]
BSS = ["bss_reference", "bss_own"]
CAPTION = ("Reliability diagrams on log-log axes (15 equal-mass bins; one line per seed); bins with "
           "no observed positives are drawn at the 1e-5 floor of the y axis.")


def load(run_id, split):
    return json.loads((ROOT / run_id / f"metrics_{split}.json").read_text())


def rows_for(model):
    """(run_id, method, fit unit, eval unit) for one model; eval unit = the unit that fit targets."""
    out = [(f"{model['key']}_uncalibrated", "uncalibrated", "-", u) for u in
           (["session", "row"] if model["comparison"] == "C1" else ["session"])]
    for unit in model["fit_units"]:
        for method in CFG["methods"]:
            out.append((f"{model['key']}_{unit}fit_{method}", method, unit, unit))
    return out


def fmt(m, k):
    std = m["std"].get(k)
    return (f"{m['mean'][k]:.4f}" + ("" if std is None else f" ± {std:.4f}")
            + (f" [{m['ci'][k][0]:.4f}, {m['ci'][k][1]:.4f}]" if k in m["ci"] else ""))


def table():
    csv, md = [], ["# Calibration (Step 7.1): before and after, calib and test", "",
                   "Fitted on calib only. Mean ± std over seeds 1-3 and 95% user-level bootstrap CI "
                   "(1,000 resamples, stratified, seed 0), src/eval.py --calibration. ECE with 15 "
                   "bins. Session = the primary unit (C1: max over scoring points; C2: the logoff "
                   "row); row = C1's secondary view, with its own row-level fit. 'Mean predicted' vs "
                   "'observed' is the prior-shift diagnostic. Brier skill score BSS = 1 − Brier / "
                   "(p(1 − p)): primary p = the calib positive rate at the same unit (known at "
                   "deployment), secondary p = the evaluated split's own rate. Rules are not "
                   "calibrated (Brier and ECE n/a).", "",
                   "Platt Brier ≈ climatology: calibrated probabilities carry little skill beyond the "
                   "base rate; ranking metrics (Tables 1–2) carry the signal.", ""]
    for model in CFG["models"]:
        md += [f"## {LABELS[model['key']]}", ""]
        for split in ("calib", "test"):
            md += [f"### {split}", "", "| Method | Fit unit | Eval unit | Brier | ECE (equal-mass) | "
                   "ECE (equal-width) | BSS (calib base rate) | BSS (own base rate) | Mean predicted | "
                   "Observed rate |", "|---|---|---|---|---|---|---|---|---|---|"]
            for run_id, method, fit_unit, unit in rows_for(model):
                m = load(run_id, split)[unit]
                md.append(f"| {METHODS[method]} | {fit_unit} | {unit} | " +
                          " | ".join(fmt(m, k) for k in CAL + BSS + ["mean_predicted"]) +
                          f" | {m['mean']['observed_rate']:.4f} |")
                for k in CAL + BSS + ["mean_predicted", "observed_rate"]:
                    csv.append({"model": model["key"], "split": split, "method": method,
                                "fit_unit": fit_unit, "eval_unit": unit, "metric": k,
                                "mean": m["mean"][k], "std": m["std"].get(k),
                                "ci_low": m["ci"].get(k, [None, None])[0],
                                "ci_high": m["ci"].get(k, [None, None])[1],
                                "boot_skipped": m["boot_skipped"].get(k)})
            md.append("")
    md += ["## Reliability diagrams", "", "`reliability_test.png` / `.pdf`: test sessions, before "
           "and after (temperature, Platt), primary (session) fit. " + CAPTION, ""]
    (ROOT / "table_calibration.md").write_text("\n".join(md))
    pd.DataFrame(csv).to_csv(ROOT / "table_calibration.csv", index=False, float_format="%.6f")


TOLERANCE = 1e-6


def ranking_check():
    """Ranking metrics must not change; differences up to TOLERANCE come from float32 ties."""
    md = ["# Ranking check: calibration must not change ranking metrics", "",
          "Every fit is strictly increasing in the margin, so PR-AUC, recall at FPR and "
          "precision/recall at k (means, per-seed values and CI bounds) must equal the "
          "uncalibrated ones, per unit and split. Exact equality holds for the SLM and XGBoost "
          "text. XGBoost v3 stores its probability as float32, which ties a few rows whose "
          "margins differ (test: 778,673 distinct scores vs 778,678 distinct margins); the "
          "float64 calibrated scores separate those ties, which moves row-level test PR-AUC by "
          f"at most 6.3e-07. The check therefore passes within {TOLERANCE:g} and reports the "
          "largest absolute difference.", "",
          "| Run | Split | Unit | Largest abs. difference | Pass |", "|---|---|---|---|---|"]
    bad = []
    for model in CFG["models"]:
        base = f"{model['key']}_uncalibrated"
        for run_id, method, _, _ in rows_for(model):
            if method == "uncalibrated":
                continue
            for split in ("calib", "test"):
                a, b = load(base, split), load(run_id, split)
                for unit in ev.UNITS:
                    diff = max(max([abs(a[unit]["mean"][k] - b[unit]["mean"][k])]
                                   + [abs(x - y) for x, y in zip(a[unit]["ci"][k], b[unit]["ci"][k])]
                                   + [abs(x[k] - y[k]) for x, y in zip(a[unit]["per_seed"], b[unit]["per_seed"])])
                               for k in ev.CI_METRICS)
                    ok = diff <= TOLERANCE
                    md.append(f"| {run_id} | {split} | {unit} | {diff:.1e} | {'yes' if ok else '**NO**'} |")
                    if not ok:
                        bad.append(f"{run_id} {split} {unit} ({diff:.1e})")
    md += ["", f"Failures: {', '.join(bad) if bad else 'none'}.", ""]
    (ROOT / "ranking_check.md").write_text("\n".join(md))
    return bad


def reliability():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, model in zip(axes.ravel(), CFG["models"]):
        unit = "session"
        runs = [(f"{model['key']}_uncalibrated", "uncalibrated"),
                (f"{model['key']}_sessionfit_temperature", "temperature"),
                (f"{model['key']}_sessionfit_platt", "platt")]
        lo = 1e-5
        ax.plot([lo, 1], [lo, 1], color="#444444", lw=1, ls="--", label="Perfect calibration")
        for run_id, method in runs:
            for i, seed in enumerate(load(run_id, "test")[unit]["per_seed"]):
                r = seed["reliability_equal_mass"]
                pts = [(c, a) for c, a, n in zip(r["mean_confidence"], r["accuracy"], r["count"])
                       if n and n > 0 and c and c > 0]
                x = [p[0] for p in pts]
                y = [max(p[1], lo) for p in pts]          # observed 0 drawn at the axis floor
                ax.plot(x, y, color=COLOURS[method], marker=MARKERS[method], ms=4, lw=1.5,
                        alpha=0.9, label=METHODS[method] if i == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, 1)
        ax.set_ylim(lo, 1)
        ax.set_title(f"{LABELS[model['key']]}, test sessions")
        ax.set_xlabel("Mean predicted probability (15 equal-mass bins)")
        ax.set_ylabel("Observed positive rate")
        ax.grid(True, which="major", color="#dddddd", lw=0.6)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Reliability before and after calibration (fitted on calib; one line per seed)")
    fig.text(0.5, -0.01, CAPTION, ha="center", va="top", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"reliability_test.{ext}", dpi=150, bbox_inches="tight")


def run_log():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for model in CFG["models"]:
        for run_id, method, fit_unit, unit in rows_for(model):
            for i, seed in enumerate(CFG["seeds"]):
                per = {split: load(run_id, split)[unit]["per_seed"][i] for split in ("calib", "test")}
                params = ({} if method == "uncalibrated" else
                          json.loads((ROOT / run_id / f"s{seed}" / "params.json").read_text()))
                rows.append({
                    "date": datetime.date.today().isoformat(), "git_commit": commit,
                    "run_id": run_id, "step": "7.1 calibration", "model": model["key"],
                    "config": json.dumps({"method": method, "fit_unit": fit_unit, "eval_unit": unit,
                                          "fit_split": CFG["fit_split"],
                                          "params": params.get("params", {})}),
                    "seed": seed,
                    "machine": params.get("machine", f"{platform.node()} ({platform.machine()})"),
                    "metrics": json.dumps({split: {k: per[split].get(k) for k in
                                                   CAL + BSS + ["mean_predicted", "observed_rate"]}
                                           for split in per}),
                    "runtime_seconds": params.get("fit_seconds"),
                    "outputs": str(ROOT / run_id)})
    frame = pd.DataFrame(rows)
    if RUN_LOG.exists():
        old = pd.read_csv(RUN_LOG)
        frame = pd.concat([old[old["step"] != "7.1 calibration"], frame], ignore_index=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RUN_LOG, index=False)


if __name__ == "__main__":
    table()
    bad = ranking_check()
    reliability()
    run_log()
    print(f"wrote calibration table, ranking check ({len(bad)} mismatches), reliability diagrams, run log")
