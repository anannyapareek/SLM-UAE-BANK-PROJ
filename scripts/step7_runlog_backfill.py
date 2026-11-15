"""Step 7 wrap-up (P8): add results/step7/run.csv rows for 7.2 (hybrid) and 7.3 (rules + SLM).

Those steps wrote their outputs (band_selection.json, band_sensitivity.json, build.json,
report_valid.json) but no run-log rows. This copies the recorded values into run.csv without
rerunning anything; git_commit and date = the commit where each output file first landed;
config carries "backfilled": true. Reads no data or labels. Idempotent (replaces its own rows).
Run from the repo root: python -m scripts.step7_runlog_backfill
"""
import json
import pathlib
import subprocess

import pandas as pd

RUN_LOG = pathlib.Path("results/step7/run.csv")
H, R = pathlib.Path("results/step7/hybrid"), pathlib.Path("results/step7/rules_slm")
MACHINE = "Mac (arm64, Darwin)"
STEPS = ("7.2 hybrid", "7.3 rules_slm")


def landed(path):
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    if not out:
        raise RuntimeError(f"{path} is not committed")
    return out[-2], out[-1]                    # oldest commit adding the file


def row(step, run_id, model, seed, config, metrics, runtime, outputs, source):
    commit, date = landed(source)
    return {"date": date, "git_commit": commit, "run_id": run_id, "step": step, "model": model,
            "config": json.dumps(config | {"backfilled": True, "source": str(source)}), "seed": seed,
            "machine": MACHINE, "metrics": json.dumps(metrics), "runtime_seconds": runtime, "outputs": str(outputs)}


def main():
    sel, sens = json.loads((H / "band_selection.json").read_text()), json.loads((H / "band_sensitivity.json").read_text())
    build, rep = json.loads((H / "build.json").read_text()), json.loads((R / "report_valid.json").read_text())
    best, seeds = sel["best"], sel["config"]["seeds"]
    rows = [row("7.2 hybrid", "hybrid_select", "hybrid", s,
                {k: sel["config"][k] for k in ("select_split", "max_share", "level_step")},
                {"a": best["a"], "b": best["b"], "row_pr_auc_valid": best[f"row_pr_auc_s{s}"],
                 "row_pr_auc_valid_mean": best["row_pr_auc_mean"], "session_pr_auc_valid_mean": best["session_pr_auc_mean"],
                 "share_valid_mean": best["share_mean"]},
                sel["runtime_seconds"], H, H / "band_selection.json") for s in seeds]
    rows.append(row("7.2 hybrid", "hybrid_sensitivity", "hybrid", None, {"split": sens["split"]},
                    {"caps": sens["caps"]}, sens["runtime_seconds"], H, H / "band_sensitivity.json"))
    rows += [row("7.2 hybrid", "hybrid_build", "hybrid", s, {"band": build["band"]},
                 {x["split"]: {k: v for k, v in x.items() if k not in ("split", "seed")} for x in build["shares"] if x["seed"] == s},
                 None, H, H / "build.json") for s in seeds]
    rows.append(row("7.3 rules_slm", "rules_slm_report", "rules_slm", None, {"split": rep["split"]},
                    {k: rep[k] for k in ("zero_rule_positives", "rows_with_a_rule_fired_share", "pr_auc")},
                    rep["runtime_seconds"], R, R / "report_valid.json"))
    old = pd.read_csv(RUN_LOG)
    frame = pd.concat([old[~old["step"].isin(STEPS)], pd.DataFrame(rows)], ignore_index=True)
    order = {s: i for i, s in enumerate(sorted(frame["step"].unique()))}
    frame = frame.sort_values("step", key=lambda c: c.map(order), kind="stable")
    frame.to_csv(RUN_LOG, index=False)
    print(frame.groupby("step", sort=False).size().to_string())


if __name__ == "__main__":
    main()
