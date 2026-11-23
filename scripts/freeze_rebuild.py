"""Freeze (S8-4): results-level rebuild in a clean clone, compared with the frozen tree.

Run INSIDE a fresh clone whose results/ holds a copy of the reference tree's untracked results
files (stored predictions, eval JSONs) and whose data/processed points at the reference data:
  python -m scripts.freeze_rebuild run --reference /path/to/reference/repo [--workers 3]
  python -m scripts.freeze_rebuild compare --reference /path/to/reference/repo --since EPOCH

`run` re-runs, from the stored predictions, every Mac-side producer and report in dependency
order: Step 7 calibration (fits, calibrated predictions, evals, tables), hybrid (select,
sensitivity, build), rules + SLM, Table 3 evals and table, abstention, review budget; every eval
JSON of Comparisons 1-2, the validation baselines and the granularity study, rebuilt from the
command its own JSON records (name, split, preds, session aggregation, calibration, BSS
reference; --rows final when every preds file is a logoff-only file); text gain / DiD; fairness
and time to detection; the Comparison 1-2 tables, the Step 8 report and the paper figures and
tables. No training, no feature building, no GPU work.

`compare` then checks every file under results/ and paper/ written since --since against the
same path in the reference tree: bytes identical, or JSON / CSV / parquet equal within 1e-6
(the logged float32 tolerance) after dropping volatile fields (runtime, timestamps, dates,
commits), or PNG pixels identical; run logs (run.csv) are listed, not compared. Writes
results/freeze/rebuild.json and prints a summary; exit status 1 if anything differs.
"""
import argparse
import concurrent.futures as cf
import json
import math
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.cwd()
PY = sys.executable
TOL = 1e-6
VOLATILE = {"runtime_seconds", "timestamp", "date", "git_commit", "started", "finished", "runtime", "log",
            "elapsed_seconds", "seconds", "fit_seconds"}
EVAL_DIRS = ["results/comparison1/eval", "results/comparison2", "results/eval", "results/granularity"]


def eval_jobs():
    """One job per eval JSON, its command rebuilt from what the JSON records."""
    jobs = {}
    for d in EVAL_DIRS:
        for path in sorted((ROOT / d).glob("*.json")):
            j = json.loads(path.read_text())
            if not isinstance(j, dict) or "config" not in j or not ("preds" in j or "preds_a" in j):
                continue
            rel = str(path.relative_to(ROOT))
            cfg = j["config"] or {}
            preds = j.get("preds") or (j["preds_a"] + j["preds_b"])
            args = ["--split", j["split"], "--out", rel]
            if j["split"] == "test":
                args.append("--allow-test")
            if cfg.get("session_agg"):
                args += ["--session-agg", cfg["session_agg"]]
            if cfg.get("calibration"):
                args.append("--calibration")
            if cfg.get("bss_reference"):
                args += ["--bss-ref-split", cfg["bss_reference"]["split"]]
            if all(str(p).endswith("_final.parquet") for p in preds):
                args += ["--rows", "final"]
            if "preds_a" in j:
                cmd = ["paired", *args, "--name-a", j["name_a"], "--a", *j["preds_a"], "--name-b", j["name_b"],
                       "--b", *j["preds_b"]]
            else:
                cmd = ["single", *args, "--name", j["name"], "--preds", *j["preds"]]
            jobs[f"eval:{rel}"] = ([PY, "run.py", "eval", *cmd], [])
    return jobs


def jobs():
    j = {
        "calibrate": ([PY, "-m", "src.calibrate"], []),
        "calib_eval": ([PY, "-m", "scripts.step7_calibration_eval"], ["calibrate"]),
        "calib_tables": ([PY, "-m", "scripts.step7_calibration_tables"], ["calib_eval"]),
        "hybrid_select": ([PY, "-m", "src.hybrid", "select"], ["calibrate"]),
        "hybrid_sensitivity": ([PY, "-m", "src.hybrid", "sensitivity"], ["calibrate"]),
        "hybrid_build": ([PY, "-m", "src.hybrid", "build"], ["calibrate"]),
        "rules_slm_build": ([PY, "-m", "src.rules_slm", "build", "--splits", "valid", "calib", "test", "--allow-test"], []),
        "rules_slm_report": ([PY, "-m", "src.rules_slm", "report"], ["rules_slm_build"]),
        "table3_eval": ([PY, "-m", "scripts.step7_table3_eval"], ["calibrate", "hybrid_build", "rules_slm_build"]),
        "abstention": ([PY, "-m", "src.abstention"], ["calibrate", "hybrid_build"]),
        "abstention_report": ([PY, "-m", "scripts.step7_abstention_report"], ["abstention"]),
        "review_budget": ([PY, "-m", "src.review_budget"], ["calibrate", "hybrid_build", "rules_slm_build"]),
        "review_budget_report": ([PY, "-m", "src.review_budget", "report"], ["review_budget"]),
    }
    evals = eval_jobs()
    j.update(evals)
    j["table3"] = ([PY, "-m", "scripts.step7_table3"], ["table3_eval"])
    did = json.loads((ROOT / "results/comparison2/text_gain_did.json").read_text())
    j["text_gain"] = ([PY, "-m", "src.text_gain", "--split", "test", "--allow-test",
                       *sum(([f"--{m.replace('_', '-')}", *p] for m, p in did["preds"].items()), []),
                       "--check", *(f"{pair}={path}" for pair, path in did.get("draws_checked_against", {}).items()),
                       "--out", "results/comparison2/text_gain_did.json"],
                      [e for e in evals if "comparison2/paired_" in e])
    after_evals = list(evals)
    test = lambda f: json.loads((ROOT / f).read_text())["preds"]
    for name, csv, valid, test_json, rows in (
            ("slm_c1", "comparison1/fairness_slm_test.csv", "comparison1/eval/slm_valid_max.json", "comparison1/eval/slm_test_max.json", []),
            ("xgb_struct_v3", "comparison1/fairness_xgb_test.csv", "comparison1/eval/xgb_valid_max.json", "comparison1/eval/xgb_test_max.json", []),
            ("slm_text", "comparison2/fairness_slm_text_test.csv", "comparison2/slm_text_valid_final.json", "comparison2/slm_text_test_final.json", ["--rows", "final"]),
            ("xgb_text", "comparison2/fairness_xgb_text_test.csv", "comparison2/xgb_text_valid_final.json", "comparison2/xgb_text_test_final.json", ["--rows", "final"])):
        j[f"fairness:{name}"] = ([PY, "-m", "src.fairness", "--name", name, "--split", "test", "--allow-test",
                                  "--eval-json", f"results/{valid}", "--preds", *test(f"results/{test_json}"), *rows,
                                  "--out", f"results/{csv}"], after_evals)
    for key, name in (("slm", "slm_c1"), ("xgb", "xgb_struct_v3")):
        j[f"ttd:{key}"] = ([PY, "-m", "src.granularity", "--name", name, "--split", "test", "--allow-test",
                            "--eval-json", f"results/comparison1/eval/{key}_valid_max.json",
                            "--preds", *test(f"results/comparison1/eval/{key}_test_max.json"),
                            "--out", f"results/comparison1/ttd_{key}_test.json"], after_evals)
    j["ttd:granularity"] = ([PY, "-m", "src.granularity", "--name", "xgb_struct_v3",
                             "--eval-json", "results/granularity/xgb_struct_v3_hourly_valid.json",
                             "--preds", *[f"results/xgb_struct_v3_s{s}/preds_valid.parquet" for s in (1, 2, 3)],
                             "--out", "results/granularity/xgb_struct_v3_time_to_detection.json"], after_evals)
    everything = [k for k in j]
    j["comparison1_tables"] = ([PY, "scripts/comparison1_tables.py"], everything)
    j["comparison2_tables"] = ([PY, "scripts/comparison2_tables.py"], everything)
    j["step8_report"] = ([PY, "-m", "src.slm.step8_report"], everything)
    j["paper"] = ([PY, "-m", "src.figures"], everything + ["comparison1_tables", "comparison2_tables", "step8_report"])
    j["paper_tables"] = ([PY, "-m", "src.paper_tables"], ["paper"])
    return j


def run(workers):
    todo, done, failed, running = jobs(), set(), {}, {}
    logs = ROOT / "logs/freeze"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with cf.ThreadPoolExecutor(workers) as pool:
        while todo or running:
            for name in [n for n, (_, deps) in todo.items() if all(d in done for d in deps)]:
                if len(running) >= workers:
                    break
                cmd, _ = todo.pop(name)
                log = logs / (name.replace("/", "_").replace(":", "_") + ".log")
                running[pool.submit(subprocess.run, cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT)] = (name, time.time())
                print(f"{time.time() - started:7.0f}s start {name}", flush=True)
            if any(any(d in failed for d in deps) for _, deps in todo.values()):
                for n in [n for n, (_, deps) in todo.items() if any(d in failed for d in deps)]:
                    failed[n] = "dependency failed"
                    todo.pop(n)
            if not running:
                if todo:
                    raise SystemExit(f"stuck: {list(todo)}")
                break
            finished, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                name, t0 = running.pop(f)
                rc = f.result().returncode
                (done.add(name) if rc == 0 else failed.__setitem__(name, f"exit {rc}"))
                print(f"{time.time() - started:7.0f}s {'done' if rc == 0 else 'FAILED'} {name} ({time.time() - t0:.0f}s)", flush=True)
    print(f"finished in {time.time() - started:.0f}s; {len(done)} done, {len(failed)} failed: {failed}")
    (ROOT / "results/freeze").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/freeze/rebuild_run.json").write_text(json.dumps(
        {"seconds": round(time.time() - started), "done": sorted(done), "failed": failed}, indent=1) + "\n")
    return 1 if failed else 0


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def close(a, b, path="$"):
    """First difference beyond TOL as a string, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))[:5]}"
        for k in a:
            d = close(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = close(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
            return None
        return None if abs(a - b) <= TOL else f"{path}: {a!r} vs {b!r}"
    return None if a == b else f"{path}: {str(a)[:80]!r} vs {str(b)[:80]!r}"


def frames_close(a, b):
    import numpy as np
    import pandas as pd
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return f"shape/columns {a.shape} vs {b.shape}"
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            bad = ~((np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= TOL))
            if bad.any():
                i = int(np.argmax(bad))
                return f"{c}[{i}]: {xv[i]!r} vs {yv[i]!r} (max |diff| {np.nanmax(np.abs(xv - yv)):.2e})"
        elif not x.astype(str).equals(y.astype(str)):
            return f"{c}: values differ"
    return None


def compare_file(new, ref):
    if not ref.exists():
        return "new", "no reference file"
    if new.read_bytes() == ref.read_bytes():
        return "identical", ""
    suffix = new.suffix
    if new.name == "run.csv" or suffix in (".log", ".txt"):
        return "run log", "not compared (dates, commits, runtimes)"
    if suffix == ".json":
        d = close(strip(json.loads(new.read_text())), strip(json.loads(ref.read_text())))
        return ("within 1e-6", "") if d is None else ("DIFF", d)
    if suffix in (".csv", ".parquet"):
        import pandas as pd
        read = pd.read_csv if suffix == ".csv" else pd.read_parquet
        d = frames_close(read(new), read(ref))
        return ("within 1e-6", "") if d is None else ("DIFF", d)
    if suffix == ".png":
        import matplotlib.image as mpimg
        import numpy as np
        a, b = mpimg.imread(new), mpimg.imread(ref)
        return ("pixels identical", "") if a.shape == b.shape and np.array_equal(a, b) else ("DIFF", "pixels differ")
    if suffix == ".pdf":
        png = new.with_suffix(".png")
        if png.exists() and mpimg_equal(png, ref.with_suffix(".png")):
            return "pdf metadata only", "PNG twin pixel-identical"
        return "DIFF", "pdf bytes differ"
    if suffix == ".md":
        a, b = new.read_text().splitlines(), ref.read_text().splitlines()
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return "DIFF", f"line {first + 1}: {a[first][:100] if first < len(a) else ''!r} vs {b[first][:100] if first < len(b) else ''!r}"
    return "DIFF", "bytes differ"


def mpimg_equal(a, b):
    import matplotlib.image as mpimg
    import numpy as np
    if not b.exists():
        return False
    x, y = mpimg.imread(a), mpimg.imread(b)
    return x.shape == y.shape and np.array_equal(x, y)


def compare(reference, since):
    tracked = set(subprocess.run(["git", "ls-files", "results", "paper"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.split())
    rows = []
    for base in ("results", "paper"):
        for new in sorted((ROOT / base).rglob("*")):
            rel = str(new.relative_to(ROOT))
            if not new.is_file() or new.stat().st_mtime < since or rel.startswith("results/freeze/"):
                continue
            status, detail = compare_file(new, reference / rel)
            rows.append({"file": rel, "tracked": rel in tracked, "status": status, "detail": detail})
    regenerated = {r["file"] for r in rows}
    not_regenerated = sorted(f for f in tracked if f not in regenerated and not f.startswith("results/freeze/"))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = {"reference_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                              text=True).stdout.strip(),
           "tolerance": TOL, "counts": counts, "files": rows, "tracked_not_regenerated": not_regenerated}
    (ROOT / "results/freeze").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/freeze/rebuild.json").write_text(json.dumps(out, indent=1) + "\n")
    bad = [r for r in rows if r["status"] in ("DIFF", "new")]
    print(f"compared {len(rows)} regenerated files: {counts}")
    for r in bad:
        print(f"  {r['status']}: {r['file']}: {r['detail']}")
    print(f"tracked results/paper files not regenerated: {len(not_regenerated)}")
    return 1 if any(r["status"] == "DIFF" for r in rows) else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["run", "compare"])
    p.add_argument("--reference", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--since", type=float, default=0.0)
    args = p.parse_args()
    if pathlib.Path(args.reference).resolve() == ROOT.resolve():
        raise SystemExit("run inside the clone, not the reference tree")
    sys.exit(run(args.workers) if args.command == "run" else compare(pathlib.Path(args.reference), args.since))


if __name__ == "__main__":
    main()
