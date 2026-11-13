"""Text gain: difference-in-differences (SLM text - SLM struct) - (XGB text - XGB struct).

Uses src/eval.py's own functions (load_meta, prepare, core_metrics, bootstrap, interval)
on logoff rows (--rows final), so every model is resampled with the same user draws as
eval.py's paired tests (same SeedSequence, same units, same strata). Point estimates are
the seed-averaged metrics, as in eval.py paired. Also recomputes, from the same resamples,
the three paired differences, whose CIs must equal those of the paired-test JSONs given
with --check (proof that the draws are shared).

Run: python -m src.text_gain --split test --allow-test \\
         --slm-text p1 p2 p3 --slm-struct p1 p2 p3 --xgb-text p1 p2 p3 --xgb-struct p1 p2 p3 \\
         --out results/comparison2/text_gain_did.json [--check name=paired.json ...]
"""
import argparse
import json
import pathlib
import time

import duckdb
import numpy as np
import yaml

from src import eval as ev

MODELS = ("slm_text", "slm_struct", "xgb_text", "xgb_struct")
PAIRS = {"slm_text_vs_slm_struct": ("slm_text", "slm_struct"),
         "xgb_text_vs_xgb_struct": ("xgb_text", "xgb_struct"),
         "slm_text_vs_xgb_text": ("slm_text", "xgb_text")}


def did(r):
    return (r["slm_text"] - r["slm_struct"]) - (r["xgb_text"] - r["xgb_struct"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["valid", "calib", "test"])
    p.add_argument("--allow-test", action="store_true")
    for name in MODELS:
        p.add_argument(f"--{name.replace('_', '-')}", required=True, nargs="+")
    p.add_argument("--check", nargs="*", default=[],
                   help="pair=paired.json: assert the pair's CI equals that eval.py output")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    checksum = ev.verify_test_checksum()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing to evaluate on test without --allow-test")
    cfg = {k: v for k, v in yaml.safe_load(ev.CONFIG.read_text()).items() if k != "baselines"}
    started = time.perf_counter()
    con = duckdb.connect(config={"threads": 1})
    meta = ev.load_meta(con, args.split, "final")
    paths = {name: getattr(args, name) for name in MODELS}
    out = {"split": args.split, "rows": "final", "unit": "session (= logoff row)",
           "preds": paths, "bootstrap": cfg["bootstrap"], "test_ids_sha256": checksum,
           "n_units": {"row": int(len(meta)), "session": int(meta["session_id"].nunique())},
           "did": {}, "pairs": {}}
    # With --rows final a session has exactly one row: the session unit is the logoff row.
    units = {name: ev.prepare(meta, paths[name], "session", cfg) for name in MODELS}
    point = {name: {k: float(np.mean([ev.core_metrics(s, np.ones(len(s.score)), cfg)[k]
                                      for s in u["scored"]])) for k in ev.CI_METRICS}
             for name, u in units.items()}
    resamples = ev.bootstrap(units, cfg)
    for k in ev.CI_METRICS:
        values = {name: np.array([r[name][k] for r in resamples]) for name in MODELS}
        ci, skipped = ev.interval(did(values), cfg["bootstrap"]["ci"])
        out["did"][k] = {"slm_gain": point["slm_text"][k] - point["slm_struct"][k],
                         "xgb_gain": point["xgb_text"][k] - point["xgb_struct"][k],
                         "difference_in_differences": did({n: point[n][k] for n in MODELS}),
                         "ci": ci, "significant": bool(ci[0] > 0 or ci[1] < 0),
                         "boot_skipped": skipped}
        for pair, (a, b) in PAIRS.items():
            pci, pskip = ev.interval(values[a] - values[b], cfg["bootstrap"]["ci"])
            out["pairs"].setdefault(pair, {})[k] = {"a": point[a][k], "b": point[b][k],
                                                    "difference": point[a][k] - point[b][k],
                                                    "ci": pci, "boot_skipped": pskip}
    out["point"] = point
    out["boot_skipped"] = max(v["boot_skipped"] for v in out["did"].values())
    out["runtime_seconds"] = round(time.perf_counter() - started, 1)

    for spec in args.check:
        pair, path = spec.split("=", 1)
        ref = json.loads(pathlib.Path(path).read_text())["session"]
        for k in ev.CI_METRICS:
            mine = out["pairs"][pair][k]
            if not (np.allclose(mine["ci"], ref[k]["ci"], rtol=0, atol=1e-12)
                    and abs(mine["difference"] - ref[k]["difference"]) < 1e-12):
                raise RuntimeError(f"{pair} {k}: CI {mine['ci']} differs from {path} {ref[k]['ci']}"
                                   " - the user draws are not shared")
        out.setdefault("draws_checked_against", {})[pair] = path
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev.to_json(out), indent=2) + "\n")
    print(f"wrote {path} in {out['runtime_seconds']} s (boot_skipped {out['boot_skipped']}); "
          f"shared draws checked against {len(args.check)} paired test(s)")


if __name__ == "__main__":
    main()
