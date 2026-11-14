"""Step 7.2 hybrid: XGBoost scores every row; rows whose calibrated XGBoost score falls in the
uncertain band [a, b] take the SLM's score instead (configs/step7.yaml, decisions log 2026-11-14).

Both models' scores are Platt row-level calibrated (fitted on calib), so they share a
probability scale. Band selection uses validation only: candidates a, b are quantiles of the
pooled calibrated XGBoost validation scores; a band is admissible if, for every seed, at
most max_share of validation rows fall in it; the objective is row-level validation PR-AUC
(src/eval.py's average_precision) averaged over the seed pairings (XGBoost seed i with SLM
seed i). Session-level PR-AUC is recorded for the top bands. Comparison 1's SLM already
scored every hourly row, so band rows are looked up, not rescored.

Run: python -m src.hybrid select       (validation only; writes results/step7/hybrid/band_selection.*)
     python -m src.hybrid sensitivity  (validation only; other caps X; band_sensitivity.*)
     python -m src.hybrid build        (logged band applied to build_splits; s<seed>/preds_<split>.parquet
                                        with source = xgb/slm; reports shares sent, no metrics)
"""
import argparse
import json
import pathlib
import time

import duckdb
import numpy as np
import pandas as pd
import yaml

from src import eval as ev

CFG = yaml.safe_load(pathlib.Path("configs/step7.yaml").read_text())["hybrid"]


def band_mask(p_xgb, a, b):
    """Rows sent to the SLM: calibrated XGBoost score within [a, b] (both inclusive)."""
    return (p_xgb >= a) & (p_xgb <= b)


def hybrid_score(p_xgb, p_slm, a, b):
    """SLM score inside the band, XGBoost score outside."""
    return np.where(band_mask(p_xgb, a, b), p_slm, p_xgb)


def files(kind, split):
    return [str(pathlib.Path(CFG[kind].format(seed=s)) / f"preds_{split}.parquet") for s in CFG["seeds"]]


def session_pr_auc(meta, scores):
    units, (s,) = ev.aggregate_sessions(meta, [scores])
    return ev.pr_auc(s, units["y"].to_numpy())


def search(xgb, slm, y, max_share):
    """All admissible bands for one cap, sorted by mean row-level PR-AUC (best first)."""
    levels = np.round(np.arange(0, 1 + CFG["level_step"] / 2, CFG["level_step"]), 10)
    edges = np.quantile(np.concatenate(xgb), levels)
    rows, n = [], len(levels)
    for i in range(n):
        for j in range(i + 1, n):
            if levels[j] - levels[i] > max_share + 1e-9:
                break
            a, b = float(edges[i]), float(edges[j])
            shares = [float(band_mask(x, a, b).mean()) for x in xgb]
            if max(shares) > max_share:
                continue
            aucs = [ev.pr_auc(hybrid_score(x, s, a, b), y) for x, s in zip(xgb, slm)]
            rows.append({"i": i, "j": j, "level_a": levels[i], "level_b": levels[j], "a": a, "b": b,
                         "share_mean": float(np.mean(shares)), "share_max": max(shares),
                         "row_pr_auc_mean": float(np.mean(aucs)), "row_pr_auc_std": float(np.std(aucs, ddof=1)),
                         **{f"row_pr_auc_s{s}": v for s, v in zip(CFG["seeds"], aucs)}})
    return pd.DataFrame(rows).sort_values("row_pr_auc_mean", ascending=False).reset_index(drop=True), n


def session_aucs(meta, xgb, slm, a, b):
    return [session_pr_auc(meta, hybrid_score(x, s, a, b)) for x, s in zip(xgb, slm)]


def load_select_split():
    checksum = ev.verify_test_checksum()
    split = CFG["select_split"]
    if split == "test":
        raise SystemExit("the band is never chosen on test")
    con = duckdb.connect(config={"threads": 1})
    meta = ev.load_meta(con, split)
    xgb = ev.load_preds(files("xgb", split), meta)            # aligned to meta, labels checked
    slm = ev.load_preds(files("slm", split), meta)
    return checksum, split, meta, meta["y"].to_numpy(), xgb, slm


def references(meta, y, xgb, slm):
    return {"xgb_alone": {"row": float(np.mean([ev.pr_auc(x, y) for x in xgb])),
                          "session": float(np.mean([session_pr_auc(meta, x) for x in xgb]))},
            "slm_alone": {"row": float(np.mean([ev.pr_auc(s, y) for s in slm])),
                          "session": float(np.mean([session_pr_auc(meta, s) for s in slm]))}}


def select():
    started = time.perf_counter()
    checksum, split, meta, y, xgb, slm = load_select_split()
    frame, n = search(xgb, slm, y, CFG["max_share"])
    top = frame.head(CFG["top"]).copy()
    top["session_pr_auc_mean"] = [float(np.mean(session_aucs(meta, xgb, slm, r.a, r.b))) for r in top.itertuples()]
    best = top.iloc[0]
    reference = references(meta, y, xgb, slm)
    neighbours = frame[(frame["i"].between(best.i - 1, best.i + 1)) & (frame["j"].between(best.j - 1, best.j + 1))
                       & ~((frame["i"] == best.i) & (frame["j"] == best.j))]
    flat = {"neighbours": int(len(neighbours)),
            "largest_drop_to_a_neighbour": float(best.row_pr_auc_mean - neighbours["row_pr_auc_mean"].min()),
            "bands_within_0.001": int((frame["row_pr_auc_mean"] >= best.row_pr_auc_mean - 0.001).sum()),
            "bands_within_0.005": int((frame["row_pr_auc_mean"] >= best.row_pr_auc_mean - 0.005).sum())}
    edge = {"a_at_lowest_level": bool(best.i == 0), "b_at_highest_level": bool(best.j == n - 1),
            "width_at_share_cap": bool(best.level_b - best.level_a >= CFG["max_share"] - 1e-9)}
    out = pathlib.Path(CFG["out"])
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "band_selection.csv", index=False, float_format="%.8g")
    result = {"split": split, "test_ids_sha256": checksum, "config": CFG, "candidates": int(len(frame)),
              "best": best.to_dict(), "top": top.to_dict("records"), "reference": reference,
              "flatness": flat, "edge": edge, "runtime_seconds": round(time.perf_counter() - started, 1)}
    (out / "band_selection.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    md = ["# Hybrid band selection (Step 7.2, validation only)", "",
          f"{len(frame):,} admissible bands (at most {CFG['max_share']:.0%} of validation rows sent to "
          "the SLM for every seed). Scores: Platt row-level calibrated XGBoost v3 and SLM C1; seed i "
          "paired with seed i; objective = row-level validation PR-AUC averaged over the 3 pairs.", "",
          f"References: XGBoost alone row {reference['xgb_alone']['row']:.4f} / session "
          f"{reference['xgb_alone']['session']:.4f}; SLM alone row {reference['slm_alone']['row']:.4f} / "
          f"session {reference['slm_alone']['session']:.4f}.", "",
          "| Rank | a (level) | b (level) | Share to SLM (mean, max) | Row PR-AUC (mean ± std) | Session PR-AUC (mean) |",
          "|---|---|---|---|---|---|"]
    for k, r in enumerate(top.itertuples(), 1):
        md.append(f"| {k} | {r.a:.6g} ({r.level_a:.2f}) | {r.b:.6g} ({r.level_b:.2f}) | "
                  f"{r.share_mean:.3f}, {r.share_max:.3f} | {r.row_pr_auc_mean:.4f} ± {r.row_pr_auc_std:.4f} | "
                  f"{r.session_pr_auc_mean:.4f} |")
    md += ["", f"Flatness: {flat}", f"Edge: {edge}", ""]
    (out / "band_selection.md").write_text("\n".join(md))
    print("\n".join(md))


def sensitivity():
    """Secondary, validation only: the best band under other caps X. Builds no predictions."""
    started = time.perf_counter()
    checksum, split, meta, y, xgb, slm = load_select_split()
    reference = references(meta, y, xgb, slm)
    xgb_row = [ev.pr_auc(x, y) for x in xgb]
    xgb_session = [session_pr_auc(meta, x) for x in xgb]
    rows = []
    for cap in CFG["sensitivity_caps"]:
        frame, _ = search(xgb, slm, y, cap)
        best = frame.iloc[0]
        row_aucs = [best[f"row_pr_auc_s{s}"] for s in CFG["seeds"]]
        sess = session_aucs(meta, xgb, slm, best.a, best.b)
        rows.append({"cap": cap, "primary": cap == CFG["max_share"], "candidates": int(len(frame)),
                     "a": float(best.a), "b": float(best.b), "level_a": float(best.level_a),
                     "level_b": float(best.level_b), "share_mean": float(best.share_mean),
                     "share_max": float(best.share_max),
                     "row_pr_auc_mean": float(np.mean(row_aucs)), "row_pr_auc_std": float(np.std(row_aucs, ddof=1)),
                     "session_pr_auc_mean": float(np.mean(sess)), "session_pr_auc_std": float(np.std(sess, ddof=1)),
                     "row_gain_mean": float(np.mean(np.subtract(row_aucs, xgb_row))),
                     "row_gain_std": float(np.std(np.subtract(row_aucs, xgb_row), ddof=1)),
                     "session_gain_mean": float(np.mean(np.subtract(sess, xgb_session))),
                     "session_gain_std": float(np.std(np.subtract(sess, xgb_session), ddof=1))})
        print(f"cap {cap:.0%}: {rows[-1]}", flush=True)
    out = pathlib.Path(CFG["out"])
    pd.DataFrame(rows).to_csv(out / "band_sensitivity.csv", index=False, float_format="%.8g")
    (out / "band_sensitivity.json").write_text(json.dumps(
        {"split": split, "test_ids_sha256": checksum, "caps": rows, "reference": reference,
         "runtime_seconds": round(time.perf_counter() - started, 1)}, indent=2, default=float) + "\n")
    md = ["# Hybrid cap sensitivity (Step 7.2, validation only, secondary)", "",
          f"X = {CFG['max_share']:.0%} is the primary, pre-declared cap; the other caps are a "
          "sensitivity analysis for the discussion. Same search as band_selection.md (quantile step "
          f"{CFG['level_step']}, share <= X for every seed, objective = mean row-level validation "
          "PR-AUC over the 3 seed pairings). No test predictions are built for the other caps. Gain = "
          "hybrid minus XGBoost alone, per seed pairing (mean ± std over the 3 pairings).", "",
          f"XGBoost alone: row {reference['xgb_alone']['row']:.4f}, session {reference['xgb_alone']['session']:.4f}. "
          f"SLM alone: row {reference['slm_alone']['row']:.4f}, session {reference['slm_alone']['session']:.4f}.", "",
          "| X | Best [a, b] (quantile levels) | Share to SLM (mean, max) | Row PR-AUC | Session PR-AUC | "
          "Row gain over XGBoost | Session gain over XGBoost |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cap']:.0%}{' (primary)' if r['primary'] else ''} | [{r['a']:.6g}, {r['b']:.6g}] "
                  f"({r['level_a']:.2f}–{r['level_b']:.2f}) | {r['share_mean']:.3f}, {r['share_max']:.3f} | "
                  f"{r['row_pr_auc_mean']:.4f} ± {r['row_pr_auc_std']:.4f} | "
                  f"{r['session_pr_auc_mean']:.4f} ± {r['session_pr_auc_std']:.4f} | "
                  f"{r['row_gain_mean']:+.4f} ± {r['row_gain_std']:.4f} | "
                  f"{r['session_gain_mean']:+.4f} ± {r['session_gain_std']:.4f} |")
    md.append("")
    (out / "band_sensitivity.md").write_text("\n".join(md))
    print("\n".join(md))


def platt_margin(kind, seed, split, meta):
    """Calibrated logit a * m + b of one model's rows, aligned to meta (for logit-space use later)."""
    folder = pathlib.Path(CFG[kind].format(seed=seed))
    params = json.loads((folder / "params.json").read_text())["params"]
    m = pd.read_parquet(folder / f"preds_{split}.parquet", columns=["row_id", "margin"]).set_index("row_id")
    return params["a"] * m["margin"].reindex(meta["row_id"]).to_numpy(float) + params["b"]


def build():
    """Hybrid predictions with the logged band (no re-selection); reports only shares sent."""
    checksum = ev.verify_test_checksum()
    record = json.loads((pathlib.Path(CFG["out"]) / "band_selection.json").read_text())
    a, b = record["best"]["a"], record["best"]["b"]
    con = duckdb.connect(config={"threads": 1})
    shares = []
    for split in CFG["build_splits"]:
        meta = ev.load_meta(con, split)
        xgb = ev.load_preds(files("xgb", split), meta)
        slm = ev.load_preds(files("slm", split), meta)
        for seed, x, s in zip(CFG["seeds"], xgb, slm):
            inside = band_mask(x, a, b)
            margin = np.where(inside, platt_margin("slm", seed, split, meta), platt_margin("xgb", seed, split, meta))
            out = pathlib.Path(CFG["out"]) / f"s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"row_id": meta["row_id"], "y_true": meta["y"].to_numpy(),
                          "score": hybrid_score(x, s, a, b), "margin": margin,
                          "source": np.where(inside, "slm", "xgb")}).to_parquet(out / f"preds_{split}.parquet", index=False)
            sessions = pd.Series(inside).groupby(meta["session_id"].to_numpy()).any()
            shares.append({"split": split, "seed": seed, "rows": int(len(x)), "rows_to_slm": int(inside.sum()),
                           "row_share": float(inside.mean()), "sessions": int(len(sessions)),
                           "sessions_with_a_row_to_slm": int(sessions.sum()),
                           "session_share": float(sessions.mean())})
            print(f"{split} s{seed}: {shares[-1]}", flush=True)
    (pathlib.Path(CFG["out"]) / "build.json").write_text(json.dumps(
        {"band": {"a": a, "b": b}, "test_ids_sha256": checksum, "shares": shares,
         "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["select", "sensitivity", "build"])
    args = p.parse_args()
    {"select": select, "sensitivity": sensitivity, "build": build}[args.command]()


if __name__ == "__main__":
    main()
