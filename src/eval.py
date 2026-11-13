"""The only evaluation code for every model.

    python run.py eval single --name <model> --split <valid|calib|test> --preds <p1> [<p2> ...] --out <json>
    python run.py eval paired --split <split> --name-a A --a <p...> --name-b B --b <p...> --out <json>

Each preds file is one seed: row_id, y_true, score (margin optional). User, session,
score_time, scenario and heldout_user are joined from splits.parquet and labels.parquet.
Metrics are computed per seed and averaged; 95% CIs come from a user-level bootstrap,
stratified into users with and without a positive unit, using integer weights rather
than copied rows. --split test is refused unless --allow-test is passed.

--session-agg sets the session score: max (default) = the maximum over all of the session's
scoring points (hourly scoring); final = the score at its last scoring point only
(logoff-only scoring). The session label is "any positive row" either way.

--rows sets which scoring rows a preds file must cover: all (default) = every row of the
split; final = only the split's final (logoff) rows, is_final in features.parquet, for
models scored logoff-only (Comparison 2). The coverage check, units and metrics then use
those rows; test_ids.csv is still verified in full.
"""
import argparse
import json
import os
import pathlib
import time

import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.splits import verify_test_checksum

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
CONFIG = pathlib.Path("configs/eval.yaml")
UNITS = ["session", "row"]
CI_METRICS = ["pr_auc", "recall_at_fpr_0.01", "recall_at_fpr_0.05", "precision_at_k", "recall_at_k"]


# Weighted metrics. Integer weights act exactly like copying units that many times.

class Scored:
    """One seed's scores for a fixed set of units, pre-sorted for fast weighted metrics."""

    def __init__(self, score, y, day, key):
        self.score = np.asarray(score, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.desc = np.argsort(-self.score, kind="stable")
        sorted_scores = self.score[self.desc]
        self.group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        self.by_day = np.lexsort((key, -self.score, day))
        day_sorted = np.asarray(day)[self.by_day]
        self.day_start = np.r_[True, day_sorted[1:] != day_sorted[:-1]]
        self.asc = np.lexsort((key, self.score))
        self.probabilities = bool(self.score.min() >= 0 and self.score.max() <= 1)


def curve(s, w):
    """Cumulative weighted TP and FP at each distinct score, highest score first."""
    y, ws = s.y[s.desc], w[s.desc]
    tp, fp = np.cumsum(ws * y)[s.group_end], np.cumsum(ws * (1 - y))[s.group_end]
    keep = (tp + fp) > 0
    return tp[keep], fp[keep], s.score[s.desc][s.group_end][keep]


def average_precision(s, w):
    tp, fp, _ = curve(s, w)
    if not len(tp) or tp[-1] == 0:
        return np.nan
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (tp + fp)))


def recall_at_fpr(s, w, level):
    """Highest TPR on the ROC curve with FPR <= level, and the score threshold giving it."""
    tp, fp, thresholds = curve(s, w)
    if not len(tp) or tp[-1] == 0 or fp[-1] == 0:
        return np.nan, np.nan
    ok = np.flatnonzero(fp / fp[-1] <= level)
    if not len(ok):
        return 0.0, np.inf
    best = ok[np.argmax(tp[ok])]
    return float(tp[best] / tp[-1]), float(thresholds[best])


def top_k_per_day(s, w, k):
    """Precision and recall when each day alerts on its top k units (by weight)."""
    ws, y = w[s.by_day], s.y[s.by_day]
    cumulative = np.cumsum(ws)
    day_offset = np.maximum.accumulate(np.where(s.day_start, cumulative - ws, 0))
    taken = np.clip(k - (cumulative - day_offset - ws), 0, ws)
    alerts, hits, positives = taken.sum(), (taken * y).sum(), (ws * y).sum()
    precision = hits / alerts if alerts else np.nan
    recall = hits / positives if positives else np.nan
    return float(precision), float(recall)


def brier(s, w):
    return float(np.sum(w * (s.score - s.y) ** 2) / w.sum())


def calibration_bins(s, w, bins, equal_mass):
    """Per-bin weight, mean confidence and accuracy; equal-mass bins split units fractionally."""
    if equal_mass:
        ws = w[s.asc]
        end = np.cumsum(ws)
        start = end - ws
        edges = np.linspace(0, end[-1], bins + 1)
        share = np.clip(np.minimum(end[:, None], edges[None, 1:])
                        - np.maximum(start[:, None], edges[None, :-1]), 0, None)
        weight = share.sum(axis=0)
        confidence = (share * s.score[s.asc][:, None]).sum(axis=0)
        correct = (share * s.y[s.asc][:, None]).sum(axis=0)
    else:
        index = np.minimum((s.score * bins).astype(int), bins - 1)
        weight = np.bincount(index, w, bins)
        confidence = np.bincount(index, w * s.score, bins)
        correct = np.bincount(index, w * s.y, bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return weight, confidence / weight, correct / weight


def ece(s, w, bins, equal_mass):
    weight, confidence, accuracy = calibration_bins(s, w, bins, equal_mass)
    used = weight > 0
    return float(np.sum(weight[used] * np.abs(accuracy[used] - confidence[used])) / weight.sum())


def core_metrics(s, w, cfg):
    """The metrics that get bootstrap CIs."""
    out = {"pr_auc": average_precision(s, w)}
    for level in cfg["fpr_levels"]:
        out[f"recall_at_fpr_{level}"] = recall_at_fpr(s, w, level)[0]
    out["precision_at_k"], out["recall_at_k"] = top_k_per_day(s, w, cfg["alerts_per_day"])
    return out


# Units

ROWS = ("all", "final")


def load_meta(con, split, rows="all"):
    """The split's scoring rows the predictions must cover: all, or final (logoff) rows only."""
    if rows not in ROWS:
        raise ValueError(f"rows {rows!r}: use one of {ROWS}")
    final = (f"JOIN '{P}/features.parquet' f USING (row_id)" if rows == "final" else "")
    where = " AND f.is_final" if rows == "final" else ""
    return con.execute(f"""
        SELECT x.row_id, x.session_id, x."user", x.heldout_user, l.score_time, l.scenario, l.y
        FROM '{P}/splits.parquet' x JOIN '{P}/labels.parquet' l USING (row_id) {final}
        WHERE x.split = '{split}'{where}
    """).df()


def load_preds(paths, meta):
    """Seed score arrays aligned to meta's row order, after checking rows and labels."""
    scores, reference = [], None
    for path in paths:
        preds = pd.read_parquet(path, columns=["row_id", "y_true", "score"])
        if preds["row_id"].duplicated().any():
            raise RuntimeError(f"{path}: duplicate row_ids")
        preds = preds.set_index("row_id")
        if reference is None:
            reference = preds["y_true"]
        elif set(preds.index) != set(reference.index) or not preds["y_true"].reindex(
                reference.index).equals(reference):
            raise RuntimeError(f"{path}: row_ids or labels differ from {paths[0]}")
        if set(preds.index) != set(meta["row_id"]):
            raise RuntimeError(f"{path}: rows do not match the split ({len(preds):,} vs {len(meta):,})")
        aligned = preds.reindex(meta["row_id"])
        if not (aligned["y_true"].to_numpy() == meta["y"].to_numpy()).all():
            raise RuntimeError(f"{path}: y_true disagrees with labels.parquet")
        scores.append(aligned["score"].to_numpy(dtype=float))
    return scores


SESSION_AGG = {"max": "max", "final": "last"}


def aggregate_sessions(meta, scores, how="max"):
    """Session units, positive if any row is. Score: max over the session's rows (how="max",
    hourly scoring) or the score at its last scoring point (how="final", logoff-only)."""
    if how not in SESSION_AGG:
        raise ValueError(f"session aggregation {how!r}: use one of {sorted(SESSION_AGG)}")
    frame = meta[["session_id", "user", "heldout_user", "score_time", "scenario", "y"]].copy()
    for i, s in enumerate(scores):
        frame[f"s{i}"] = s
    if how == "final":
        if frame.duplicated(["session_id", "score_time"]).any():
            raise RuntimeError("two scoring points share a session and score_time")
        frame = frame.sort_values(["session_id", "score_time"], kind="stable")
    grouped = frame.groupby("session_id", sort=True)
    units = grouped.agg(user=("user", "first"), heldout_user=("heldout_user", "first"),
                        last_time=("score_time", "max"), scenario=("scenario", "max"),
                        y=("y", "max"),
                        **{f"s{i}": (f"s{i}", SESSION_AGG[how]) for i in range(len(scores))})
    units = units.reset_index()
    return units, [units[f"s{i}"].to_numpy() for i in range(len(scores))]


def build_units(meta, scores, unit, cfg):
    if unit == "row":
        units = meta.rename(columns={"score_time": "last_time"})
        ids, unit_scores = units["row_id"].astype(str), scores
    else:
        units, unit_scores = aggregate_sessions(meta, scores, cfg.get("session_agg", "max"))
        ids = units["session_id"].astype(str)
    # A unit is dated by its last scoring point, when its final score is known.
    day = units["last_time"].dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)
    key = pd.util.hash_array(ids.to_numpy(), hash_key=f"{cfg['tie_break_seed']:016d}")
    users, user_index = np.unique(units["user"].to_numpy(), return_inverse=True)
    scored = [Scored(s, units["y"].to_numpy(), day, key) for s in unit_scores]
    return {"units": units, "scored": scored, "users": users, "user_index": user_index,
            "day": day}


def precision_ceiling(u, k):
    """Best possible precision at k: mean over days of min(1, positives that day / k)."""
    positives = np.bincount(np.unique(u["day"], return_inverse=True)[1],
                            weights=u["units"]["y"].to_numpy())
    return float(np.mean(np.minimum(1.0, positives / k)))


def counts(u):
    units, y = u["units"], u["units"]["y"].to_numpy()
    positive_users = units.loc[y == 1, "user"].nunique()
    return {"n_units": int(len(units)), "n_pos": int(y.sum()), "n_users": int(len(u["users"])),
            "n_pos_users": int(positive_users), "positive_rate": float(y.mean()),
            "n_days": int(len(np.unique(u["day"])))}


def per_seed_detail(u, cfg, split):
    """Point metrics and extras for each seed, all units weighted 1."""
    units = u["units"]
    scenario = units["scenario"].astype(float).fillna(0).to_numpy()
    scenarios = sorted(int(x) for x in np.unique(scenario[units["y"].to_numpy() == 1]) if x > 0)
    out = []
    for s in u["scored"]:
        w = np.ones(len(s.score))
        m = core_metrics(s, w, cfg)
        m["thresholds"] = {f"fpr_{level}": recall_at_fpr(s, w, level)[1] for level in cfg["fpr_levels"]}
        threshold = recall_at_fpr(s, w, cfg["scenario_fpr"])[1]
        m["recall_by_scenario"] = {
            str(sc): float((s.score[(s.y == 1) & (scenario == sc)] >= threshold).mean())
            for sc in scenarios}
        if s.probabilities:
            m["brier"] = brier(s, w)
            for name, mass in (("equal_mass", True), ("equal_width", False)):
                weight, confidence, accuracy = calibration_bins(s, w, cfg["ece_bins"], mass)
                m[f"ece_{name}"] = ece(s, w, cfg["ece_bins"], mass)
                m[f"reliability_{name}"] = {"count": weight.tolist(),
                                            "mean_confidence": confidence.tolist(),
                                            "accuracy": accuracy.tolist()}
        if split == "test":
            m["heldout_vs_seen"] = {}
            for name, mask in (("heldout", units["heldout_user"].to_numpy()),
                               ("seen", ~units["heldout_user"].to_numpy())):
                m["heldout_vs_seen"][name] = core_metrics(s, mask.astype(float), cfg)
        out.append(m)
    return out


def summarise(per_seed):
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], float)]
    values = {k: np.array([m[k] for m in per_seed], dtype=float) for k in keys}
    mean = {k: float(np.nanmean(v)) for k, v in values.items()}
    std = {k: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for k, v in values.items()}
    by_scenario = {sc: float(np.mean([m["recall_by_scenario"][sc] for m in per_seed]))
                   for sc in per_seed[0]["recall_by_scenario"]}
    return mean, std, by_scenario


# Bootstrap

def user_strata(u):
    positive = np.zeros(len(u["users"]), bool)
    positive[u["user_index"][u["units"]["y"].to_numpy() == 1]] = True
    return np.flatnonzero(positive), np.flatnonzero(~positive)


def draw_weights(rng, strata, u):
    """Stratified user resample, as unit weights: each user's draw count."""
    draws = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
    per_user = np.bincount(draws, minlength=len(u["users"]))
    return per_user[u["user_index"]].astype(float)


def resample_batch(seeds, models, cfg):
    """For each resample: per model, each metric averaged over its seeds."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        weights = {}
        result = {}
        for name, u in models.items():
            if u["unit_key"] not in weights:
                weights[u["unit_key"]] = draw_weights(rng, u["strata"], u)
            w = weights[u["unit_key"]]
            per_seed = [core_metrics(s, w, cfg) for s in u["scored"]]
            result[name] = {k: float(np.mean([m[k] for m in per_seed])) for k in CI_METRICS}
        out.append(result)
    return out


def bootstrap(models, cfg, n_jobs=-1):
    """Resample results per model; every model shares the same user draws."""
    boot = cfg["bootstrap"]
    seeds = np.random.SeedSequence(boot["seed"]).spawn(boot["resamples"])
    batches = np.array_split(np.arange(len(seeds)), max(1, min(len(seeds), os.cpu_count() or 1)))
    results = Parallel(n_jobs=n_jobs)(
        delayed(resample_batch)([seeds[i] for i in batch], models, cfg) for batch in batches)
    return [r for batch in results for r in batch]


def interval(values, ci):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [50 * (1 - ci), 100 - 50 * (1 - ci)]) if len(finite) else (np.nan, np.nan)
    return [float(low), float(high)], int(len(values) - len(finite))


# Commands

def prepare(meta, paths, unit, cfg):
    u = build_units(meta, load_preds(paths, meta), unit, cfg)
    u["strata"] = user_strata(u)
    u["unit_key"] = unit
    return u


def single(args, cfg, meta):
    started = time.perf_counter()
    out = {"name": args.name, "split": args.split, "primary_unit": cfg["unit"], "preds": args.preds,
           "config": cfg}
    skipped_total = 0
    for unit in UNITS:
        u = prepare(meta, args.preds, unit, cfg)
        per_seed = per_seed_detail(u, cfg, args.split)
        mean, std, by_scenario = summarise(per_seed)
        resamples = bootstrap({args.name: u}, cfg)
        ci, skipped = {}, {}
        for k in CI_METRICS:
            ci[k], skipped[k] = interval([r[args.name][k] for r in resamples], cfg["bootstrap"]["ci"])
        skipped_total += max(skipped.values())
        out[unit] = {"counts": counts(u), "mean": mean, "std": std, "ci": ci,
                     "precision_at_k_ceiling": precision_ceiling(u, cfg["alerts_per_day"]),
                     "recall_by_scenario": by_scenario, "boot_skipped": skipped,
                     "per_seed": per_seed}
    out["boot_skipped"] = skipped_total
    out["runtime_seconds"] = round(time.perf_counter() - started, 1)
    return out


def paired(args, cfg, meta):
    started = time.perf_counter()
    out = {"name_a": args.name_a, "name_b": args.name_b, "split": args.split,
           "primary_unit": cfg["unit"], "preds_a": args.a, "preds_b": args.b, "config": cfg}
    skipped_total = 0
    for unit in UNITS:
        a, b = prepare(meta, args.a, unit, cfg), prepare(meta, args.b, unit, cfg)
        point = {}
        for name, u in (("a", a), ("b", b)):
            per_seed = [core_metrics(s, np.ones(len(s.score)), cfg) for s in u["scored"]]
            point[name] = {k: float(np.mean([m[k] for m in per_seed])) for k in CI_METRICS}
        resamples = bootstrap({"a": a, "b": b}, cfg)
        result = {}
        for k in CI_METRICS:
            ci, skipped = interval([r["a"][k] - r["b"][k] for r in resamples], cfg["bootstrap"]["ci"])
            skipped_total = max(skipped_total, skipped)
            result[k] = {"a": point["a"][k], "b": point["b"][k],
                         "difference": point["a"][k] - point["b"][k], "ci": ci,
                         "significant": bool(ci[0] > 0 or ci[1] < 0), "boot_skipped": skipped}
        result["precision_at_k_ceiling"] = precision_ceiling(a, cfg["alerts_per_day"])
        out[unit] = result
    out["boot_skipped"] = skipped_total
    out["runtime_seconds"] = round(time.perf_counter() - started, 1)
    return out


def parse(argv):
    parser = argparse.ArgumentParser(prog="python run.py eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("single", "paired"):
        p = sub.add_parser(mode)
        p.add_argument("--split", required=True, choices=["valid", "calib", "test"])
        p.add_argument("--out", required=True)
        p.add_argument("--allow-test", action="store_true",
                       help="permit --split test (Comparison 1 onwards only)")
        p.add_argument("--session-agg", choices=sorted(SESSION_AGG), default="max",
                       help="session score: max over scoring points (hourly, default) or "
                            "the final scoring point only (logoff-only)")
        p.add_argument("--rows", choices=ROWS, default="all",
                       help="rows the preds must cover: all (default) or final (logoff) rows only")
        if mode == "single":
            p.add_argument("--name", required=True)
            p.add_argument("--preds", required=True, nargs="+")
        else:
            p.add_argument("--name-a", required=True)
            p.add_argument("--a", required=True, nargs="+")
            p.add_argument("--name-b", required=True)
            p.add_argument("--b", required=True, nargs="+")
    return parser.parse_args(argv)


def to_json(value):
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_baselines():
    """The validation evals for the baselines, with names and paths from configs/eval.yaml."""
    cfg = yaml.safe_load(CONFIG.read_text())["baselines"]
    split, out = cfg["split"], pathlib.Path(cfg["out_dir"])
    for name, preds in cfg["models"].items():
        main(["single", "--name", name, "--split", split, "--preds", *preds,
              "--out", str(out / f"{name}_{split}.json")])
    for a, b in cfg["paired"]:
        main(["paired", "--split", split, "--name-a", a, "--a", *cfg["models"][a],
              "--name-b", b, "--b", *cfg["models"][b], "--out", str(out / f"{a}_vs_{b}_{split}.json")])


def main(argv):
    checksum = verify_test_checksum()
    args = parse(argv)
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("refusing to evaluate on test without --allow-test")
    # The baselines list names runs, not metric settings, so it is not recorded per eval.
    cfg = {k: v for k, v in yaml.safe_load(CONFIG.read_text()).items() if k != "baselines"}
    if args.session_agg != "max":            # recorded only when not the default
        cfg["session_agg"] = args.session_agg
    con = duckdb.connect(config={"threads": 1})
    meta = load_meta(con, args.split, args.rows)
    out = single(args, cfg, meta) if args.mode == "single" else paired(args, cfg, meta)
    out["test_ids_sha256"] = checksum        # always the full frozen test_ids.csv
    if args.rows != "all":                   # recorded only when not the default
        out["rows"] = args.rows
        out["n_units"] = {"row": int(len(meta)), "session": int(meta["session_id"].nunique())}
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(out), indent=2) + "\n")
    print(f"wrote {path} in {out['runtime_seconds']} s (boot_skipped {out['boot_skipped']})")
    return out
