"""XGBoost on the structured features.

    python run.py xgb --tune   50-trial Optuna search; writes configs/xgb_tuned_<run>.yaml
                               (and configs/xgb_best.yaml only with --overwrite-config)
    python run.py xgb [--fit]  fit seeds 1-3 from configs/xgb_best.yaml (the default)
    python run.py xgb --score --splits valid calib test --allow-test
                               score the saved seed models (results/<run_prefix>_s<seed>/
                               model.json) on these splits; predictions only

Tuning and fitting load train and valid rows only. --score loads one split at a time, with
its labels only to write y_true into the predictions file (eval.py's input); nothing is
computed from them here. Test needs --allow-test. Every scoring point (hourly row) is scored,
so either session aggregation can be computed later. A predictions file that already exists
must be reproduced exactly; it is never overwritten.
"""
import datetime
import json
import os
import pathlib

import duckdb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from optuna_integration import XGBoostPruningCallback
from sklearn.metrics import average_precision_score, roc_auc_score

from src import tuning
from src.splits import git_commit, verify_test_checksum

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
FEATURES_CONFIG = pathlib.Path("configs/features.yaml")
BEST_CONFIG = pathlib.Path("configs/xgb_best.yaml")
RESULTS = pathlib.Path("results")
RUN_LOG = RESULTS / "run.csv"
LOG = pathlib.Path("decisions_log.md")
SPLITS_USED = ("train", "valid")
CATEGORICAL = ["role", "department", "functional_unit"]
N_TRIALS = 50
TUNING_SEED = 1
# v2: max_depth widened to 3-14 (v1 found its best at the old upper bound, 10).
# v3: early stopping patience 200 and no pruning before 100 boosting rounds.
RUN_PREFIX = "xgb_struct_v3"
TUNING_RUN = f"{RUN_PREFIX}_tuning"
MAX_DEPTH = (3, 14)
N_JOBS = -1  # XGBoost: -1 means one thread per logical core
SEEDS = [1, 2, 3]
MAX_TREES = 2000
EARLY_STOPPING = 200
PRUNER = {"n_startup_trials": 5, "n_warmup_steps": 100}
RUN_COLUMNS = ["timestamp", "run_id", "model", "features", "seed", "n_train", "n_valid",
               "valid_positive_rate", "best_iteration", "valid_pr_auc", "valid_roc_auc",
               "params", "test_ids_sha256", "git_commit", "status"]


def load(con, names):
    """Train and valid rows only; the split filter keeps test labels out of memory."""
    columns = ", ".join(f'f."{n}"' for n in names)
    frame = con.execute(f"""
        SELECT f.row_id, f."user", x.split, l.y, {columns}
        FROM '{P}/features.parquet' f
        JOIN '{P}/splits.parquet' x USING (row_id)
        JOIN '{P}/labels.parquet' l USING (row_id)
        WHERE x.split IN {SPLITS_USED}
        ORDER BY x.split, f.row_id
    """).df()
    if set(frame["split"]) != set(SPLITS_USED):
        raise RuntimeError(f"expected only {SPLITS_USED}, got {sorted(set(frame['split']))}")
    categories = {c: sorted(frame[c].unique()) for c in CATEGORICAL if c in names}
    for column, values in categories.items():
        frame[column] = pd.Categorical(frame[column], categories=values)
    parts = {s: frame[frame["split"] == s].reset_index(drop=True) for s in SPLITS_USED}
    return parts, categories


def load_split(con, names, split, categories):
    """One split's rows, sorted by row_id, with the training categorical encoding."""
    columns = ", ".join(f'f."{n}"' for n in names)
    frame = con.execute(f"""
        SELECT f.row_id, f."user", l.y, {columns}
        FROM '{P}/features.parquet' f
        JOIN '{P}/splits.parquet' x USING (row_id)
        JOIN '{P}/labels.parquet' l USING (row_id)
        WHERE x.split = '{split}'
        ORDER BY f.row_id
    """).df()
    for column, values in categories.items():
        unseen = sorted(set(frame[column]) - set(values))
        if unseen:
            raise RuntimeError(f"{split}: {column} values not seen in training: {unseen}")
        frame[column] = pd.Categorical(frame[column], categories=values)
    return frame


def write_or_match(frame, path):
    """Write predictions, or check that an existing file holds exactly the same."""
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.equals(frame):
            diff = (existing["score"] - frame["score"]).abs().max() if len(existing) == len(frame) else "n/a"
            raise RuntimeError(f"{path} exists and differs from the new predictions "
                               f"(max |score diff| {diff}); not overwritten")
        return "identical to the existing file (kept)"
    frame.to_parquet(path, index=False)
    return "written"


def score_saved(splits, allow_test=False):
    """Predictions of the saved seed models on the given splits (Comparison 1)."""
    if "test" in splits and not allow_test:
        raise SystemExit("scoring the test split needs --allow-test")
    checksum = verify_test_checksum()
    print(f"test_ids.csv SHA-256 {checksum}")
    best = yaml.safe_load(BEST_CONFIG.read_text())
    names = [f["name"] for f in yaml.safe_load(FEATURES_CONFIG.read_text())["features"]]
    categories = {c: list(v) for c, v in best["categories"].items()}
    con = duckdb.connect(config={"threads": 1})
    for split in splits:
        frame = load_split(con, names, split, categories)
        for seed in best["seeds"]:
            folder = RESULTS / f"{best['run_prefix']}_s{seed}"
            model = xgb.XGBClassifier()
            model.load_model(folder / "model.json")
            X = frame[names]
            preds = pd.DataFrame({"row_id": frame["row_id"], "user": frame["user"],
                                  "y_true": frame["y"].to_numpy(),
                                  "score": model.predict_proba(X)[:, 1],
                                  "margin": model.predict(X, output_margin=True)})
            status = write_or_match(preds, folder / f"preds_{split}.parquet")
            print(f"{split} seed {seed}: {len(preds):,} rows, best iteration "
                  f"{model.best_iteration}; {folder / f'preds_{split}.parquet'} {status}")


def classifier(params, seed, scale_pos_weight, callbacks=None):
    return xgb.XGBClassifier(
        tree_method="hist", n_jobs=N_JOBS, objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight, early_stopping_rounds=EARLY_STOPPING,
        n_estimators=MAX_TREES, enable_categorical=True, random_state=seed,
        callbacks=callbacks, **params)


def suggest(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", *MAX_DEPTH),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def search(X, y, scale_pos_weight):
    def objective(trial):
        model = classifier(suggest(trial), TUNING_SEED, scale_pos_weight,
                           [XGBoostPruningCallback(trial, "validation_0-aucpr")])
        model.fit(X["train"], y["train"], eval_set=[(X["valid"], y["valid"])], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("n_trees", int(model.best_iteration) + 1)
        return average_precision_score(y["valid"], model.predict_proba(X["valid"])[:, 1])

    study = tuning.create_study(RESULTS / TUNING_RUN, TUNING_SEED, **PRUNER)
    print(f"optuna study: {RESULTS / TUNING_RUN / 'optuna.db'}", flush=True)
    with tuning.Usage() as usage:
        study.optimize(objective, n_trials=tuning.remaining(study, N_TRIALS),
                       callbacks=[tuning.TrialPrinter()])
    seconds = tuning.trial_seconds(study)
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    best = study.best_trial
    return {
        "trials": len(study.trials), "completed": states.get("COMPLETE", 0),
        "pruned": states.get("PRUNED", 0), "failed": states.get("FAIL", 0),
        "wall_clock_seconds": round(seconds, 1), "best_trial": best.number,
        "best_valid_pr_auc": round(best.value, 6),
        "best_iteration": best.user_attrs.get("best_iteration"), "params": best.params,
        "n_jobs": N_JOBS, "average_busy_cores": usage.average_busy_cores,
        **tuning.machine(),
    }


def append_run(row):
    """Add a run; an earlier current row with the same run_id is marked as replaced."""
    RESULTS.mkdir(exist_ok=True)
    runs = pd.read_csv(RUN_LOG) if RUN_LOG.exists() else pd.DataFrame(columns=RUN_COLUMNS)
    replaced = (runs["run_id"] == row["run_id"]) & (runs["status"] == "current")
    runs.loc[replaced, "status"] = f"replaced by rerun at {row['timestamp']}"
    pd.concat([runs, pd.DataFrame([row], columns=RUN_COLUMNS)]).to_csv(RUN_LOG, index=False)


def train_seed(seed, params, parts, X, y, names, scale_pos_weight, checksum, run_prefix):
    model = classifier(params, seed, scale_pos_weight)
    model.fit(X["train"], y["train"], eval_set=[(X["valid"], y["valid"])], verbose=False)
    score = model.predict_proba(X["valid"])[:, 1]
    margin = model.predict(X["valid"], output_margin=True)
    out = RESULTS / f"{run_prefix}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": parts["valid"]["row_id"], "user": parts["valid"]["user"],
                  "y_true": y["valid"], "score": score, "margin": margin}
                 ).to_parquet(out / "preds_valid.parquet", index=False)
    metrics = {
        "split": "valid", "seed": seed, "rows": int(len(score)),
        "positives": int(y["valid"].sum()), "positive_rate": float(y["valid"].mean()),
        "pr_auc": float(average_precision_score(y["valid"], score)),
        "roc_auc": float(roc_auc_score(y["valid"], score)),
        "best_iteration": int(model.best_iteration), "scale_pos_weight": scale_pos_weight,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model.save_model(out / "model.json")
    append_run({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": out.name, "model": "xgboost", "features": "structured", "seed": seed,
        "n_train": len(y["train"]), "n_valid": len(y["valid"]),
        "valid_positive_rate": round(metrics["positive_rate"], 6),
        "best_iteration": metrics["best_iteration"], "valid_pr_auc": round(metrics["pr_auc"], 6),
        "valid_roc_auc": round(metrics["roc_auc"], 6), "params": str(BEST_CONFIG),
        "test_ids_sha256": checksum, "git_commit": git_commit(), "status": "current"})
    if seed == SEEDS[0]:
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        total = booster.get_score(importance_type="total_gain")
        pd.DataFrame({"feature": names, "gain": [gain.get(n, 0.0) for n in names],
                      "total_gain": [total.get(n, 0.0) for n in names]}
                     ).sort_values("total_gain", ascending=False
                                   ).to_csv(out / "importance.csv", index=False)
    return metrics


def log_budget(budget, scale_pos_weight):
    text = LOG.read_text()
    params = ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                       for k, v in budget["params"].items())
    LOG.write_text(text + ("" if text.endswith("\n") else "\n") +
                   f"| {datetime.date.today()} | XGBoost tuning budget: {budget['trials']} Optuna "
                   f"trials (TPE seed {TUNING_SEED}, median pruning after {PRUNER['n_warmup_steps']} "
                   f"rounds, early stopping {EARLY_STOPPING}), {budget['pruned']} pruned, "
                   f"{budget['wall_clock_seconds'] / 60:.1f} min of trial time on this Mac "
                   f"(n_jobs {budget['n_jobs']}, {budget['logical_cores']} logical cores, "
                   f"{budget['average_busy_cores'] or 'n/a'} busy on average); best trial "
                   f"{budget['best_trial']} (valid PR-AUC {budget['best_valid_pr_auc']:.4f}, "
                   f"{budget['best_iteration'] + 1} trees): {params} | Recorded so the SLM's grid can be "
                   f"compared with it in the methods; scale_pos_weight {scale_pos_weight:.1f} |\n")


def check_against_config(best, scale_pos_weight, categories):
    """The data must match what the configured parameters were tuned on."""
    expected = best["fixed"]["scale_pos_weight"]
    if abs(scale_pos_weight - expected) > 1e-9 * expected:
        raise RuntimeError(f"scale_pos_weight {scale_pos_weight} differs from {BEST_CONFIG}: {expected}")
    if {k: list(v) for k, v in categories.items()} != best["categories"]:
        raise RuntimeError(f"LDAP categories differ from {BEST_CONFIG}")


def config(budget, scale_pos_weight, categories):
    return {
        "model": "xgboost", "features": "structured (configs/features.yaml)",
        "fixed": {"tree_method": "hist", "objective": "binary:logistic", "eval_metric": "aucpr",
                  "n_estimators": MAX_TREES, "early_stopping_rounds": EARLY_STOPPING,
                  "enable_categorical": True, "scale_pos_weight": scale_pos_weight},
        "params": budget["params"], "tuning": {k: v for k, v in budget.items() if k != "params"},
        "categories": categories, "seeds": SEEDS, "run_prefix": RUN_PREFIX,
        "search_space": {"max_depth": list(MAX_DEPTH)}, "pruner": PRUNER,
    }


def run(tune=False, overwrite_config=False):
    checksum = verify_test_checksum()
    names = [f["name"] for f in yaml.safe_load(FEATURES_CONFIG.read_text())["features"]]
    con = duckdb.connect(config={"threads": 1})
    parts, categories = load(con, names)
    X = {s: parts[s][names] for s in SPLITS_USED}
    y = {s: parts[s]["y"].to_numpy() for s in SPLITS_USED}
    scale_pos_weight = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
    print(f"train {len(y['train']):,} rows ({y['train'].mean():.4%} positive); "
          f"valid {len(y['valid']):,} rows ({y['valid'].mean():.4%} positive); "
          f"scale_pos_weight {scale_pos_weight:.1f}")

    if tune:
        budget = search(X, y, scale_pos_weight)
        print(f"tuning: {budget}")
        text = yaml.safe_dump(config(budget, scale_pos_weight, categories), sort_keys=False)
        tuned = BEST_CONFIG.with_name(f"xgb_tuned_{RUN_PREFIX}.yaml")
        tuned.write_text(text)
        print(f"wrote {tuned}")
        if overwrite_config:
            BEST_CONFIG.write_text(text)
            print(f"replaced {BEST_CONFIG}")
        log_budget(budget, scale_pos_weight)
        return

    best = yaml.safe_load(BEST_CONFIG.read_text())
    check_against_config(best, scale_pos_weight, categories)
    print(f"fitting seeds {best['seeds']} from {BEST_CONFIG} ({best['run_prefix']})")
    for seed in best["seeds"]:
        m = train_seed(seed, best["params"], parts, X, y, names, scale_pos_weight, checksum,
                       best["run_prefix"])
        print(f"seed {seed}: valid PR-AUC {m['pr_auc']:.4f}, ROC-AUC {m['roc_auc']:.4f}, "
              f"best iteration {m['best_iteration']} ({m['best_iteration'] + 1} trees); "
              f"valid positive rate {m['positive_rate']:.4%}")
