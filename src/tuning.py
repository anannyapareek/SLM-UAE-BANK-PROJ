"""Shared Optuna setup for every tuning run: SQLite storage and one line per trial.

The study is stored in results/<run_id>/optuna.db, so trials survive a crash and
progress can be read from another terminal. Rerunning resumes the same study.
"""
import os
import pathlib
import subprocess
import time

import optuna


def create_study(run_dir, seed, n_startup_trials=5, n_warmup_steps=0):
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=run_dir.name, storage=f"sqlite:///{run_dir / 'optuna.db'}",
        load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                           n_warmup_steps=n_warmup_steps))


def trial_seconds(study):
    """Total time of all finished trials, including ones from earlier, resumed runs."""
    return sum(t.duration.total_seconds() for t in study.trials
               if t.state.is_finished() and t.duration)


def remaining(study, n_trials):
    finished = [t for t in study.trials if t.state.is_finished()]
    return max(n_trials - len(finished), 0)


class TrialPrinter:
    """Prints trial number, validation PR-AUC, trees, pruned and elapsed time, flushed."""

    def __init__(self):
        self.started = time.perf_counter()

    def __call__(self, study, trial):
        pruned = trial.state == optuna.trial.TrialState.PRUNED
        trees = trial.user_attrs.get("n_trees", len(trial.intermediate_values))
        value = "-" if trial.value is None else f"{trial.value:.4f}"
        seconds = trial.duration.total_seconds() if trial.duration else float("nan")
        print(f"trial {trial.number:3d} | valid PR-AUC {value:>6} | trees {trees:5d} | "
              f"pruned {'yes' if pruned else 'no '} | {seconds:6.1f} s | "
              f"total {time.perf_counter() - self.started:7.1f} s", flush=True)


def machine():
    """Core counts, for logging alongside tuning time."""
    def sysctl(key):
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        return int(out) if out.isdigit() else None
    return {"logical_cores": os.cpu_count(), "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
            "efficiency_cores": sysctl("hw.perflevel1.physicalcpu")}


class Usage:
    """Wall-clock and average busy cores (process CPU time / wall time) for a block."""

    def __enter__(self):
        self.wall, self.cpu = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall_seconds = time.perf_counter() - self.wall
        busy = (time.process_time() - self.cpu) / self.wall_seconds if self.wall_seconds > 1 else None
        self.average_busy_cores = None if busy is None else round(busy, 1)
