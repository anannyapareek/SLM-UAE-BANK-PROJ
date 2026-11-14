"""Step 7.1 calibration: temperature scaling (primary) and Platt scaling (sensitivity).

Fitted on the calibration split only (configs/step7.yaml), on each model's margin (SLM: yes-no
logit; XGBoost: output_margin). Fit units (decisions log 2026-11-14):
- session: each calib session's max margin, label = any positive row (C1 primary unit);
- row: calib rows (C1 secondary view, and the scale the hybrid band acts on).
C2 files hold one logoff row per session, so their single fit serves both.

    temperature: p = sigmoid(m / T), T minimises the NLL on calib, minimize_scalar bounds
                 (0.05, 20), method "bounded" (runbook 7.1)
    platt:       p = sigmoid(a * m + b), (a, b) minimise the NLL on calib (a > 0 asserted)

Both maps are strictly increasing, so applying the session-fit parameters to every row and
taking eval.py's max over a session gives exactly the calibrated session-max probability,
and ranking metrics are unchanged. Writes, per model, fit unit, method and seed,
results/step7/calibration/<model>_<unit>fit_<method>/s<seed>/preds_<split>[_final].parquet
(row_id, y_true, score = calibrated probability, margin = original margin) and params.json.
Computes no metrics.

Run: python -m src.calibrate
"""
import datetime
import json
import pathlib
import platform
import time

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize, minimize_scalar

from src.splits import verify_test_checksum

CONFIG = pathlib.Path("configs/step7.yaml")
P = pathlib.Path("data/processed")


def nll(z, y):
    """Mean negative log-likelihood of sigmoid(z) for labels y, computed stably."""
    return float(np.mean(np.where(y == 1, np.logaddexp(0, -z), np.logaddexp(0, z))))


def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def fit_temperature(m, y, bounds):
    result = minimize_scalar(lambda t: nll(m / t, y), bounds=tuple(bounds), method="bounded")
    return {"T": float(result.x), "nll": float(result.fun), "converged": bool(result.success),
            "at_bound": bool(min(abs(result.x - b) for b in bounds) < 1e-3)}


def fit_platt(m, y):
    result = minimize(lambda ab: nll(ab[0] * m + ab[1], y), x0=np.array([1.0, 0.0]),
                      method="L-BFGS-B")
    a, b = (float(v) for v in result.x)
    if a <= 0:
        raise RuntimeError(f"Platt slope a = {a} <= 0: the map would not preserve the ranking")
    return {"a": a, "b": b, "nll": float(result.fun), "converged": bool(result.success)}


def apply(method, params, m):
    if method == "temperature":
        return sigmoid(m / params["T"])
    return sigmoid(params["a"] * m + params["b"])


def fit_data(preds, unit, sessions):
    """Margins and labels at the fit unit."""
    if unit == "row":
        return preds["margin"].to_numpy(float), preds["y_true"].to_numpy(int)
    d = preds.merge(sessions, on="row_id")
    g = d.groupby("session_id").agg(m=("margin", "max"), y=("y_true", "max"))
    return g["m"].to_numpy(float), g["y"].to_numpy(int)


def main():
    checksum = verify_test_checksum()
    cfg = yaml.safe_load(CONFIG.read_text())["calibration"]
    con = duckdb.connect(config={"threads": 1})
    sessions = con.execute(f"""SELECT row_id, session_id FROM '{P}/splits.parquet'
                               WHERE split = '{cfg['fit_split']}'""").df()
    out_root = pathlib.Path(cfg["out"])
    records = []
    for model in cfg["models"]:
        for seed in cfg["seeds"]:
            folder = pathlib.Path(model["folder"].format(seed=seed))
            calib = pd.read_parquet(folder / f"preds_{cfg['fit_split']}{model['suffix']}.parquet")
            for unit in model["fit_units"]:
                m, y = fit_data(calib, unit, sessions)
                for method in cfg["methods"]:
                    started = time.perf_counter()
                    params = (fit_temperature(m, y, cfg["temperature_bounds"])
                              if method == "temperature" else fit_platt(m, y))
                    run_id = f"{model['key']}_{unit}fit_{method}"
                    out = out_root / run_id / f"s{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    for split in cfg["splits"]:
                        name = f"preds_{split}{model['suffix']}.parquet"
                        d = pd.read_parquet(folder / name, columns=["row_id", "y_true", "score", "margin"])
                        d = d.assign(score=apply(method, params, d["margin"].to_numpy(float)))
                        d.to_parquet(out / name, index=False)
                    record = {"run_id": run_id, "model": model["key"], "comparison": model["comparison"],
                              "fit_unit": unit, "method": method, "seed": seed,
                              "fit_split": cfg["fit_split"], "fit_units_n": int(len(m)),
                              "fit_positives": int(y.sum()), "params": params,
                              "source": str(folder), "test_ids_sha256": checksum,
                              "fit_seconds": round(time.perf_counter() - started, 2),
                              "date": datetime.datetime.now().isoformat(timespec="seconds"),
                              "machine": f"{platform.node()} ({platform.machine()}, {platform.system()})"}
                    (out / "params.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    shown = (f"T = {params['T']:.4f}{' (AT BOUND)' if params['at_bound'] else ''}"
                             if method == "temperature" else f"a = {params['a']:.4f}, b = {params['b']:.4f}")
                    print(f"{run_id} s{seed}: {shown}; fitted on {len(m):,} calib {unit}s "
                          f"({int(y.sum())} positive)", flush=True)
    (out_root / "fits.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} fits to {out_root}")


if __name__ == "__main__":
    main()
