#!/usr/bin/env bash
# Text XGBoost (Comparison 2): 50-trial Optuna search, then seeds 1-3 scored on validation.
# Same search as the structured model (python run.py xgb --tune): objective row-level
# validation PR-AUC, same space, TPE seed, median pruning and early stopping.
# Run from the repo root, in the background:
#   nohup bash scripts/run_xgb_text.sh > logs/xgb_text.log 2>&1 &
# Progress: tail -f logs/xgb_text.log (one line per trial). The Optuna study is stored in
# results/xgb_text_tuning/optuna.db, so rerunning resumes the search where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
echo "$(date '+%F %T') start: text XGBoost search (50 trials)"
$PY run.py xgb --variant text --tune --overwrite-config
echo "$(date '+%F %T') search done; fitting seeds 1-3 (validation predictions only)"
$PY run.py xgb --variant text --fit
echo "$(date '+%F %T') XGB TEXT FINISHED"
