# Evaluation plan

Planned for src/eval.py (not written yet).

- **Per-scenario recall:** at each chosen threshold, recall broken down by scenario, computed
  on positive rows only (y = 1), using `scenario` from labels.parquet. With 227, 7,680 and
  160 positive rows for scenarios 1–3, the overall recall is dominated by scenario 2, so
  report all three separately.

Note: src/eval/ is an empty package, which would stop src/eval.py from importing. Remove it
when eval.py is written.
