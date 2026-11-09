# Methods notes

Points to carry into the paper's methods section. Details and reasons are in
decisions_log.md.

- **XGBoost search space:** max_depth is capped at 14 by design. The v3 best trial chose
  14, the bound, and the range was not widened further.
- **Model selection and reporting:** model selection for both XGBoost and the SLM uses
  row-level validation PR-AUC. Reported metrics are session-level (primary) and
  row-level (secondary).
