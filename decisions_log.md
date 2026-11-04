# Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-04 | Keep everything under `data/` out of git; no files in `data/` are tracked. Folders are recreated by the scripts | Raw r4.2 cannot be redistributed and is too large for git; processed data can be regenerated from `src/` |
| 2026-11-04 | Standalone git repo in `SLM-UAE-BANK/` | The enclosing git repo is rooted at the home directory and holds unrelated work |
| 2026-11-04 | Python 3.12 via uv in .venv | Library support; Homebrew unsupported on macOS 14 |
| 2026-11-04 | SLM stages on GPU; data, baselines and evaluation on Mac | Speed; one evaluation script for all models |
| 2026-11-04 | Experiments freeze 21 Nov; final compute 24 Nov | First draft due 25 Nov, paper due 6 Dec |
| 2026-11-04 | No removable-media field in r4.2 file.csv; proxy = file events between a USB Connect and the next Disconnect (same user, PC); if no Disconnect, until session end | Field absent in r4.2 |
| 2026-11-04 | Content fields are generated keyword strings, not natural text | Limits text-based SLM advantage; state in limitations |
| 2026-11-04 | Tenure = months since user first appears in monthly LDAP | LDAP has no hire date |
| 2026-11-04 | External email = any to/cc/bcc recipient outside @dtaa.com | DTAA is the simulated company domain |
| 2026-11-04 | Psychometric table not used | Not in the feature design |
| 2026-11-04 | LDAP user key is user_id; event tables use user | Join on user = user_id |
| 2026-11-04 | All DuckDB scripts run single-threaded (threads=1) | Multi-threaded builds corrupted id data in http_content.parquet twice; single-threaded builds were clean. Cause suspected, not confirmed |
| 2026-11-04 | check_processed.py reads every id and content value, not just row counts | Footer row counts did not detect the corruption |