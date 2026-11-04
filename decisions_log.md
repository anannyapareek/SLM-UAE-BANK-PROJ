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
| 2026-11-04 | Session = Logon to next Logoff (same user, PC); with no Logoff before the next Logon, end at the earlier of next Logon and start + session_cap_hours (12) | Logon has 86,340 sessions with no Logoff. A real Logoff is kept even past 12h: 7,350 sessions (1.56%), max 13.72h. The cap never fired on r4.2 (longest Logon-to-Logon gap is 6h) |
| 2026-11-04 | Orphan Logoffs are counted and dropped | 17 of 384,268 Logoffs have no open session; each follows another Logoff by about 1 to 5 days, which suggests a missing Logon record |
| 2026-11-04 | Orphan events (session_id NULL) are dropped from session features but kept in *_sessioned.parquet for the per-user behavioural baseline | 1,916 events: device 26 (0.0064%), file 31 (0.0070%), email 154 (0.0059%), http 1,705 (0.0060%). 1,776 of them fall just before an orphan Logoff, so most belong to sessions with a missing Logon |
| 2026-11-04 | Session `end` and `end_type` must never become features | They use future information; they may only define the session and the final scoring point |
| 2026-11-04 | 0 of 7,323 malicious events are among the 1,916 dropped orphan events and 17 dropped orphan Logoffs; the drop rule is safe | Checked by event id against all r4.2 answer files (docs/orphan_check.txt); every answer id was found in the processed tables |
| 2026-11-04 | Sessions with a real Logoff are not capped; 7,350 (1.56%) exceed 12 h, max 13.72 h | Cap applies only to missing Logoffs; real end times kept |
| 2026-11-04 | 12 h cap never triggers in r4.2 (max Logon-to-Logon gap 6 h; every user/PC ends with a Logoff) | Rule kept for robustness; noted for methods |
