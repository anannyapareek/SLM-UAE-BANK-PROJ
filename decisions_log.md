# Decisions log

| date | decision | reason |
|------|----------|--------|
| 2026-11-04 | Keep everything under `data/` out of git (only `.gitkeep` files are tracked) | Raw r4.2 and the parquet outputs are too large for git, and processed data can be regenerated from `src/` |
| 2026-11-04 | Standalone git repo in `SLM-UAE-BANK/` | The enclosing git repo is rooted at the home directory and holds unrelated work |
