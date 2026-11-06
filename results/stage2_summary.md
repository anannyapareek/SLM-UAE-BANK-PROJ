# Stage 2 summary

- **Sessions:** 384,251 (86,340 Logons are unlocks; all end at a Logoff, the 12 h cap never fired; 9,605 or 2.50% run over 12 h, max 13.72 h).
- **Scoring rows:** 3,148,909 from 359,850 sessions, after a 30-day burn-in removed 210,346 rows and 24,401 sessions.
- **Features:** 44: 21 session-cumulative, 8 rolling 1 h, 7 hour-matched baseline z-scores, 6 static, 2 day-cumulative.
- **Text rows:** 29,551,223: http 26,667,373, email 2,467,177, file 416,673. Median about 50 tokens per event.
- **Tests:** 27 pass on the full outputs (`make test`, 83 s); the 23 fast tests take 14 s on the full outputs and 4 s on the 20-user build.
- **Mutation:** 19 of 19 deliberate faults caught (`make test-mutation`).
- **Reproducibility:** `make stage2` from an empty `data/processed` took 668 s, and `sessions`, `features` and `features_text` are byte-identical to the earlier outputs.
- **Coverage:** all 7,323 malicious events fall inside a session. The answer files were used for this check only.

## Open items for Comparison 2

- **Token budget:** about 82 events per kept session at about 50 whitespace tokens each is roughly 4,000 tokens, and more with a real tokeniser, so the SLM prompt needs a budget rule.
- **Empty file content:** 20,934 file events contain only a header, so their content is empty after the strip. TF-IDF and the prompt must handle empty strings.
