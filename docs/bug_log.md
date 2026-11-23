# Bug log

Entered at the freeze (tag `frozen-2026-11-23`). Each entry: what happened, its effect on reported
results, the fix, and whether it changes a design decision. Details and dates are in
`decisions_log.md`.

| # | Date found | Entry | Effect on reported numbers | Fix | Changes a design decision? |
|---|---|---|---|---|---|
| 1 | 2026-11-22 | Rationale sample of 17 Nov generated with Qwen default repetition_penalty 1.1 (not overridden), deviating from the logged greedy protocol; flips 10 C2 labels and affects reason wording; scores unaffected. | None: scores come from `score_texts`, which does not use `generate()`; the generated label is used nowhere; rationales feed only transparency examples. Step 8.1 latency config (c) was timed with the penalty active (timing only). | Regenerated 24 Nov with repetition_penalty=1.0 (rationales_v2). `src/slm/rationales.py` passes it explicitly and refuses to overwrite the 17 Nov file, which is kept; cause confirmed on the Mac (`results/step8/rationale_penalty_check.json`) and in the lab by `src/slm/rationale_diag.py` step (f). | no |
| 2 | 2026-11-09 | Two sessions committed concurrently at Gate 1: the Gate 1 record (report + log line) landed in e9a0e0b after an empty commit 06d75d4. | None: the content was verified byte-identical; the fixes in e9a0e0b were checked individually. | Fix forward (history not rewritten); the freeze rebuild in a clean clone (`results/freeze/rebuild.json`) re-checks every Mac-side result from the stored predictions. | no |
| 3 | 2026-11-14 | Commit 410fef2 (pushed; Step 7 P0 + 7.1) recorded the machine hostname in results metadata. | None: metadata only, no number. | Fix forward in 5deb353 and 4a78b28 (generic machine name "Mac (arm64, Darwin)"); history not rewritten (project decision); tracked files grep-checked for hostnames, names and home paths at every commit since. | no |
