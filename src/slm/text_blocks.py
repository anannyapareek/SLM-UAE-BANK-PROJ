"""Build the Comparison 2 text blocks (runbook 6.2) for the rows that need them.

Needs data/processed/features_text.parquet (event-level email, http and file content) and
the *_sessioned tables, so it runs on the data machine; the output file is small and is
copied to the GPU machine, where make_train --text and score.py --text read it.

Rows: the training rows of make_train (train.jsonl and valid.jsonl: same rows, seed and 1:10
negatives as the structured set) and the final (logoff) scoring row of every valid, calib
and test session (Comparison 2 scores logoff-only). Test rows need --allow-test; only their
inputs are built, no labels are read for them.

Per row: the session's events with ts <= score_time, rendered one per line and selected by
serialise.text_block (tiers, newest first, whole events, 256-token budget measured with the
configured model tokenizer). Output columns: row_id, session_id, score_time, sets, block,
tokens, tokens_untruncated, eligible_events, selected_events, truncated.

Run: python -m src.slm.text_blocks --allow-test [--out data/processed/slm_text_blocks.parquet]
"""
import json
import pathlib
import time

import duckdb
import numpy as np
import pandas as pd
import yaml

from src.slm import common, serialise
from src.splits import verify_test_checksum

P = common.P
CHUNK_SESSIONS = 4000


def scoring_rows(splits, train_rows=None, valid_rows=None):
    """row_id, session_id, score_time and the sets each row belongs to."""
    from src.slm.make_train import select_rows
    train, valid = select_rows(train_rows, valid_rows)
    parts = [train[["row_id", "session_id", "score_time"]].assign(sets="train_jsonl"),
             valid[["row_id", "session_id", "score_time"]].assign(sets="valid_jsonl")]
    con = duckdb.connect(config={"threads": 1})
    for split in splits:
        final = con.execute(f"""
            SELECT f.row_id, f.session_id, f.score_time FROM '{P}/features.parquet' f
            JOIN '{P}/splits.parquet' x USING (row_id)
            WHERE x.split = '{split}' AND f.is_final ORDER BY f.row_id""").df()
        parts.append(final.assign(sets=f"final_{split}"))
    rows = pd.concat(parts, ignore_index=True)
    rows = (rows.groupby(["row_id", "session_id", "score_time"], as_index=False)["sets"]
                .agg(lambda s: ",".join(sorted(set(s)))))
    return rows.sort_values(["session_id", "score_time", "row_id"]).reset_index(drop=True)


def extract_events(sessions, out, log):
    """One pass over the event tables: the needed sessions' events with their line metadata,
    sorted by session and time, written to `out`."""
    features_cfg = yaml.safe_load((common.ROOT / "configs/features.yaml").read_text())
    domain = features_cfg["internal_email_domain"]
    lists = features_cfg["http_keyword_lists"][features_cfg["http_lists"]]
    hosts = sorted({h.lower() for members in lists.values() for h in members})
    con = duckdb.connect(config={"threads": 1})
    tmp = out.parent / ".duckdb_tmp"
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute(f"SET memory_limit = '{features_cfg['text_build']['memory_limit']}'")
    con.execute("CREATE TEMP TABLE need (session_id BIGINT)")
    con.executemany("INSERT INTO need VALUES (?)", [(int(s),) for s in sessions])
    con.execute("CREATE TEMP TABLE flagged_hosts (host VARCHAR)")
    con.executemany("INSERT INTO flagged_hosts VALUES (?)", [(h,) for h in hosts])
    start = time.time()
    con.execute(f"""
        COPY (
            WITH http AS (
                SELECT id, lower(regexp_extract(url, '^https?://([^/]+)', 1)) AS host
                FROM '{P}/http_sessioned.parquet' SEMI JOIN need USING (session_id)),
            mail AS (
                SELECT id, attachments::INTEGER AS attachments,
                       len([x FOR x IN list_distinct(string_split(lower(concat_ws(';', "to", cc, bcc)), ';'))
                            IF x <> '']) AS recipients,
                       len([x FOR x IN list_distinct(string_split(lower(concat_ws(';', "to", cc, bcc)), ';'))
                            IF x <> '' AND split_part(x, '@', 2) <> '{domain}']) AS external
                FROM '{P}/email_sessioned.parquet' SEMI JOIN need USING (session_id)),
            files AS (
                SELECT id, lower(regexp_extract(filename, '\\.([A-Za-z0-9]+)$', 1)) AS ext
                FROM '{P}/file_sessioned.parquet' SEMI JOIN need USING (session_id))
            SELECT t.session_id, t.event_id, t.ts, t.source, t.content,
                   CASE t.source
                       WHEN 'http' THEN http.host
                       WHEN 'email' THEN printf('to %d, %d external, %d attachments',
                                                mail.recipients, mail.external, mail.attachments)
                       WHEN 'file' THEN CASE WHEN files.ext <> '' THEN '.' || files.ext END
                   END AS meta,
                   coalesce(t.source = 'http' AND http.host IN (SELECT host FROM flagged_hosts),
                            false) AS flagged
            FROM '{P}/features_text.parquet' t SEMI JOIN need USING (session_id)
            LEFT JOIN http ON t.source = 'http' AND http.id = t.event_id
            LEFT JOIN mail ON t.source = 'email' AND mail.id = t.event_id
            LEFT JOIN files ON t.source = 'file' AND files.id = t.event_id
            ORDER BY t.session_id, t.ts, t.event_id
        ) TO '{out}' (FORMAT parquet, ROW_GROUP_SIZE 65536)""")
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    log.info("extracted %d events for %d sessions in %.0f s", n, len(sessions), time.time() - start)


def build(out_path, tok, log, splits=("valid", "calib", "test"), train_rows=None,
          valid_rows=None, budget=None):
    budget = budget or common.config()["text"]["budget_tokens"]
    rows = scoring_rows(splits, train_rows, valid_rows)
    log.info("%d rows (%s)", len(rows), rows["sets"].str.split(",").explode().value_counts().to_dict())
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_path = out_path.with_name(out_path.stem + "_events.tmp.parquet")
    sessions = np.sort(rows["session_id"].unique())
    extract_events(sessions, events_path, log)

    count = lambda text: len(tok.encode(text, add_special_tokens=False))
    con = duckdb.connect(config={"threads": 1})
    results, start = [], time.time()
    for c in range(0, len(sessions), CHUNK_SESSIONS):
        chunk = sessions[c:c + CHUNK_SESSIONS]
        ev = con.execute(f"""SELECT * FROM '{events_path}'
                             WHERE session_id BETWEEN {int(chunk[0])} AND {int(chunk[-1])}""").df()
        ev = ev[ev["session_id"].isin(chunk)]
        records = ev.to_dict("records")
        lines = [serialise.event_line(e) for e in records]
        ids = tok(lines, add_special_tokens=False)["input_ids"] if lines else []
        line_tokens = {e["event_id"]: len(i) for e, i in zip(records, ids)}
        by_session = {}
        for e in records:
            by_session.setdefault(e["session_id"], []).append(e)
        for r in rows[rows["session_id"].isin(chunk)].itertuples(index=False):
            block, info = serialise.text_block(by_session.get(r.session_id, []), r.score_time,
                                               count, budget, line_tokens)
            results.append({"row_id": r.row_id, "session_id": r.session_id,
                            "score_time": r.score_time, "sets": r.sets, "block": block, **info})
        log.info("sessions %d / %d, rows %d (%.0f s)", min(c + CHUNK_SESSIONS, len(sessions)),
                 len(sessions), len(results), time.time() - start)
    events_path.unlink()
    frame = pd.DataFrame(results).sort_values("row_id").reset_index(drop=True)
    if (frame["tokens"] > budget).any():
        raise RuntimeError("a text block exceeds the budget")
    frame.to_parquet(out_path, index=False)
    return frame


def summarise(frame, budget):
    out = {}
    for name in sorted(set(",".join(frame["sets"]).split(","))):
        f = frame[frame["sets"].str.split(",").apply(lambda s: name in s)]
        over = f["tokens_untruncated"] > budget
        out[name] = {"rows": int(len(f)),
                     "over_budget_before_truncation": int(over.sum()),
                     "share_over_budget": round(float(over.mean()), 4),
                     "no_events": int((f["eligible_events"] == 0).sum()),
                     "median_eligible_events": float(f["eligible_events"].median()),
                     "median_selected_events": float(f["selected_events"].median()),
                     "median_block_tokens": float(f["tokens"].median()),
                     "max_block_tokens": int(f["tokens"].max()),
                     "median_untruncated_tokens": float(f["tokens_untruncated"].median())}
    return out


def main():
    cfg = common.config()["text"]
    p = common.parser(__doc__, device=False)
    p.add_argument("--out", default=str(common.ROOT / cfg["blocks"]))
    p.add_argument("--models-dir", help="local model copies (scripts/download_models.py)")
    p.add_argument("--allow-test", action="store_true", help="also build test final rows")
    p.add_argument("--train-rows", type=int, help="cap (tests)")
    p.add_argument("--valid-rows", type=int, help="override valid_loss_rows (tests)")
    p.add_argument("--splits", nargs="+", default=["valid", "calib", "test"],
                   choices=["valid", "calib", "test"], help="final rows of these splits")
    args = p.parse_args()
    log = common.setup_logging("text_blocks", args.log)
    log.info("test_ids.csv SHA-256 %s", verify_test_checksum())
    if "test" in args.splits and not args.allow_test:
        raise SystemExit("building test rows needs --allow-test (inputs only; no labels read)")
    tok = common.load_tokenizer(cfg["tokenizer"], args.models_dir)
    frame = build(args.out, tok, log, tuple(args.splits), args.train_rows, args.valid_rows)
    summary = summarise(frame, cfg["budget_tokens"])
    common.write_json(pathlib.Path(args.out).with_suffix(".summary.json"),
                      {"budget": cfg["budget_tokens"], "tokenizer": common.model_entry(cfg["tokenizer"]),
                       "sets": summary})
    log.info("wrote %s (%d rows)\n%s", args.out, len(frame), json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
