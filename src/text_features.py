"""Per-scoring-point text features for XGBoost (Comparison 2, runbook 6.2).

From data/processed/features_text.parquet (event-level email, http and file content; the
content fields are keyword lists). Configuration: configs/features_text.yaml.

1. Fit, per source, a TF-IDF vectoriser and a TruncatedSVD on the events of train-split
   sessions only (a seeded sample). Saved to data/processed/text_features_model.joblib.
2. For every scoring row and source: the mean, over the session's events of that source
   with ts <= score_time, of the event's L2-normalised TF-IDF vector projected by the SVD.
   NaN when there is no such event yet. Because TF-IDF (per event) and SVD are linear, the
   mean of projections equals the projection of the mean, so cumulative sums give every
   scoring point exactly.
3. Join to features.parquet: data/processed/features_with_text.parquet.

Feature code: reads no labels. The leakage rule (ts <= score_time) is the structured one.
Run: python run.py text_features
"""
import os
import pathlib
import time

import duckdb
import joblib
import numpy as np
import pandas as pd
import yaml

from src.splits import verify_test_checksum

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
CONFIG = pathlib.Path("configs/features_text.yaml")


def config():
    return yaml.safe_load(CONFIG.read_text())


def column_names(cfg):
    return [f"txt_{s}_{i:02d}" for s, spec in cfg["sources"].items()
            for i in range(spec["svd_dims"])]


def fit(con, cfg, log=print):
    """TF-IDF + SVD per source, fitted on a seeded sample of train-session events."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    t, f = cfg["tfidf"], cfg["fit"]
    models = {}
    for source, spec in cfg["sources"].items():
        # The sample is drawn from this source's train-session events (subquery first).
        sample = con.execute(f"""
            SELECT content FROM (
                SELECT t.event_id, t.content FROM '{P}/features_text.parquet' t
                SEMI JOIN (SELECT DISTINCT session_id FROM '{P}/splits.parquet'
                           WHERE split = '{f['split']}') x USING (session_id)
                WHERE t.source = '{source}'
            ) USING SAMPLE {int(f['max_events_per_source'])} ROWS (reservoir, {int(f['seed'])})
            ORDER BY event_id
        """).df()["content"].tolist()
        vec = TfidfVectorizer(max_features=t["max_features"], token_pattern=t["token_pattern"],
                              lowercase=t["lowercase"], sublinear_tf=t["sublinear_tf"],
                              norm=t["norm"], min_df=t["min_df"], dtype=np.float32)
        X = vec.fit_transform(sample)
        svd = TruncatedSVD(n_components=spec["svd_dims"], algorithm=cfg["svd"]["algorithm"],
                           n_iter=cfg["svd"]["n_iter"], random_state=cfg["svd"]["seed"])
        svd.fit(X)
        models[source] = {"tfidf": vec, "svd": svd, "fit_events": len(sample),
                          "vocabulary": len(vec.vocabulary_),
                          "explained_variance": float(svd.explained_variance_ratio_.sum())}
        log(f"{source}: fitted on {len(sample):,} train-session events; vocabulary "
            f"{len(vec.vocabulary_):,}; SVD {spec['svd_dims']} dims, explained variance "
            f"{models[source]['explained_variance']:.3f}")
    return models


def project(model, contents):
    """Each event's L2-normalised TF-IDF vector projected by the SVD (float64)."""
    return model["svd"].transform(model["tfidf"].transform(contents)).astype(np.float64)


def cumulative_means(rows, events, vectors):
    """Mean of `vectors` over each row's session events with ts <= score_time.

    rows: DataFrame with session_id, score_time (any order).
    events: DataFrame with session_id, ts, sorted by (session_id, ts, event_id); vectors are
    aligned with it. Returns an array (len(rows), dims), NaN where no event qualifies.
    """
    dims = vectors.shape[1]
    out = np.full((len(rows), dims), np.nan)
    if len(events) == 0:
        return out
    cum = np.vstack([np.zeros((1, dims)), np.cumsum(vectors, axis=0)])
    ev = events[["session_id", "ts"]].reset_index(drop=True).copy()
    ev["idx"] = np.arange(len(ev))
    first = ev.groupby("session_id")["idx"].min().rename("first")
    left = rows[["session_id", "score_time"]].reset_index(drop=True).copy()
    left["pos"] = np.arange(len(left))
    left["score_time"] = left["score_time"].astype("datetime64[us]")
    ev["ts"] = ev["ts"].astype("datetime64[us]")
    matched = pd.merge_asof(left.sort_values("score_time"), ev.sort_values("ts"),
                            left_on="score_time", right_on="ts", by="session_id",
                            direction="backward", allow_exact_matches=True)
    matched = matched.dropna(subset=["idx"]).join(first, on="session_id")
    last = matched["idx"].to_numpy(np.int64)
    start = matched["first"].to_numpy(np.int64)
    count = (last - start + 1)[:, None]
    out[matched["pos"].to_numpy()] = (cum[last + 1] - cum[start]) / count
    return out


def build(con, cfg, models, log=print):
    """Text columns for every scoring row, written with the structured features."""
    names = column_names(cfg)
    rows = con.execute(f"""SELECT row_id, session_id, score_time FROM '{P}/features.parquet'
                           ORDER BY session_id, score_time, row_id""").df()
    sessions = np.sort(rows["session_id"].unique())
    size = int(cfg["build"]["chunk_sessions"])
    parts, start = [], time.time()
    for c in range(0, len(sessions), size):
        chunk = sessions[c:c + size]
        lo, hi = int(chunk[0]), int(chunk[-1])
        ev = con.execute(f"""
            SELECT session_id, ts, source, event_id, content FROM '{P}/features_text.parquet'
            WHERE session_id BETWEEN {lo} AND {hi}
            ORDER BY session_id, ts, event_id""").df()
        r = rows[(rows["session_id"] >= lo) & (rows["session_id"] <= hi)]
        block = [r[["row_id"]].reset_index(drop=True)]
        for source, model in models.items():
            e = ev[ev["source"] == source].reset_index(drop=True)
            vectors = project(model, e["content"].tolist()) if len(e) else np.zeros(
                (0, model["svd"].n_components))
            cols = [n for n in names if n.startswith(f"txt_{source}_")]
            block.append(pd.DataFrame(cumulative_means(r, e, vectors).astype(np.float32),
                                      columns=cols))
        parts.append(pd.concat(block, axis=1))
        log(f"sessions {min(c + size, len(sessions)):,} / {len(sessions):,}, "
            f"events {len(ev):,} ({time.time() - start:.0f} s)")
    text = pd.concat(parts, ignore_index=True)
    if len(text) != len(rows) or text["row_id"].duplicated().any():
        raise RuntimeError("text features do not cover every scoring row exactly once")
    return text


def run():
    checksum = verify_test_checksum()
    cfg = config()
    print(f"test_ids.csv SHA-256 {checksum}", flush=True)
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    con = duckdb.connect(config={"threads": 1})
    models = fit(con, cfg, log)
    model_path = pathlib.Path(cfg["out"]["model"])
    joblib.dump({"config": cfg, "models": models}, model_path)
    log(f"saved {model_path}")
    text = build(con, cfg, models, log)
    out = pathlib.Path(cfg["out"]["features"])
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"SET temp_directory = '{P}/.duckdb_tmp'")
    con.execute("SET memory_limit = '8GB'")
    con.register("text", text)
    con.execute(f"""COPY (SELECT f.*, {', '.join(f't."{n}"' for n in column_names(cfg))}
                          FROM '{P}/features.parquet' f JOIN text t USING (row_id)
                          ORDER BY f.row_id)
                    TO '{tmp}' (FORMAT parquet)""")
    tmp.replace(out)
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    log(f"wrote {out}: {n:,} rows, {len(column_names(cfg))} text columns")


if __name__ == "__main__":
    run()
