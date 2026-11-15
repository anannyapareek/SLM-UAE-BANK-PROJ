"""Step 8.1 inputs for the GPU lab session (built on the Mac; decisions log 2026-11-15).

build  (Mac, has the test predictions and text blocks)
  Rationale sample: every positive test session (session label = any row positive) plus 100
  negative test sessions drawn with numpy default_rng(0). Comparison 1 uses each session's
  max-score scoring point under the C1 seed-1 test scores (primary aggregation; ties: earliest
  score_time, then row_id); Comparison 2 uses the session's logoff row. Prompts are serialised
  here (serialise.scoring_prompt / text_scoring_prompt), so the lab needs no feature data.
  Latency sample: 1,000 test logoff rows drawn with default_rng(0) (text blocks exist for
  logoff rows only, so structured, text and XGBoost inputs cover the same rows), with both
  prompts, the XGBoost v3 features of those rows, the v3 seed-1 model.json and that model's
  stored test scores for the rows (a correctness check on the lab, not a metric).
  Writes data/step8/ and bundle/step8_bundle.tar (+ .sha256); MANIFEST.sha256 inside. The tar
  is reproducible (fixed mtimes and owners): rebuilding gives the same SHA-256.
verify (lab)
  Every file against data/step8/MANIFEST.sha256, the repo's frozen test_ids.csv, and every
  sample row_id in the frozen test set.

Run: python -m src.slm.step8_inputs build
     python -m src.slm.step8_inputs verify [--root data/step8]
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/step8"
BUNDLE = ROOT / "bundle/step8_bundle.tar"
C1_PREDS = ROOT / "results/c1_qwen15_r8_lr1e-4_s1/preds_test.parquet"
XGB_DIR = ROOT / "results/xgb_struct_v3_s1"
N_NEG, N_LATENCY, SEED = 100, 1000, 0
FILES = ["rationale_sample.parquet", "latency_sample.parquet", "xgb_features.parquet",
         "xgb_v3_s1_model.json", "inputs.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_rows(con, row_ids):
    import pandas as pd
    from src.slm import common
    ids = pd.DataFrame({"row_id": list(row_ids)})
    con.register("ids", ids)
    frame = con.execute(f"SELECT f.* FROM '{common.P}/features.parquet' f JOIN ids USING (row_id)").df()
    con.unregister("ids")
    if len(frame) != len(ids):
        raise RuntimeError(f"{len(ids) - len(frame)} rows missing from features.parquet")
    return frame.set_index("row_id")


def build(checksum):
    import duckdb
    import numpy as np
    import pandas as pd
    import yaml
    from src.baselines import xgb as base
    from src.slm import common, serialise
    from src.slm.make_train import load_blocks

    con = duckdb.connect(config={"threads": 4})
    from src import eval as ev                    # the evaluation stage's label loader
    meta = ev.load_meta(con, "test")[["row_id", "session_id", "score_time", "y"]]
    final_ids = set(ev.load_meta(con, "test", "final")["row_id"])
    meta["is_final"] = meta["row_id"].isin(final_ids)
    c1 = pd.read_parquet(C1_PREDS, columns=["row_id", "y_true", "score"])
    meta = meta.merge(c1, on="row_id", how="left", validate="1:1")
    if meta["score"].isna().any() or not (meta["y_true"] == meta["y"]).all():
        raise RuntimeError(f"{C1_PREDS} does not cover the test rows with the same labels")

    # Rationale sample: sessions
    sessions = meta.groupby("session_id")["y"].max()
    positives = np.sort(sessions.index[sessions == 1].to_numpy())
    negatives_all = np.sort(sessions.index[sessions == 0].to_numpy())
    negatives = np.sort(np.random.default_rng(SEED).choice(negatives_all, N_NEG, replace=False))
    chosen = pd.Series(1, index=positives).combine_first(pd.Series(0, index=negatives)).astype(int)
    sub = meta[meta["session_id"].isin(chosen.index)]
    c1_rows = (sub.sort_values(["session_id", "score", "score_time", "row_id"],
                               ascending=[True, False, True, True], kind="stable")
               .groupby("session_id").head(1))
    final = sub[sub["is_final"]]
    if final["session_id"].duplicated().any() or set(final["session_id"]) != set(chosen.index):
        raise RuntimeError("each sampled session needs exactly one logoff row")
    feats = feature_rows(con, set(c1_rows["row_id"]) | set(final["row_id"]))
    blocks = load_blocks(ROOT / common.config()["text"]["blocks"], list(final["row_id"]))
    records = []
    for model, frame in (("c1", c1_rows), ("c2", final)):
        for r in frame.sort_values("session_id").itertuples():
            row = feats.loc[r.row_id].to_dict()
            prompt = (serialise.scoring_prompt(row) if model == "c1"
                      else serialise.text_scoring_prompt(row, blocks[r.row_id]))
            records.append({"session_id": r.session_id, "row_id": r.row_id, "model": model,
                            "y_true": int(chosen[r.session_id]), "y_row": int(r.y),
                            "score_time": r.score_time, "prompt": prompt})
    rationale = pd.DataFrame(records)

    # Latency sample: logoff rows, no labels
    pool = np.sort(meta.loc[meta["is_final"], "row_id"].to_numpy())
    lat_ids = np.sort(np.random.default_rng(SEED).choice(pool, N_LATENCY, replace=False))
    lat_feats = feature_rows(con, lat_ids)
    lat_blocks = load_blocks(ROOT / common.config()["text"]["blocks"], list(lat_ids))
    latency = pd.DataFrame({
        "row_id": lat_ids,
        "structured_prompt": [serialise.scoring_prompt(lat_feats.loc[i].to_dict()) for i in lat_ids],
        "text_prompt": [serialise.text_scoring_prompt(lat_feats.loc[i].to_dict(), lat_blocks[i]) for i in lat_ids]})

    # XGBoost v3 seed 1: features of the latency rows, model, stored scores
    best = yaml.safe_load(base.BEST_CONFIG.read_text())
    names = base.feature_names()
    categories = {c: list(v) for c, v in best["categories"].items()}
    X = base.load_split(con, names, "test", categories)
    X = X.set_index("row_id").loc[lat_ids, names].reset_index()
    stored = pd.read_parquet(XGB_DIR / "preds_test.parquet", columns=["row_id", "score"]).set_index("row_id")
    X["expected_score"] = stored.loc[lat_ids, "score"].to_numpy()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rationale.to_parquet(OUT / "rationale_sample.parquet", index=False)
    latency.to_parquet(OUT / "latency_sample.parquet", index=False)
    X.to_parquet(OUT / "xgb_features.parquet", index=False)
    shutil.copy2(XGB_DIR / "model.json", OUT / "xgb_v3_s1_model.json")
    info = {"test_ids_sha256": checksum, "builder": "src/slm/step8_inputs.py", "seed": SEED,
            "rationale_sample": {"positive_sessions": int(len(positives)), "negative_sessions": int(len(negatives)),
                                 "negative_pool": int(len(negatives_all)), "rows": int(len(rationale)),
                                 "c1_rule": "max C1 seed-1 test score in the session (ties: earliest score_time, row_id)",
                                 "c2_rule": "logoff row", "c1_scores": str(C1_PREDS.relative_to(ROOT))},
            "latency_sample": {"rows": N_LATENCY, "pool": "test logoff rows", "pool_size": int(len(pool))},
            "xgb": {"model": str((XGB_DIR / "model.json").relative_to(ROOT)), "features": names,
                    "categories": categories},
            "files": FILES[:-1]}
    (OUT / "inputs.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    (OUT / "MANIFEST.sha256").write_text("".join(f"{sha256(OUT / f)}  data/step8/{f}\n" for f in FILES))
    BUNDLE.parent.mkdir(exist_ok=True)
    with tarfile.open(BUNDLE, "w", format=tarfile.PAX_FORMAT) as tar:
        for f in ["MANIFEST.sha256"] + FILES:
            info_ = tar.gettarinfo(OUT / f, arcname=f"data/step8/{f}")
            info_.uid = info_.gid = info_.mtime = 0    # reproducible: same inputs, same SHA-256
            info_.mode = 0o644
            info_.uname = info_.gname = ""
            with open(OUT / f, "rb") as fh:
                tar.addfile(info_, fh)
    digest = sha256(BUNDLE)
    BUNDLE.with_name(BUNDLE.name + ".sha256").write_text(f"{digest}  {BUNDLE.name}\n")
    print(json.dumps({k: v for k, v in info.items() if k != "xgb"}, indent=2, default=str))
    print(f"{BUNDLE} {BUNDLE.stat().st_size:,} bytes\nSHA-256 {digest}")


def verify(root, checksum):
    import pandas as pd
    root = pathlib.Path(root)
    base_dir = root.parents[1]                     # manifest names are data/step8/<file>
    bad = 0
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = base_dir / name
        ok = path.exists() and sha256(path) == expected
        bad += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} {name}")
    info = json.loads((root / "inputs.json").read_text())
    if info["test_ids_sha256"] != checksum:
        print(f"MISMATCH inputs.json test SHA-256 {info['test_ids_sha256']} vs repo {checksum}")
        bad += 1
    test_ids = set(pd.read_csv(ROOT / "data/processed/test_ids.csv")["row_id"])
    for name in ("rationale_sample.parquet", "latency_sample.parquet", "xgb_features.parquet"):
        ids = pd.read_parquet(root / name, columns=["row_id"])["row_id"]
        outside = int((~ids.isin(test_ids)).sum())
        bad += outside > 0
        print(f"{'OK      ' if not outside else 'MISMATCH'} {name}: {len(ids)} rows, {outside} outside the frozen test set")
    print(f"test_ids.csv SHA-256 {checksum}")
    if bad:
        print("STEP 8 INPUTS CHECK FAILED")
        sys.exit(1)
    print("STEP 8 INPUTS OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "verify"])
    p.add_argument("--root", default=str(OUT))
    args = p.parse_args()
    from src.splits import verify_test_checksum
    checksum = verify_test_checksum()
    build(checksum) if args.command == "build" else verify(args.root, checksum)


if __name__ == "__main__":
    main()
