"""Evaluation metrics: weighting, agreement with sklearn, edge cases, CLI guard."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_curve

from src import eval as ev

CFG = {"fpr_levels": [0.01, 0.05], "alerts_per_day": 3, "ece_bins": 15, "scenario_fpr": 0.05,
       "tie_break_seed": 0, "bootstrap": {"resamples": 200, "seed": 0, "ci": 0.95}}


def scored(score, y, day=None, key=None):
    n = len(score)
    day = np.zeros(n, int) if day is None else np.asarray(day)
    key = np.arange(n) if key is None else np.asarray(key)
    return ev.Scored(score, y, day, key)


def all_metrics(s, w):
    out = ev.core_metrics(s, w, CFG)
    out["recall_threshold"] = ev.recall_at_fpr(s, w, 0.05)[1]
    out["brier"] = ev.brier(s, w)
    out["ece_mass"] = ev.ece(s, w, 15, True)
    out["ece_width"] = ev.ece(s, w, 15, False)
    return out


@pytest.fixture
def small():
    rng = np.random.default_rng(3)
    n = 60
    score = np.round(rng.random(n), 1)          # rounding forces many ties
    y = (rng.random(n) < 0.3).astype(int)
    day = rng.integers(0, 4, n)
    return score, y, day, np.arange(n)


def test_unit_weights_match_plain_metrics(small):
    score, y, day, key = small
    s = scored(score, y, day, key)
    assert ev.average_precision(s, np.ones(len(y))) == pytest.approx(average_precision_score(y, score))


def test_weights_equal_explicit_duplication(small):
    score, y, day, key = small
    w = np.random.default_rng(5).integers(0, 4, len(y)).astype(float)
    weighted = all_metrics(scored(score, y, day, key), w)
    copies = np.repeat(np.arange(len(y)), w.astype(int))
    duplicated = all_metrics(scored(score[copies], y[copies], day[copies], key[copies]),
                             np.ones(len(copies)))
    for name, value in weighted.items():
        assert value == pytest.approx(duplicated[name], abs=1e-12), name


def test_pr_auc_and_roc_match_sklearn_with_ties(small):
    score, y, _, _ = small
    s = scored(score, y)
    w = np.ones(len(y))
    assert ev.average_precision(s, w) == pytest.approx(average_precision_score(y, score))
    fpr, tpr, thresholds = roc_curve(y, score)
    for level in (0.01, 0.05, 0.3):
        ok = fpr <= level
        recall, threshold = ev.recall_at_fpr(s, w, level)
        assert recall == pytest.approx(tpr[ok].max())
        if recall > 0:
            assert threshold == pytest.approx(thresholds[ok][np.argmax(tpr[ok])])


def test_perfect_and_random_scorers():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.1).astype(int)
    perfect = scored(y + rng.random(len(y)) * 0.5, y)
    w = np.ones(len(y))
    assert ev.average_precision(perfect, w) == pytest.approx(1.0)
    assert ev.recall_at_fpr(perfect, w, 0.01)[0] == pytest.approx(1.0)
    noise = scored(rng.random(len(y)), y)
    assert ev.average_precision(noise, w) == pytest.approx(y.mean(), abs=0.02)


def test_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(1)
    score = rng.random(200000)
    y = (rng.random(len(score)) < score).astype(int)
    s = scored(score, y)
    w = np.ones(len(y))
    assert ev.ece(s, w, 15, True) < 0.01
    assert ev.ece(s, w, 15, False) < 0.01


def test_top_k_per_day_counts():
    s = scored(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), np.array([1, 0, 1, 0, 1]), day=[0, 0, 0, 1, 1])
    precision, recall = ev.top_k_per_day(s, np.ones(5), 2)
    # day 0 alerts on 0.9 and 0.8; day 1 alerts on both units
    assert precision == pytest.approx(2 / 4)
    assert recall == pytest.approx(2 / 3)


def unit_set(seed):
    rng = np.random.default_rng(seed)
    n = 400
    users = rng.integers(0, 40, n)
    y = (rng.random(n) < 0.1).astype(int)
    meta = pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], "session_id": np.arange(n),
                         "user": [f"u{x}" for x in users], "heldout_user": False,
                         "score_time": pd.Timestamp("2010-09-01") + pd.to_timedelta(
                             rng.integers(0, 10, n), unit="D"),
                         "scenario": np.where(y == 1, 2, np.nan), "y": y})
    scores = [np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1) for _ in range(2)]
    u = ev.build_units(meta, scores, "row", {**CFG, "tie_break_seed": 0})
    u["strata"], u["unit_key"] = ev.user_strata(u), "row"
    return u


def test_paired_model_against_itself_is_zero():
    u = unit_set(2)
    resamples = ev.bootstrap({"a": u, "b": u}, CFG, n_jobs=1)
    for metric in ev.CI_METRICS:
        diffs = [r["a"][metric] - r["b"][metric] for r in resamples]
        ci, skipped = ev.interval(diffs, 0.95)
        assert ci == [0.0, 0.0] and skipped == 0
        assert not (ci[0] > 0 or ci[1] < 0)


def test_session_aggregation():
    meta = pd.DataFrame({
        "row_id": ["a", "b", "c", "d"], "session_id": [1, 1, 2, 2], "user": ["u", "u", "v", "v"],
        "heldout_user": False, "score_time": pd.to_datetime(
            ["2010-09-01 09:00", "2010-09-01 10:00", "2010-09-02 09:00", "2010-09-02 10:00"]),
        "scenario": [np.nan, 1, np.nan, np.nan], "y": [0, 1, 0, 0]})
    units, scores = ev.aggregate_sessions(meta, [np.array([0.2, 0.7, 0.9, 0.1])])
    assert units["y"].tolist() == [1, 0]
    assert scores[0].tolist() == [0.7, 0.9]
    assert units["scenario"].tolist()[0] == 1


def test_session_aggregation_final_row():
    """final = the score at each session's last scoring point, whatever the row order."""
    meta = pd.DataFrame({
        "row_id": ["a", "b", "c", "d", "e"], "session_id": [1, 2, 1, 2, 1],
        "user": ["u", "v", "u", "v", "u"], "heldout_user": False,
        "score_time": pd.to_datetime(["2010-09-01 10:00", "2010-09-02 09:00",
                                      "2010-09-01 09:00", "2010-09-02 11:00",
                                      "2010-09-01 09:30"]),
        "scenario": [1, np.nan, np.nan, np.nan, np.nan], "y": [1, 0, 0, 0, 0]})
    s = np.array([0.2, 0.9, 0.8, 0.1, 0.5])
    units, final = ev.aggregate_sessions(meta, [s], "final")
    _, hourly = ev.aggregate_sessions(meta, [s], "max")
    assert units["session_id"].tolist() == [1, 2]
    assert final[0].tolist() == [0.2, 0.1]          # last points: 10:00 and 11:00
    assert hourly[0].tolist() == [0.8, 0.9]
    assert units["y"].tolist() == [1, 0]            # label: any positive row, either way
    assert units["last_time"].tolist() == list(pd.to_datetime(["2010-09-01 10:00",
                                                               "2010-09-02 11:00"]))
    with pytest.raises(ValueError):
        ev.aggregate_sessions(meta, [s], "mean")


def test_session_agg_flag_defaults_to_max():
    base = ["single", "--name", "m", "--split", "valid", "--preds", "p", "--out", "o"]
    assert ev.parse(base).session_agg == "max"
    assert ev.parse(base + ["--session-agg", "final"]).session_agg == "final"


def test_test_split_needs_allow_test(monkeypatch):
    monkeypatch.setattr(ev, "verify_test_checksum", lambda: "x")
    with pytest.raises(RuntimeError, match="allow-test"):
        ev.main(["single", "--name", "m", "--split", "test", "--preds", "p.parquet",
                 "--out", "o.json"])


def test_paired_follows_session_agg(tmp_path):
    """eval paired aggregates both models the same way, as --session-agg says."""
    from types import SimpleNamespace
    rng = np.random.default_rng(5)
    rows = []
    for session in range(40):
        user = f"u{session % 10}"
        positive = session % 5 == 0
        for hour in range(3):
            rows.append({"row_id": f"r{session}_{hour}", "session_id": session, "user": user,
                         "heldout_user": False,
                         "score_time": pd.Timestamp("2010-09-01") + pd.Timedelta(days=session % 4,
                                                                                 hours=hour),
                         "scenario": 1.0 if positive else np.nan,
                         "y": int(positive and hour == 2)})
    meta = pd.DataFrame(rows)
    paths = {}
    for model in ("a", "b"):
        frame = pd.DataFrame({"row_id": meta["row_id"], "y_true": meta["y"],
                              "score": rng.random(len(meta)) + 0.5 * meta["y"]})
        paths[model] = str(tmp_path / f"{model}.parquet")
        frame.to_parquet(paths[model])
    args = SimpleNamespace(name_a="a", a=[paths["a"]], name_b="b", b=[paths["b"]], split="valid")
    cfg = dict(CFG, unit="session")
    results = {}
    for agg in ("max", "final"):
        cfg_agg = dict(cfg, session_agg=agg)
        out = ev.paired(args, cfg_agg, meta)
        for model in ("a", "b"):
            s = pd.read_parquet(paths[model])["score"].to_numpy()
            units, scores = ev.aggregate_sessions(meta, [s], agg)
            day = units["last_time"].dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)
            key = pd.util.hash_array(units["session_id"].astype(str).to_numpy(),
                                     hash_key=f"{cfg['tie_break_seed']:016d}")
            direct = ev.core_metrics(ev.Scored(scores[0], units["y"].to_numpy(), day, key),
                                     np.ones(len(units)), cfg)
            assert out["session"]["pr_auc"][model] == pytest.approx(direct["pr_auc"])
        results[agg] = out["session"]["pr_auc"]["a"]
    assert results["max"] != results["final"]


@pytest.fixture
def tiny_split(tmp_path, monkeypatch):
    """A small processed folder: 12 users x 2 sessions x 3 hourly rows, last row final."""
    rng = np.random.default_rng(11)
    rows = []
    for u in range(12):
        for k in range(2):
            session = u * 10 + k
            positive = (u % 3 == 0) and k == 1
            for h in range(3):
                rows.append({"row_id": f"r{session}_{h}", "session_id": session, "user": f"u{u}",
                             "heldout_user": False, "split": "valid",
                             "score_time": pd.Timestamp("2010-09-01") + pd.Timedelta(days=k, hours=h),
                             "scenario": 1.0 if positive else np.nan,
                             "y": int(positive and h >= 1), "is_final": h == 2})
    frame = pd.DataFrame(rows)
    frame[["row_id", "session_id", "user", "heldout_user", "split"]].to_parquet(tmp_path / "splits.parquet")
    frame[["row_id", "score_time", "scenario", "y"]].to_parquet(tmp_path / "labels.parquet")
    frame[["row_id", "is_final"]].to_parquet(tmp_path / "features.parquet")
    monkeypatch.setattr(ev, "P", tmp_path)
    monkeypatch.setattr(ev, "verify_test_checksum", lambda: "frozen-hash")
    preds = frame.assign(y_true=frame["y"], score=rng.random(len(frame)) + 0.5 * frame["y"])
    full, final = tmp_path / "preds_valid.parquet", tmp_path / "preds_valid_final.parquet"
    preds[["row_id", "y_true", "score"]].to_parquet(full)
    preds.loc[preds["is_final"], ["row_id", "y_true", "score"]].to_parquet(final)
    return tmp_path, full, final


def test_rows_final_accepts_logoff_only_preds(tiny_split):
    tmp, _, final = tiny_split
    args = ["single", "--name", "m", "--split", "valid", "--preds", str(final), "--out",
            str(tmp / "final.json")]
    with pytest.raises(RuntimeError, match="rows do not match the split"):
        ev.main(args + ["--rows", "all"])
    out = ev.main(args + ["--rows", "final"])
    saved = json.loads((tmp / "final.json").read_text())
    assert saved["rows"] == "final" and saved["n_units"] == {"row": 24, "session": 24}
    assert saved["session"]["counts"]["n_units"] == 24 and saved["row"]["counts"]["n_units"] == 24
    assert saved["test_ids_sha256"] == "frozen-hash"
    assert out["session"]["counts"]["n_pos"] == 4


def test_rows_final_refuses_hourly_preds_and_default_is_unchanged(tiny_split):
    tmp, full, _ = tiny_split
    args = ["single", "--name", "m", "--split", "valid", "--preds", str(full), "--out",
            str(tmp / "all.json")]
    with pytest.raises(RuntimeError, match="rows do not match the split"):
        ev.main(args + ["--rows", "final"])
    ev.main(args)                                       # default: all rows
    saved = json.loads((tmp / "all.json").read_text())
    assert "rows" not in saved and "n_units" not in saved
    assert saved["row"]["counts"]["n_units"] == 72


def test_rows_final_in_paired(tiny_split):
    tmp, _, final = tiny_split
    out = ev.main(["paired", "--split", "valid", "--rows", "final", "--name-a", "a", "--a",
                   str(final), "--name-b", "b", "--b", str(final), "--out", str(tmp / "p.json")])
    assert out["rows"] == "final" and out["session"]["pr_auc"]["difference"] == 0.0


def test_calibration_metrics_weights_equal_duplication(small):
    score, y, day, key = small
    w = np.array([2, 0, 1, 3] * 15, dtype=float)
    keep = np.repeat(np.arange(len(score)), w.astype(int))
    a = ev.calibration_metrics(scored(score, y, day, key), w, CFG)
    b = ev.calibration_metrics(scored(score[keep], y[keep], day[keep], key[keep]),
                               np.ones(len(keep)), CFG)
    assert a == pytest.approx(b)


def test_calibration_flag(tiny_split):
    tmp, full, _ = tiny_split
    d = pd.read_parquet(full)
    prob = tmp / "prob.parquet"
    d.assign(score=d["score"] / d["score"].max()).to_parquet(prob)      # scores in [0, 1]
    base = ["single", "--name", "m", "--split", "valid", "--preds", str(prob), "--out"]
    ev.main(base + [str(tmp / "plain.json")])
    ev.main(base + [str(tmp / "cal.json"), "--calibration"])
    plain, cal = (json.loads((tmp / f).read_text()) for f in ("plain.json", "cal.json"))
    for unit in ("session", "row"):
        assert set(plain[unit]["ci"]) == set(ev.CI_METRICS)
        assert set(cal[unit]["ci"]) == set(ev.CI_METRICS + ev.CALIBRATION_METRICS)
        for k in ev.CI_METRICS:                       # ranking metrics untouched by the flag
            assert cal[unit]["ci"][k] == plain[unit]["ci"][k]
            assert cal[unit]["mean"][k] == plain[unit]["mean"][k]
        seed = cal[unit]["per_seed"][0]
        assert "mean_predicted" in seed and "observed_rate" in seed
        assert "mean_predicted" not in plain[unit]["per_seed"][0]
        lo, hi = cal[unit]["ci"]["brier"]
        assert lo <= cal[unit]["mean"]["brier"] <= hi
    assert cal["config"]["calibration"] is True and "calibration" not in plain["config"]


def test_calibration_flag_skips_non_probability_scores(tiny_split):
    tmp, full, _ = tiny_split
    d = pd.read_parquet(full)
    d.assign(score=d["score"] * 4).to_parquet(tmp / "counts.parquet")     # rules-like scale
    out = ev.main(["single", "--name", "m", "--split", "valid", "--preds", str(tmp / "counts.parquet"),
                   "--out", str(tmp / "c.json"), "--calibration"])
    assert set(out["session"]["ci"]) == set(ev.CI_METRICS)
    assert "brier" not in out["session"]["mean"]


def test_brier_skill_score(tiny_split):
    tmp, full, _ = tiny_split
    d = pd.read_parquet(full)
    p_row = float(d["y_true"].mean())
    clim = tmp / "climatology.parquet"
    d.assign(score=p_row).to_parquet(clim)                    # constant forecast = base rate
    out = ev.main(["single", "--name", "m", "--split", "valid", "--preds", str(clim), "--out",
                   str(tmp / "b.json"), "--calibration", "--bss-ref-split", "valid"])
    row = out["row"]
    assert row["mean"]["bss_own"] == pytest.approx(0.0, abs=1e-12)   # climatology has no skill
    ref = out["config"]["bss_reference"]
    assert ref["split"] == "valid" and ref["rates"]["row"] == pytest.approx(p_row)
    sess = out["session"]
    p_sess = ref["rates"]["session"]
    assert sess["mean"]["bss_reference"] == pytest.approx(1 - sess["mean"]["brier"] / (p_sess * (1 - p_sess)))
    lo, hi = sess["ci"]["bss_reference"]
    assert lo <= sess["mean"]["bss_reference"] <= hi
    with pytest.raises(RuntimeError, match="needs --calibration"):
        ev.main(["single", "--name", "m", "--split", "valid", "--preds", str(clim), "--out",
                 str(tmp / "x.json"), "--bss-ref-split", "valid"])
    plain = ev.main(["single", "--name", "m", "--split", "valid", "--preds", str(clim), "--out",
                     str(tmp / "p.json"), "--calibration"])
    assert "bss_reference" not in plain["row"]["mean"] and "bss_reference" not in plain["config"]


def test_pr_auc_shortcut_equals_scored_path(small):
    score, y, day, key = small
    full = ev.average_precision(scored(score, y, day, key), np.ones(len(score)))
    assert ev.pr_auc(score, y) == full                       # identical, ties included
