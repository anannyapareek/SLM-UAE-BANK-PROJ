"""XGBoost text features (src/text_features.py): aggregation rule and leakage on real rows."""
import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from src import text_features as tf

T0 = pd.Timestamp("2010-03-01 09:00")


def t(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def test_cumulative_means_rule():
    events = pd.DataFrame({"session_id": [1, 1, 1, 2, 2], "ts": [t(0), t(30), t(90), t(5), t(200)]})
    vectors = np.array([[1.0, 0], [3.0, 0], [100.0, 100], [0, 2.0], [0, 50.0]])
    rows = pd.DataFrame({"session_id": [1, 1, 1, 2, 2, 3],
                         "score_time": [t(-1), t(30), t(60), t(60), t(200), t(60)]})
    out = tf.cumulative_means(rows, events, vectors)
    assert np.isnan(out[0]).all()                       # before the first event
    assert out[1].tolist() == [2.0, 0.0]                # an event at score_time counts
    assert out[2].tolist() == [2.0, 0.0]                # the 90-minute event never does
    assert out[3].tolist() == [0.0, 2.0] and out[4].tolist() == [0.0, 26.0]
    assert np.isnan(out[5]).all()                       # no events in the session


def test_cumulative_means_ignores_row_order():
    rng = np.random.default_rng(0)
    events = pd.DataFrame({"session_id": np.repeat([1, 2, 3], 20),
                           "ts": [t(int(m)) for m in np.tile(np.sort(rng.integers(0, 600, 20)), 3)]})
    vectors = rng.random((60, 4))
    rows = pd.DataFrame({"session_id": rng.integers(1, 4, 50),
                         "score_time": [t(int(m)) for m in rng.integers(-10, 700, 50)]})
    a = tf.cumulative_means(rows, events, vectors)
    shuffled = rows.sample(frac=1, random_state=1)
    b = tf.cumulative_means(shuffled, events, vectors)
    assert np.allclose(a[shuffled.index.to_numpy()], b, equal_nan=True)


def test_config_dimensions():
    cfg = tf.config()
    names = tf.column_names(cfg)
    assert len(names) == sum(s["svd_dims"] for s in cfg["sources"].values()) == 100
    assert cfg["fit"]["split"] == "train"


@pytest.mark.data
def test_text_features_use_only_past_events(con):
    """Recompute 300 random rows from the raw events, one event at a time: the stored values
    must equal the mean projection of events with ts <= score_time (and differ from what
    later events would give, where there are any)."""
    cfg = tf.config()
    out, model_path = ROOT / cfg["out"]["features"], ROOT / cfg["out"]["model"]
    if not (out.exists() and model_path.exists()):
        pytest.skip("features_with_text.parquet not built")
    models = joblib.load(model_path)["models"]
    names = tf.column_names(cfg)
    rows = con.execute(f"""
        SELECT row_id, session_id, score_time, {', '.join(names)} FROM '{out}'
        USING SAMPLE 300 ROWS (reservoir, 7)""").df()
    sessions = ",".join(str(int(s)) for s in rows["session_id"].unique())
    events = con.execute(f"""SELECT session_id, ts, source, content
                             FROM '{ROOT}/data/processed/features_text.parquet'
                             WHERE session_id IN ({sessions})""").df()
    later_checked = 0
    for r in rows.itertuples(index=False):
        for source, model in models.items():
            cols = [n for n in names if n.startswith(f"txt_{source}_")]
            stored = np.array([getattr(r, c) for c in cols], dtype=float)
            ev = events[(events["session_id"] == r.session_id) & (events["source"] == source)]
            past = ev[ev["ts"] <= r.score_time]
            if past.empty:
                assert np.isnan(stored).all()
                continue
            expected = tf.project(model, past["content"].tolist()).mean(axis=0)
            assert np.allclose(stored, expected, atol=1e-5, rtol=1e-4), (r.row_id, source)
            if len(ev) > len(past):
                leaked = tf.project(model, ev["content"].tolist()).mean(axis=0)
                if not np.allclose(leaked, expected, atol=1e-5, rtol=1e-4):
                    later_checked += 1
    assert later_checked > 0          # the test can see leakage when it would matter
