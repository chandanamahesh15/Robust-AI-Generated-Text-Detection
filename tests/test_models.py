"""Tests for metric computation and the TF-IDF model save/load roundtrip."""

from __future__ import annotations

from src.config import Config
from src.evaluate import compute_metrics
from src.models.tfidf import TfidfLogRegModel


def test_compute_metrics_perfect_prediction():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], "dummy", "test")
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0
    assert m["model"] == "dummy" and m["split"] == "test"


def test_compute_metrics_handles_all_wrong():
    m = compute_metrics([0, 0, 0, 0], [1, 1, 1, 1], "dummy")
    assert m["accuracy"] == 0.0
    # precision/recall guarded by zero_division=0, so no exception
    assert m["precision"] == 0.0


def test_tfidf_train_save_load_roundtrip(tmp_path):
    cfg = Config()
    cfg.artifacts.dir = str(tmp_path)
    cfg.tfidf.min_df = 1  # tiny corpus

    texts = ["human wrote this nice text"] * 10 + ["ai generated synthetic output here"] * 10
    labels = [0] * 10 + [1] * 10

    model = TfidfLogRegModel(cfg).fit(texts, labels)
    model.save(cfg)

    reloaded = TfidfLogRegModel.load(cfg)
    preds = reloaded.predict(["human wrote this nice text"])
    assert preds[0] == 0
    assert 0.0 <= reloaded.predict_proba(["anything"])[0] <= 1.0


def test_predict_before_fit_raises():
    import pytest

    with pytest.raises(RuntimeError):
        TfidfLogRegModel(Config()).predict(["x"])
