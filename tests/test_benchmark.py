"""Tests for the benchmark orchestrator's CPU paths and metric persistence."""

from __future__ import annotations

import pandas as pd

from src.benchmark import run_baselines, run_tfidf
from src.config import Config
from src.data import make_splits
from src.evaluate import save_metrics


def _splits():
    df = pd.DataFrame(
        {"text": [f"human written sample {i} about daily life" if i % 2 == 0
                  else f"ai generated sample {i} with systematic structure" for i in range(120)],
         "generated": [i % 2 for i in range(120)]}
    )
    return make_splits(df, Config())


def test_baselines_return_two_rows_with_expected_models():
    cfg = Config()
    cfg.tfidf.min_df = 1
    rows = run_baselines(_splits(), cfg)
    names = {r["model"] for r in rows}
    assert names == {"Majority Class Baseline", "Stratified Baseline"}
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)


def test_tfidf_benchmark_reports_val_and_test():
    cfg = Config()
    cfg.tfidf.min_df = 1
    rows = run_tfidf(_splits(), cfg)
    assert {r["split"] for r in rows} == {"validation", "test"}


def test_save_metrics_writes_csv(tmp_path):
    rows = [{"model": "m", "split": "test", "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}]
    out = tmp_path / "metrics.csv"
    df = save_metrics(rows, out)
    assert out.exists()
    assert list(df.columns) == ["model", "split", "accuracy", "precision", "recall", "f1"]
