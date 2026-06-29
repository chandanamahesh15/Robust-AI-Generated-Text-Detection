"""Tests for typed config loading and overrides."""

from __future__ import annotations

import textwrap

from src.config import Config, load_config


def _write(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_loads_nested_dataclasses(tmp_path):
    path = _write(tmp_path, """
        project:
          seed: 7
        tfidf:
          max_features: 500
    """)
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert cfg.project.seed == 7
    assert cfg.tfidf.max_features == 500
    # Unspecified fields fall back to dataclass defaults.
    assert cfg.tfidf.logreg_c == 1.0


def test_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_env_override(tmp_path, monkeypatch):
    path = _write(tmp_path, "tfidf:\n  max_features: 1000\n")
    monkeypatch.setenv("AVH__TFIDF__MAX_FEATURES", "2222")
    cfg = load_config(path)
    assert cfg.tfidf.max_features == 2222


def test_artifact_paths(tmp_path):
    path = _write(tmp_path, "artifacts:\n  dir: out\n")
    cfg = load_config(path)
    assert str(cfg.artifacts.path("model")).endswith("tfidf_logreg_model.joblib")
    assert str(cfg.artifacts.path("model")).startswith("out")
