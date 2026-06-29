"""Typed configuration loading.

Loads ``config/config.yaml`` into nested dataclasses so the rest of the code
gets autocomplete and type checking instead of stringly-typed dict access.
Environment variables of the form ``AVH__SECTION__KEY`` override file values,
which keeps secrets and deployment-specific paths out of source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
_ENV_PREFIX = "AVH"  # e.g. AVH__TFIDF__MAX_FEATURES=20000


@dataclass
class ProjectConfig:
    name: str = "ai-vs-human-text-detection"
    seed: int = 42


@dataclass
class DataConfig:
    zip_path: str = "data/raw/AI_Human.csv.zip"
    extract_dir: str = "data/extracted"
    csv_name: str = "AI_Human.csv"
    text_col: str = "text"
    label_col: str = "generated"
    drop_duplicates: bool = True
    min_word_count: int = 0


@dataclass
class SplitConfig:
    test_size: float = 0.30
    val_test_ratio: float = 0.50
    stratify: bool = True


@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    strip_urls: bool = True
    strip_html: bool = True
    letters_only: bool = True


@dataclass
class TfidfConfig:
    max_features: int = 10_000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    logreg_c: float = 1.0
    max_iter: int = 1000


@dataclass
class ArtifactsConfig:
    dir: str = "artifacts"
    model_file: str = "tfidf_logreg_model.joblib"
    vectorizer_file: str = "tfidf_vectorizer.joblib"
    metrics_file: str = "tfidf_metrics.csv"

    def path(self, which: str) -> Path:
        mapping = {
            "model": self.model_file,
            "vectorizer": self.vectorizer_file,
            "metrics": self.metrics_file,
        }
        return Path(self.dir) / mapping[which]


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Research-model blocks are kept as raw dicts: they are optional and only
    # consumed by GPU scripts, so they don't need strict schemas here.
    bilstm: dict[str, Any] = field(default_factory=dict)
    transformer: dict[str, Any] = field(default_factory=dict)
    qwen: dict[str, Any] = field(default_factory=dict)
    external_validation: dict[str, Any] = field(default_factory=dict)


def _build(cls: type[Any], data: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a dict, ignoring unknown keys."""
    if not is_dataclass(cls):
        return data
    # Resolve string annotations (PEP 563) to real types before introspection.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        ftype = hints.get(f.name, f.type)
        # ftype from get_type_hints is always a class for dataclass fields; the
        # cast tells mypy what is_dataclass's TypeGuard can't express.
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[f.name] = _build(cast("type[Any]", ftype), data[f.name])
        else:
            kwargs[f.name] = data[f.name]
    return cast("type[Any]", cls)(**kwargs)


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """AVH__TFIDF__MAX_FEATURES=20000 -> raw['tfidf']['max_features'] = '20000'."""
    for env_key, value in os.environ.items():
        if not env_key.startswith(f"{_ENV_PREFIX}__"):
            continue
        parts = env_key.lower().split("__")[1:]
        cursor = raw
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)  # cast "20000" -> int, "true" -> bool
    return raw


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read YAML, apply env overrides, and return a typed ``Config``.

    Raises:
        FileNotFoundError: if the config file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. Pass --config or create config/config.yaml."
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _apply_env_overrides(raw)
    return _build(Config, raw)