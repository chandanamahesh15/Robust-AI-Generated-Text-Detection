"""The deployable model: TF-IDF features + Logistic Regression.

Chosen for production because it is CPU-only, ~1 MB on disk, millisecond
inference, and within a few F1 points of the heavy transformers on this task.
Wrapped as a small class so training, persistence, and inference share one
definition of "what the model is" — no more globals scattered across cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.logging_utils import get_logger

logger = get_logger(__name__)


class TfidfLogRegModel:
    """TF-IDF vectorizer + Logistic Regression classifier."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.tfidf.max_features,
            ngram_range=(cfg.tfidf.ngram_min, cfg.tfidf.ngram_max),
            min_df=cfg.tfidf.min_df,
        )
        self.classifier = LogisticRegression(
            max_iter=cfg.tfidf.max_iter,
            C=cfg.tfidf.logreg_c,
            random_state=cfg.project.seed,
        )
        self._fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfLogRegModel":
        """Fit the vectorizer and classifier. Vectorizer is fit on TRAIN ONLY."""
        logger.info("Fitting TF-IDF vectorizer on %d training texts", len(texts))
        x = self.vectorizer.fit_transform(texts)
        logger.info("Training LogisticRegression on %s feature matrix", x.shape)
        self.classifier.fit(x, labels)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        self._check_fitted()
        return self.classifier.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Probability of the positive (AI) class."""
        self._check_fitted()
        return self.classifier.predict_proba(self.vectorizer.transform(texts))[:, 1]

    def save(self, cfg: Config | None = None) -> None:
        """Persist vectorizer and classifier as separate joblib artifacts."""
        self._check_fitted()
        cfg = cfg or self.cfg
        Path(cfg.artifacts.dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, cfg.artifacts.path("vectorizer"))
        joblib.dump(self.classifier, cfg.artifacts.path("model"))
        logger.info("Saved artifacts to %s/", cfg.artifacts.dir)

    @classmethod
    def load(cls, cfg: Config) -> "TfidfLogRegModel":
        """Load a previously trained model from artifacts on disk."""
        vec_path = cfg.artifacts.path("vectorizer")
        model_path = cfg.artifacts.path("model")
        for p in (vec_path, model_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Artifact '{p}' not found. Train first: python -m src.pipeline"
                )
        instance = cls(cfg)
        instance.vectorizer = joblib.load(vec_path)
        instance.classifier = joblib.load(model_path)
        instance._fitted = True
        logger.info("Loaded TF-IDF model from %s/", cfg.artifacts.dir)
        return instance
