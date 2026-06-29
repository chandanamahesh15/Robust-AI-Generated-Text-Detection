"""BiLSTM with a from-scratch embedding (research track, Keras/TensorFlow).

Extracted from the notebook's LSTM cells. This is part of the *experiment* that
justified choosing TF-IDF for production — it is NOT served. Requires
tensorflow (see requirements-research.txt) and ideally a GPU.

NOTE: not executed in CI; it is GPU-bound and heavy. Kept here so the
experiment is reproducible, not so it ships.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.config import Config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def build_tokenizer(train_texts: Sequence[str], cfg: Config):
    """Fit a Keras tokenizer on TRAIN ONLY (avoids vocabulary leakage)."""
    from tensorflow.keras.preprocessing.text import Tokenizer  # local import

    tok = Tokenizer(num_words=cfg.bilstm["max_vocab"], oov_token="<OOV>")
    tok.fit_on_texts(list(train_texts))
    return tok


def to_padded(tokenizer, texts: Sequence[str], cfg: Config) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seqs = tokenizer.texts_to_sequences(list(texts))
    return pad_sequences(
        seqs, maxlen=cfg.bilstm["max_len"], padding="post", truncating="post"
    )


def build_model(cfg: Config):
    """Bidirectional LSTM over a randomly-initialised, trained-from-scratch embedding."""
    from tensorflow.keras.layers import (
        LSTM, Bidirectional, Dense, Dropout, Embedding,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Embedding(
                input_dim=cfg.bilstm["max_vocab"],
                output_dim=cfg.bilstm["embed_dim"],
                input_length=cfg.bilstm["max_len"],
                name="scratch_embedding",
            ),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ],
        name="BiLSTM_scratch_embedding",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(
    x_train: np.ndarray, y_train: Sequence[int],
    x_val: np.ndarray, y_val: Sequence[int],
    cfg: Config,
) -> tuple[Any, Any]:
    """Train with early stopping. Returns (model, history)."""
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_model(cfg)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=cfg.bilstm["patience"], restore_best_weights=True
    )
    logger.info("Training BiLSTM for up to %d epochs", cfg.bilstm["epochs"])
    history = model.fit(
        x_train, np.asarray(y_train),
        validation_data=(x_val, np.asarray(y_val)),
        epochs=cfg.bilstm["epochs"],
        batch_size=cfg.bilstm["batch_size"],
        callbacks=[early_stop],
    )
    return model, history


def predict(model, x: np.ndarray) -> np.ndarray:
    return (model.predict(x) > 0.5).astype(int).flatten()
