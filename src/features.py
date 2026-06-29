"""Exploratory text features (length, punctuation, stopword ratio).

Extracted from the EDA cell. These are descriptive features for analysis and
plots in ``notebooks/`` — they are NOT inputs to the deployable model, so they
live apart from the serving path.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_PUNCT_RE = re.compile(r"[.,!?;:]")
_STOPWORDS = set(ENGLISH_STOP_WORDS)


def add_length_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of ``df`` with descriptive text-length columns added."""
    out = df.copy()
    texts = out[text_col].astype(str)

    out["word_count"] = texts.str.split().str.len()
    out["char_count"] = texts.str.len()
    out["sent_count"] = texts.apply(
        lambda x: len([s for s in _SENT_SPLIT_RE.split(x) if s.strip()])
    )
    out["avg_word_len"] = texts.apply(
        lambda x: float(np.mean([len(w) for w in x.split()])) if x.split() else 0.0
    )
    out["punct_count"] = texts.apply(lambda x: len(_PUNCT_RE.findall(x)))
    out["stopword_ratio"] = texts.apply(
        lambda x: sum(1 for w in x.lower().split() if w in _STOPWORDS)
        / max(len(x.split()), 1)
    )
    return out
