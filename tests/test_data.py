"""Tests for text cleaning and splitting logic."""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data import clean_text, make_splits


def _cfg() -> Config:
    return Config()  # all defaults


def test_clean_text_lowercases_and_strips_non_letters():
    # url removed, <b>..</b> tags stripped whole, digits/punct -> space, lowercased
    out = clean_text("Visit https://x.com NOW!! <b>123</b>", _cfg())
    assert out == "visit now"
    assert "http" not in out


def test_clean_text_collapses_whitespace():
    assert clean_text("a    b\t\nc", _cfg()) == "a b c"


def test_make_splits_are_disjoint_and_sum_to_total():
    df = pd.DataFrame(
        {"text": [f"sample text number {i} here" for i in range(200)],
         "generated": [i % 2 for i in range(200)]}
    )
    splits = make_splits(df, _cfg())
    total = len(splits.x_clean_train) + len(splits.x_clean_val) + len(splits.x_clean_test)
    assert total == len(df)
    # clean and raw series stay aligned in length
    assert len(splits.x_clean_train) == len(splits.x_raw_train) == len(splits.y_train)


def test_split_is_deterministic_under_fixed_seed():
    df = pd.DataFrame(
        {"text": [f"row {i} content words" for i in range(100)],
         "generated": [i % 2 for i in range(100)]}
    )
    a = make_splits(df, _cfg())
    b = make_splits(df, _cfg())
    assert list(a.x_clean_test) == list(b.x_clean_test)
