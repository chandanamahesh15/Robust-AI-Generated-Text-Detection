"""Data loading, text cleaning, and train/val/test splitting.

Extracted from the notebook's load / clean / split cells. All Colab-specific
``/content`` paths are gone; everything is driven by :class:`~src.config.Config`.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.logging_utils import get_logger

logger = get_logger(__name__)

_URL_RE = re.compile(r"http\S+|www\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")
_WS_RE = re.compile(r"\s+")


class DataSplits(NamedTuple):
    """Both cleaned and raw text are returned: classical/LSTM models use
    ``*_clean``; transformers use ``*_raw`` (they tokenize raw text themselves)."""

    x_clean_train: pd.Series
    x_clean_val: pd.Series
    x_clean_test: pd.Series
    x_raw_train: pd.Series
    x_raw_val: pd.Series
    x_raw_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def clean_text(text: str, cfg: Config) -> str:
    """Lightweight cleaning for classical ML + the BiLSTM.

    Transformers (BERT/RoBERTa/Qwen) must receive RAW text, so this is applied
    only to the ``clean_text`` column.
    """
    if cfg.preprocessing.lowercase:
        text = text.lower()
    if cfg.preprocessing.strip_urls:
        text = _URL_RE.sub("", text)
    if cfg.preprocessing.strip_html:
        text = _HTML_RE.sub("", text)
    if cfg.preprocessing.letters_only:
        text = _NON_LETTER_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_dataframe(cfg: Config) -> pd.DataFrame:
    """Extract the source zip (if needed), read the CSV, and apply basic hygiene.

    Raises:
        FileNotFoundError: if the source zip or expected CSV is missing.
    """
    zip_path = Path(cfg.data.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found at '{zip_path}'. "
            "Download AI_Human.csv.zip into data/raw/ (see README)."
        )

    extract_dir = Path(cfg.data.extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    csv_path = extract_dir / cfg.data.csv_name

    if not csv_path.exists():
        logger.info("Extracting %s -> %s", zip_path, extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected '{csv_path}' after extraction; not found.")

    logger.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)

    text_col, label_col = cfg.data.text_col, cfg.data.label_col
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {csv_path}")

    df[label_col] = df[label_col].astype(int)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)

    if cfg.data.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        logger.info("Dropped %d duplicate rows (leakage guard)", before - len(df))

    if cfg.data.min_word_count > 0:
        mask = df[text_col].str.split().str.len() >= cfg.data.min_word_count
        df = df[mask].reset_index(drop=True)

    logger.info("Loaded %d rows | class balance:\n%s",
                len(df), df[label_col].value_counts().to_string())
    return df


def make_splits(df: pd.DataFrame, cfg: Config) -> DataSplits:
    """Stratified 70/15/15 train/val/test split, carrying clean + raw text."""
    text_col, label_col = cfg.data.text_col, cfg.data.label_col
    x_clean = df[text_col].apply(lambda t: clean_text(t, cfg))
    x_raw = df[text_col]
    y = df[label_col]
    strat = y if cfg.split.stratify else None

    xc_tr, xc_tmp, xr_tr, xr_tmp, y_tr, y_tmp = train_test_split(
        x_clean, x_raw, y,
        test_size=cfg.split.test_size,
        random_state=cfg.project.seed,
        stratify=strat,
    )
    strat_tmp = y_tmp if cfg.split.stratify else None
    xc_val, xc_te, xr_val, xr_te, y_val, y_te = train_test_split(
        xc_tmp, xr_tmp, y_tmp,
        test_size=cfg.split.val_test_ratio,
        random_state=cfg.project.seed,
        stratify=strat_tmp,
    )

    logger.info("Split sizes -> train=%d val=%d test=%d",
                len(xc_tr), len(xc_val), len(xc_te))
    return DataSplits(
        xc_tr.reset_index(drop=True), xc_val.reset_index(drop=True), xc_te.reset_index(drop=True),
        xr_tr.reset_index(drop=True), xr_val.reset_index(drop=True), xr_te.reset_index(drop=True),
        y_tr.reset_index(drop=True), y_val.reset_index(drop=True), y_te.reset_index(drop=True),
    )
