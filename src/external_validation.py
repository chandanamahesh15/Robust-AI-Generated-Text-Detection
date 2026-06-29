"""External (cross-dataset) validation against an unseen HuggingFace dataset.

Extracted from the notebook's final block. Reshapes the wide external dataset
(one human column + several model columns) into the binary
``text`` / ``generated`` schema, then scores the deployable model on it.

Cross-dataset evaluation is the honest test of a detector: high in-distribution
scores mean little if the model collapses on text from models/domains it never saw.

Requires the ``datasets`` library (research extra) for the download step only.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data import clean_text
from src.logging_utils import get_logger

logger = get_logger(__name__)


def load_external_eval_df(cfg: Config) -> pd.DataFrame:
    """Download and reshape the external dataset into text/generated rows."""
    from datasets import load_dataset  # research-only dependency

    ev = cfg.external_validation
    logger.info("Loading external dataset: %s", ev["hf_dataset"])
    raw = load_dataset(ev["hf_dataset"])["train"].to_pandas()

    human_col = ev["human_col"]
    if human_col not in raw.columns:
        raise ValueError(f"'{human_col}' not found in external columns: {list(raw.columns)}")
    ai_cols = [c for c in ev["ai_cols"] if c in raw.columns]
    if not ai_cols:
        raise ValueError(f"No AI columns from {ev['ai_cols']} found in external dataset.")

    human = raw[[human_col]].rename(columns={human_col: "text"})
    human["generated"] = 0
    human["source_model"] = "human"

    ai_frames = []
    for col in ai_cols:
        frame = raw[[col]].rename(columns={col: "text"})
        frame["generated"] = 1
        frame["source_model"] = col
        ai_frames.append(frame)

    df = pd.concat([human, *ai_frames], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, cfg))

    min_wc = ev.get("min_word_count", 0)
    if min_wc > 0:
        df = df[df["clean_text"].str.split().str.len() >= min_wc].reset_index(drop=True)

    logger.info("External eval set: %d rows\n%s",
                len(df), df["generated"].value_counts().to_string())
    return df
