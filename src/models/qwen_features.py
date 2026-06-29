"""Qwen2.5 frozen-embedding feature extractor + LogReg head (research track).

Extracted from the notebook's Qwen cells: pass text through a frozen Qwen
model, mean-pool the last hidden state over non-padding tokens, then fit a
Logistic Regression on the pooled vectors. Requires torch + transformers + GPU.
NOT on the serving path; not executed in CI.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.config import Config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def load_qwen(cfg: Config):
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = cfg.qwen["model_name"]
    logger.info("Loading %s for feature extraction", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device


def extract_embeddings(
    texts: Sequence[str], tokenizer, model, device, cfg: Config
) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, shape (N, hidden_dim)."""
    import torch

    batch_size = cfg.qwen["batch_size"]
    max_len = cfg.qwen["max_len"]
    texts = list(texts)
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size], max_length=max_len,
            padding=True, truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        chunks.append(pooled.cpu().float().numpy())
        if (i // batch_size) % 50 == 0:
            logger.debug("Embedded %d/%d", min(i + batch_size, len(texts)), len(texts))
    return np.vstack(chunks)


def train_head(emb_train: np.ndarray, y_train: Sequence[int], cfg: Config):
    """Standardise then fit LogReg on frozen embeddings. Returns (scaler, clf)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(emb_train)  # fit on train only
    clf = LogisticRegression(
        max_iter=cfg.qwen["max_iter"], C=cfg.qwen["logreg_c"], random_state=cfg.project.seed
    )
    clf.fit(scaler.transform(emb_train), y_train)
    return scaler, clf
