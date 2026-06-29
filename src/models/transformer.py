"""Transformer fine-tuning (research track, PyTorch + HuggingFace).

Extracted from the notebook's ``train_transformer`` cell and the BERT/RoBERTa
runs. Two fixes versus the notebook:

1. The function now ALWAYS returns the model + tokenizer (the notebook's
   original version forgot to, then patched it mid-run — exactly the kind of
   "function mutated across cells" bug that modularizing prevents).
2. Subsetting is a seeded, explicit helper instead of inline ``np.random``.

Requires torch + transformers (requirements-research.txt) and a GPU.
NOT on the serving path; not executed in CI.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.config import Config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _import_torch():
    import torch  # local import keeps the serving image torch-free
    return torch


def get_device():
    torch = _import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


class TextDataset:
    """PyTorch Dataset that tokenizes on the fly. (Subclasses torch Dataset at
    runtime; declared lazily so importing this module needs no torch.)"""

    def __new__(cls, *args, **kwargs):
        torch = _import_torch()
        from torch.utils.data import Dataset

        class _Impl(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = list(texts)
                self.labels = list(labels)
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx], max_length=self.max_len,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": torch.tensor(self.labels[idx], dtype=torch.long),
                }

        return _Impl(*args, **kwargs)


def subsample(series, n: int | None, seed: int):
    """Deterministically take ``n`` rows (or all if n is None)."""
    if n is None or n >= len(series):
        return series.reset_index(drop=True)
    idx = np.random.default_rng(seed).choice(len(series), n, replace=False)
    return series.iloc[idx].reset_index(drop=True)


def fine_tune(
    model_name: str,
    x_train: Sequence[str], x_val: Sequence[str], x_test: Sequence[str],
    y_train: Sequence[int], y_val: Sequence[int], y_test: Sequence[int],
    cfg: Config,
):
    """Fine-tune a HF sequence-classification model.

    Returns (val_preds, test_preds, model, tokenizer).
    """
    torch = _import_torch()
    from sklearn.metrics import accuracy_score
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    tcfg = cfg.transformer
    device = get_device()
    logger.info("Loading %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    def loader(x, y, shuffle):
        ds = TextDataset(x, y, tokenizer, tcfg["max_len"])
        bs = tcfg["batch_size"] if shuffle else tcfg["batch_size"] * 2
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val, False)
    test_loader = loader(x_test, y_test, False)

    optimizer = AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    def _predict(data_loader) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in data_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds.extend(out.logits.argmax(-1).cpu().numpy())
        return np.array(preds)

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        val_acc = accuracy_score(list(y_val), _predict(val_loader))
        logger.info(
            "%s | epoch %d/%d | loss=%.4f | val_acc=%.4f",
            model_name, epoch, tcfg["epochs"], total_loss / len(train_loader), val_acc,
        )

    return _predict(val_loader), _predict(test_loader), model, tokenizer
