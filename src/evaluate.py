"""Evaluation utilities.

The notebook's ``evaluate_model`` computed metrics, printed a report, AND drew a
confusion matrix in one call, appending to a global ``results`` list. That mixes
computation with presentation and hides state. Here they are separate:

* :func:`compute_metrics`  -> pure, returns a dict, no I/O, unit-testable.
* :func:`plot_confusion_matrix` -> presentation only, for notebooks/reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.logging_utils import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    model_name: str,
    split: str = "validation",
) -> dict[str, Any]:
    """Compute classification metrics. Pure function — no printing, no plotting."""
    metrics = {
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    logger.info(
        "%s [%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
        model_name, split, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
    )
    return metrics


def classification_text_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_names: Sequence[str] = ("Human", "AI"),
) -> str:
    """Return sklearn's text report (logged by callers, not printed here)."""
    return classification_report(y_true, y_pred, target_names=list(target_names))


def save_metrics(rows: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    """Persist a list of metric dicts to CSV and return the DataFrame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(path, index=False)
    logger.info("Wrote %d metric rows -> %s", len(df), path)
    return df


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    model_name: str,
    target_names: Sequence[str] = ("Human", "AI"),
):  # pragma: no cover - presentation only
    """Draw a confusion matrix. Import matplotlib lazily so headless/serving
    environments don't pay for a GUI backend they never use."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(target_names))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return plt.gcf()
