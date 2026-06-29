"""Training pipeline for the deployable TF-IDF + LogReg model.

Run:
    python -m src.pipeline                       # uses config/config.yaml
    python -m src.pipeline --config other.yaml
    AVH__TFIDF__MAX_FEATURES=20000 python -m src.pipeline   # env override

This is the *production* training path. The heavy research models live in
``src/models/`` and are trained by their own scripts; they are not imported
here so this stays a fast, CPU-only, dependency-light entry point.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.data import load_dataframe, make_splits
from src.evaluate import classification_text_report, compute_metrics, save_metrics
from src.logging_utils import configure_logging, get_logger
from src.models.tfidf import TfidfLogRegModel

logger = get_logger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run(cfg: Config) -> None:
    set_seeds(cfg.project.seed)

    df = load_dataframe(cfg)
    splits = make_splits(df, cfg)

    model = TfidfLogRegModel(cfg)
    model.fit(splits.x_clean_train, splits.y_train)

    rows = []
    for name, x, y in [
        ("validation", splits.x_clean_val, splits.y_val),
        ("test", splits.x_clean_test, splits.y_test),
    ]:
        preds = model.predict(x)
        rows.append(compute_metrics(y, preds, "TF-IDF + LogReg", name))
        logger.info("\n%s", classification_text_report(y, preds))

    model.save(cfg)
    save_metrics(rows, cfg.artifacts.path("metrics"))
    logger.info("Pipeline complete. Artifacts in %s/", cfg.artifacts.dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deployable AI-vs-Human detector.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, cfg.logging.format)
    try:
        run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
