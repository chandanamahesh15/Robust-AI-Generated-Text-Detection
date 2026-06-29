"""Reproducible multi-model benchmark.

Runs every model family on the same splits and writes a results table to
``reports/benchmark_results.csv``. This is the *research* entry point that
justifies the production model choice — it is NOT the serving path.

CPU models (baselines, TF-IDF) always run. The GPU-bound models (BiLSTM, BERT,
RoBERTa, Qwen) run only if their libraries are installed; otherwise they are
skipped with a clear log message, so the script works on a laptop and a GPU box
alike.

    python -m src.benchmark                 # full benchmark
    python -m src.benchmark --skip-heavy    # baselines + TF-IDF only

The committed ``reports/benchmark_results.csv`` holds a captured GPU run so the
numbers are visible without re-running anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier

from src.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.data import DataSplits, load_dataframe, make_splits
from src.evaluate import compute_metrics, save_metrics
from src.logging_utils import configure_logging, get_logger
from src.models.tfidf import TfidfLogRegModel

logger = get_logger(__name__)


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_baselines(splits: DataSplits, cfg: Config) -> list[dict]:
    rows = []
    for strategy, name in [("most_frequent", "Majority Class Baseline"),
                           ("stratified", "Stratified Baseline")]:
        clf = DummyClassifier(strategy=strategy, random_state=cfg.project.seed)
        clf.fit(splits.x_clean_train.to_frame(), splits.y_train)
        preds = clf.predict(splits.x_clean_val.to_frame())
        rows.append(compute_metrics(splits.y_val, preds, name, "validation"))
    return rows


def run_tfidf(splits: DataSplits, cfg: Config) -> list[dict]:
    model = TfidfLogRegModel(cfg).fit(splits.x_clean_train, splits.y_train)
    return [
        compute_metrics(splits.y_val, model.predict(splits.x_clean_val), "TF-IDF + LogReg", "validation"),
        compute_metrics(splits.y_test, model.predict(splits.x_clean_test), "TF-IDF + LogReg", "test"),
    ]


def run_bilstm(splits: DataSplits, cfg: Config) -> list[dict]:
    from src.models import bilstm

    tok = bilstm.build_tokenizer(splits.x_clean_train, cfg)
    x_tr = bilstm.to_padded(tok, splits.x_clean_train, cfg)
    x_val = bilstm.to_padded(tok, splits.x_clean_val, cfg)
    x_te = bilstm.to_padded(tok, splits.x_clean_test, cfg)
    model, _ = bilstm.train(x_tr, splits.y_train, x_val, splits.y_val, cfg)
    return [
        compute_metrics(splits.y_val, bilstm.predict(model, x_val), "BiLSTM (scratch embed)", "validation"),
        compute_metrics(splits.y_test, bilstm.predict(model, x_te), "BiLSTM (scratch embed)", "test"),
    ]


def run_transformers(splits: DataSplits, cfg: Config) -> list[dict]:
    from src.models import transformer

    seed = cfg.project.seed
    n = cfg.transformer.get("n_finetune")
    xr_tr = transformer.subsample(splits.x_raw_train, n, seed)
    yr_tr = transformer.subsample(splits.y_train, n, seed)
    nv = None if n is None else n // 4
    xr_val, yr_val = transformer.subsample(splits.x_raw_val, nv, seed), transformer.subsample(splits.y_val, nv, seed)
    xr_te, yr_te = transformer.subsample(splits.x_raw_test, nv, seed), transformer.subsample(splits.y_test, nv, seed)

    rows = []
    for model_name in cfg.transformer.get("models", []):
        val_preds, test_preds, _, _ = transformer.fine_tune(
            model_name, xr_tr, xr_val, xr_te, yr_tr, yr_val, yr_te, cfg
        )
        label = f"{model_name.split('-')[0].upper()} (fine-tuned)"
        rows.append(compute_metrics(yr_val, val_preds, label, "validation"))
        rows.append(compute_metrics(yr_te, test_preds, label, "test"))
    return rows


def run_qwen(splits: DataSplits, cfg: Config) -> list[dict]:
    from src.models import qwen_features as qf
    from src.models.transformer import subsample

    seed = cfg.project.seed
    n = cfg.transformer.get("n_finetune")
    nv = None if n is None else n // 4
    xr_tr, yr_tr = subsample(splits.x_raw_train, n, seed), subsample(splits.y_train, n, seed)
    xr_val, yr_val = subsample(splits.x_raw_val, nv, seed), subsample(splits.y_val, nv, seed)
    xr_te, yr_te = subsample(splits.x_raw_test, nv, seed), subsample(splits.y_test, nv, seed)

    tok, model, device = qf.load_qwen(cfg)
    emb_tr = qf.extract_embeddings(xr_tr, tok, model, device, cfg)
    emb_val = qf.extract_embeddings(xr_val, tok, model, device, cfg)
    emb_te = qf.extract_embeddings(xr_te, tok, model, device, cfg)
    scaler, clf = qf.train_head(emb_tr, yr_tr, cfg)
    name = "Qwen2.5-0.5B + LR (frozen)"
    return [
        compute_metrics(yr_val, clf.predict(scaler.transform(emb_val)), name, "validation"),
        compute_metrics(yr_te, clf.predict(scaler.transform(emb_te)), name, "test"),
    ]


def run(cfg: Config, skip_heavy: bool) -> None:
    set_seeds(cfg.project.seed)
    df = load_dataframe(cfg)
    splits = make_splits(df, cfg)

    rows: list[dict] = []
    rows += run_baselines(splits, cfg)
    rows += run_tfidf(splits, cfg)

    if skip_heavy:
        logger.info("--skip-heavy set; skipping BiLSTM/transformers/Qwen.")
    else:
        heavy = [
            ("tensorflow", "BiLSTM", run_bilstm),
            ("torch", "transformers (BERT/RoBERTa)", run_transformers),
            ("torch", "Qwen embeddings", run_qwen),
        ]
        for dep, label, fn in heavy:
            if not _available(dep):
                logger.warning("Skipping %s — '%s' not installed (pip install -e '.[research]').", label, dep)
                continue
            try:
                rows += fn(splits, cfg)
            except Exception as exc:  # noqa: BLE001 - one model failing shouldn't kill the run
                logger.error("%s failed: %s", label, exc)

    out = Path("reports/benchmark_results.csv")
    save_metrics(rows, out)
    logger.info("Benchmark complete -> %s", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the multi-model benchmark.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--skip-heavy", action="store_true", help="Baselines + TF-IDF only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, cfg.logging.format)
    try:
        run(cfg, args.skip_heavy)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Benchmark failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
