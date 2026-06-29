"""Logging configuration — the project-wide replacement for ``print``.

Call :func:`configure_logging` once at the start of any entry point
(``pipeline.py``, ``app/main.py``, scripts), then use
``logger = get_logger(__name__)`` in every module.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """Configure the root logger once. Idempotent across repeated calls."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)
    # Quiet noisy third-party loggers instead of a blanket warnings filter.
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
