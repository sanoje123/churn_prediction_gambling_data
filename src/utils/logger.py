"""Small logging helper so every module logs consistently to stdout."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "churn", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout (container-friendly)."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root = logging.getLogger("churn")
        root.setLevel(level)
        root.handlers.clear()
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    logger = logging.getLogger(f"churn.{name}" if name != "churn" else "churn")
    logger.setLevel(level)
    return logger
