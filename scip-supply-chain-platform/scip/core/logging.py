"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger("scip")
    root.setLevel(level.upper())
    root.addHandler(handler)
    root.propagate = False
    for noisy in ("cmdstanpy", "prophet", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        from scip.core.config import get_settings
        configure_logging(get_settings().log_level)
    return logging.getLogger(f"scip.{name}")
