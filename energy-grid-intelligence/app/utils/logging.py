"""Application-wide logging: console + rotating file in logs/."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config import LOG_DIR

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        root = logging.getLogger("egi")
        root.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = RotatingFileHandler(LOG_DIR / "egi.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        root.addHandler(sh)
        root.addHandler(fh)
        root.propagate = False
        for noisy in ("cmdstanpy", "prophet", "prophet.forecaster", "pymc", "matplotlib", "numba", "pytensor"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(f"egi.{name}")
