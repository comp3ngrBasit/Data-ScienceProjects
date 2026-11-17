"""Logging setup: console + rotating file in data/logs/app.log."""
from __future__ import annotations

import json
import logging
import logging.handlers

from app.core.config import Settings

_CONFIGURED = False
_STD = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Structured logs: timestamp, level, logger, message and any ``extra={...}`` fields (run_id, video_id, stage, ...)."""

    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "logger": record.name,
             "message": record.getMessage()}
        d.update({k: v for k, v in record.__dict__.items() if k not in _STD and not k.startswith("_")})
        if record.exc_info:
            d["exception"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable lines; extra fields are appended as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = {k: v for k, v in record.__dict__.items() if k not in _STD and not k.startswith("_")}
        return base + ("" if not extra else " | " + " ".join(f"{k}={v}" for k, v in extra.items()))


def configure_logging(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    fmt = JsonFormatter() if getattr(settings, "log_format", "text") == "json" else TextFormatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        settings.logs_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quieter third-party loggers
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)
    _CONFIGURED = True
