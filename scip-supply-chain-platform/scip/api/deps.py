"""Shared API helpers: auth for mutating endpoints, DataFrame -> JSON, background job registry."""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import Header, HTTPException

from scip.core.config import get_settings
from scip.core.logging import get_logger

log = get_logger("api")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY is configured, mutating endpoints require the X-API-Key header."""
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.replace([np.inf, -np.inf], np.nan).to_json(orient="records", date_format="iso"))


def clean(obj):
    """JSON-safe copy: numpy scalars -> python, dates -> ISO strings, NaN/inf -> None."""
    def _fix(o):
        if isinstance(o, dict):
            return {k: _fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_fix(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return o
    return _fix(obj)


class Jobs:
    """Minimal in-process background job registry for long tasks (ingestion, training)."""
    _jobs: dict[str, dict] = {}
    _lock = threading.Lock()

    @classmethod
    def start(cls, name: str, fn, *args, **kwargs) -> dict:
        with cls._lock:
            running = [j for j in cls._jobs.values() if j["name"] == name and j["status"] == "running"]
            if running:
                return running[0]
            job = {"job_id": uuid.uuid4().hex[:12], "name": name, "status": "running",
                   "started_at": datetime.now().isoformat(timespec="seconds"), "result": None, "error": None}
            cls._jobs[job["job_id"]] = job

        def _run():
            try:
                job["result"] = clean(fn(*args, **kwargs))
                job["status"] = "completed"
            except Exception as exc:
                log.error("Job %s failed: %s", name, traceback.format_exc())
                job["status"], job["error"] = "failed", str(exc)
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")

        threading.Thread(target=_run, daemon=True).start()
        return job

    @classmethod
    def get(cls, job_id: str) -> dict | None:
        return cls._jobs.get(job_id)

    @classmethod
    def all(cls) -> list[dict]:
        return sorted(cls._jobs.values(), key=lambda j: j["started_at"], reverse=True)
