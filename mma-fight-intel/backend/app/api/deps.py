from __future__ import annotations

from fastapi import Request

from app.db.database import get_db  # noqa: F401  (re-export)
from app.services.job_manager import JobManager
from app.services.storage import Storage


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager
