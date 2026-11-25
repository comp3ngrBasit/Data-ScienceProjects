"""FastAPI backend.  Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000

The API serves results produced by the pipeline (it never recomputes models on request), so it is
fast, stateless and can sit behind the Streamlit dashboard today and a React frontend later.
A protected endpoint can trigger pipeline stages in the background.
"""
from __future__ import annotations

import os
import threading

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.results import router
from app.utils.logging import get_logger

log = get_logger("api")
app = FastAPI(title="Energy Grid Intelligence API", version="1.0.0",
              description="Analytical results for household consumption and grid intelligence.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("EGI_CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")
_lock = threading.Lock()


@app.post("/api/v1/pipeline/run")
def trigger(stage: str | None = None, force: bool = False, x_api_key: str | None = Header(None)) -> dict:
    expected = os.getenv("EGI_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(403, "pipeline trigger requires EGI_API_KEY")
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "pipeline already running")

    def job():
        try:
            from app.pipeline import run

            run([stage] if stage else None, force)
        finally:
            _lock.release()

    threading.Thread(target=job, daemon=True).start()
    return {"started": True, "stage": stage or "all"}


@app.get("/")
def root() -> dict:
    return {"name": "Energy Grid Intelligence API", "docs": "/docs", "health": "/api/v1/health"}
