"""FastAPI application — the single backend for the dashboard, n8n workflows and external clients.

Run: uvicorn scip.api.main:app --host 0.0.0.0 --port 8000   (OpenAPI docs at /docs)
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from scip.api.routers import analytics, copilot, operations, system
from scip.core.config import get_settings
from scip.core.db import init_db
from scip.core.logging import get_logger

log = get_logger("api")

app = FastAPI(title="SCIP — Supply Chain Intelligence & Autonomous Operations API", version="1.0.0",
              description="Unified REST backend: data, forecasting, inventory, suppliers, procurement workflow, "
                          "logistics optimisation, AI Copilot and automation.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    init_db()
    log.info("API started (%s)", get_settings().environment)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:     # last-resort handler: never leak stack traces to clients
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": type(exc).__name__})
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, (time.time() - t) * 1000)
    return response


for r in (system.router, analytics.router, operations.router, copilot.router):
    app.include_router(r)
