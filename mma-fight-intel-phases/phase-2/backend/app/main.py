"""FastAPI application entry point.

Run with:  python run.py      (or)  uvicorn app.main:app --port 8000
Swagger UI: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import fighters, jobs, system, videos
from app.vision import api as vision_api
from app.vision.job_manager import VisionJobManager
from app.core.config import BACKEND_DIR, get_settings
from app.core.logging_config import configure_logging
from sqlalchemy import select

from app.db.database import create_all, get_session_factory, init_engine, session_scope
from app.db.models import Video
from app.services.ffmpeg_tools import get_ffmpeg_tools
from app.services.job_manager import JobManager
from app.services.settings_service import get_processing_settings
from app.services.storage import Storage

log = logging.getLogger("app")
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    configure_logging(settings)
    init_engine(settings.resolved_database_url)
    create_all()

    storage = Storage(settings)
    removed = storage.cleanup_temp()
    if removed:
        log.info("Removed %d stale partial upload(s)", removed)
    with session_scope() as db:
        known = set(db.scalars(select(Video.id)).all())
    orphans = storage.sweep_orphans(known)
    if orphans:
        log.info("Removed %d orphaned file(s)/folder(s) from data directory", orphans)
    tools = get_ffmpeg_tools()
    log.info("FFmpeg: %s | FFprobe: %s (source: %s)", tools.ffmpeg, tools.ffprobe, tools.source)
    log.info("Data directory: %s", settings.data_dir.resolve())

    job_manager = JobManager(storage, workers=settings.worker_threads)
    app.state.storage = storage
    app.state.job_manager = job_manager
    vision_manager = VisionJobManager(storage)
    app.state.vision_manager = vision_manager
    job_manager.start()
    vision_manager.start()
    try:
        yield
    finally:
        vision_manager.stop()
        job_manager.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Phase 1 API for the AI-Powered MMA Fight Intelligence system: video upload, "
            "library management, metadata extraction, frame sampling, fighter profiles "
            "and background processing jobs. Phase 2 adds YOLO pose detection, ByteTrack/BoT-SORT tracking, "
            "pose landmarks, movement analysis and annotated videos under /api/vision."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        """Fail fast on uploads larger than the configured limit, before the body is received."""
        if request.method == "POST" and request.url.path == "/api/videos/upload":
            length = request.headers.get("content-length")
            if length and length.isdigit():
                db = get_session_factory()()
                try:
                    limit_mb = min(get_processing_settings(db).max_upload_size_mb, settings.hard_max_upload_size_mb)
                finally:
                    db.close()
                # allow a little multipart overhead per request
                if int(length) > limit_mb * 1024 * 1024 + 1024 * 1024:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Upload exceeds the maximum size of {limit_mb} MB"},
                    )
        return await call_next(request)

    app.include_router(videos.router)
    app.include_router(fighters.router)
    app.include_router(jobs.router)
    app.include_router(system.router)
    app.include_router(vision_api.router)

    # Serve the built frontend (npm run build) from the same server, if present.
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        assets = FRONTEND_DIST / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            candidate = (FRONTEND_DIST / full_path).resolve()
            if full_path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
