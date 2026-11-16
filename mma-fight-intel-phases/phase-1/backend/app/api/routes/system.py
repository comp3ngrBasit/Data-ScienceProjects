"""Overview stats, settings and system info."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db, get_storage
from app.core.config import get_settings
from app.db.models import Fighter, JobStatus, ProcessingJob, SampledFrame, Video, VideoStatus
from app.schemas.settings import ProcessingSettings, SystemInfo
from app.services import video_service as vs
from app.services.ffmpeg_tools import get_ffmpeg_tools
from app.services.settings_service import get_processing_settings, save_processing_settings
from app.services.storage import Storage

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", summary="Liveness check")
def health(db: Session = Depends(get_db)):
    tools = get_ffmpeg_tools()
    active = db.scalar(
        select(func.count()).select_from(ProcessingJob).where(ProcessingJob.status.in_((JobStatus.queued, JobStatus.processing)))
    ) or 0
    return {"status": "ok", "ffmpeg": bool(tools.ffmpeg), "ffprobe": bool(tools.ffprobe), "queue_size": active}


@router.get("/stats/overview", summary="Dashboard overview statistics (computed from the database)")
def overview(db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    status_counts = dict(db.execute(select(Video.status, func.count()).group_by(Video.status)).all())
    total_videos = sum(status_counts.values())
    total_fighters = db.scalar(select(func.count()).select_from(Fighter)) or 0
    total_duration = db.scalar(select(func.coalesce(func.sum(Video.duration_seconds), 0.0))) or 0.0
    total_bytes = db.scalar(select(func.coalesce(func.sum(Video.file_size), 0))) or 0
    total_frames = db.scalar(select(func.count()).select_from(SampledFrame)) or 0

    recent_videos = db.scalars(
        select(Video).options(selectinload(Video.fighters), selectinload(Video.target_fighter))
        .order_by(Video.uploaded_at.desc()).limit(6)
    ).all()
    recent_jobs = db.scalars(
        select(ProcessingJob).options(selectinload(ProcessingJob.video))
        .order_by(func.coalesce(ProcessingJob.finished_at, ProcessingJob.started_at, ProcessingJob.created_at).desc())
        .limit(8)
    ).all()

    # Uploads per day for the last 14 days (UTC days)
    since = datetime.now(timezone.utc) - timedelta(days=13)
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)
    per_day: dict[str, int] = {}
    for (ts,) in db.execute(select(Video.uploaded_at).where(Video.uploaded_at >= since)).all():
        key = ts.date().isoformat()
        per_day[key] = per_day.get(key, 0) + 1
    uploads_by_day = [
        {"date": (since + timedelta(days=i)).date().isoformat(), "count": per_day.get((since + timedelta(days=i)).date().isoformat(), 0)}
        for i in range(14)
    ]

    res_rows = db.execute(select(Video.height, Video.width).where(Video.height.is_not(None))).all()
    res_buckets: dict[str, int] = {}
    from app.pipeline.stages.quality import resolution_class

    for h, w in res_rows:
        k = resolution_class(w, h)
        res_buckets[k] = res_buckets.get(k, 0) + 1

    awaiting = sum(status_counts.get(s, 0) for s in (VideoStatus.uploaded, VideoStatus.queued, VideoStatus.processing))
    return {
        "total_videos": total_videos,
        "total_fighters": total_fighters,
        "videos_processed": status_counts.get(VideoStatus.completed, 0),
        "videos_awaiting": awaiting,
        "videos_failed": status_counts.get(VideoStatus.failed, 0),
        "status_counts": {s.value: status_counts.get(s, 0) for s in VideoStatus},
        "total_duration_seconds": round(float(total_duration), 2),
        "total_bytes": int(total_bytes),
        "total_sampled_frames": total_frames,
        "storage_bytes": storage.usage(),
        "uploads_by_day": uploads_by_day,
        "resolution_breakdown": [{"label": k, "count": v} for k, v in sorted(res_buckets.items())],
        "recent_videos": [vs.to_summary(v).model_dump(mode="json") for v in recent_videos],
        "recent_jobs": [vs.job_to_out(j).model_dump(mode="json") for j in recent_jobs],
        "active_jobs": db.scalar(
            select(func.count()).select_from(ProcessingJob).where(ProcessingJob.status.in_((JobStatus.queued, JobStatus.processing)))
        ) or 0,
    }


@router.get("/settings", response_model=ProcessingSettings, summary="Get processing settings")
def read_settings(db: Session = Depends(get_db)):
    return get_processing_settings(db)


@router.put("/settings", response_model=ProcessingSettings, summary="Update processing settings (applies to new jobs)")
def write_settings(payload: ProcessingSettings, db: Session = Depends(get_db)):
    return save_processing_settings(db, payload)


@router.get("/system", response_model=SystemInfo, summary="Environment & tooling information")
def system_info(storage: Storage = Depends(get_storage)):
    s = get_settings()
    tools = get_ffmpeg_tools()
    return SystemInfo(
        app_name=s.app_name,
        version=s.app_version,
        ffmpeg_path=tools.ffmpeg,
        ffprobe_path=tools.ffprobe,
        ffmpeg_source=tools.source,
        opencv_version=cv2.__version__,
        data_dir=str(s.data_dir.resolve()),
        database_url=s.resolved_database_url,
        allowed_extensions=s.allowed_extensions,
        worker_threads=s.worker_threads,
        storage_bytes=storage.usage(),
    )
