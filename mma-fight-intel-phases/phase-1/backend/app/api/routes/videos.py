from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db, get_job_manager, get_storage
from app.core.config import get_settings
from app.db.models import Fighter, SampledFrame, Video, VideoStatus
from app.schemas.video import (
    FrameList,
    FrameOut,
    JobOut,
    UploadItemResult,
    UploadResponse,
    VideoDetail,
    VideoList,
    VideoStatusOut,
    VideoUpdate,
)
from app.services import video_service as vs
from app.services.job_manager import JobConflictError, JobManager
from app.services.settings_service import get_processing_settings
from app.services.storage import Storage, UnsafePathError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/videos", tags=["videos"])

SORT_COLUMNS = {
    "uploaded_at": Video.uploaded_at,
    "title": func.lower(Video.title),
    "duration": Video.duration_seconds,
    "file_size": Video.file_size,
    "status": Video.status,
}


def _load_video(db: Session, video_id: str) -> Video:
    video = db.scalar(
        select(Video)
        .where(Video.id == video_id)
        .options(selectinload(Video.fighters), selectinload(Video.target_fighter), selectinload(Video.jobs))
    )
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    return video


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload one or more videos (MP4, MOV, AVI)",
    responses={409: {"description": "Duplicate"}, 413: {"description": "Too large"}, 415: {"description": "Unsupported"}},
)
async def upload_videos(
    files: list[UploadFile] = File(..., description="One or more video files"),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    jobs: JobManager = Depends(get_job_manager),
):
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files provided")
    runtime = get_processing_settings(db)
    max_bytes = min(runtime.max_upload_size_mb, get_settings().hard_max_upload_size_mb) * 1024 * 1024

    results: list[UploadItemResult] = []
    for f in files:
        name = vs.clean_display_name(f.filename)
        try:
            stored = await vs.save_upload(db, storage, f, max_bytes, runtime.auto_process_on_upload, jobs)
            results.append(
                UploadItemResult(filename=name, status="created", http_status=201, video=vs.to_summary(stored.video))
            )
        except vs.UploadRejected as exc:
            results.append(
                UploadItemResult(
                    filename=name,
                    status="duplicate" if exc.http_status == 409 else "rejected",
                    http_status=exc.http_status,
                    error=exc.message,
                    existing_video_id=exc.existing_video_id,
                )
            )
    body = UploadResponse(
        results=results,
        created=sum(r.status == "created" for r in results),
        duplicates=sum(r.status == "duplicate" for r in results),
        rejected=sum(r.status == "rejected" for r in results),
    )
    if body.created == 0:
        # Nothing stored: surface the (first) error's status code.
        return JSONResponse(status_code=results[0].http_status, content=body.model_dump(mode="json"))
    return body


@router.get("", response_model=VideoList, summary="List videos with search, filters and sorting")
def list_videos(
    search: str | None = Query(None, max_length=200, description="Matches title or original filename"),
    status_filter: VideoStatus | None = Query(None, alias="status"),
    fighter_id: str | None = Query(None, description="Videos assigned to this fighter"),
    unassigned: bool = Query(False, description="Only videos with no fighters assigned"),
    sort: Literal["uploaded_at", "title", "duration", "file_size", "status"] = "uploaded_at",
    order: Literal["asc", "desc"] = "desc",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = select(Video)
    if search:
        like = f"%{search.strip()}%"
        q = q.where(or_(Video.title.ilike(like), Video.original_filename.ilike(like)))
    if status_filter:
        q = q.where(Video.status == status_filter)
    if fighter_id:
        q = q.where(or_(Video.fighters.any(Fighter.id == fighter_id), Video.target_fighter_id == fighter_id))
    if unassigned:
        q = q.where(~Video.fighters.any(), Video.target_fighter_id.is_(None))
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    col = SORT_COLUMNS[sort]
    q = q.order_by(col.asc().nulls_last() if order == "asc" else col.desc().nulls_last(), Video.id)
    q = q.options(selectinload(Video.fighters), selectinload(Video.target_fighter)).limit(limit).offset(offset)
    items = [vs.to_summary(v) for v in db.scalars(q).all()]
    return VideoList(items=items, total=total, limit=limit, offset=offset)


@router.get("/{video_id}", response_model=VideoDetail, summary="Get full video details")
def get_video(video_id: str, db: Session = Depends(get_db)):
    return vs.to_detail(_load_video(db, video_id))


@router.patch("/{video_id}", response_model=VideoDetail, summary="Rename / annotate / assign fighters")
def update_video(video_id: str, payload: VideoUpdate, db: Session = Depends(get_db)):
    video = _load_video(db, video_id)
    data = payload.model_dump(exclude_unset=True)

    if "title" in data and data["title"] is not None:
        video.title = data["title"]
    if "notes" in data:
        video.notes = (data["notes"] or "").strip() or None
    if "target_fighter_notes" in data:
        video.target_fighter_notes = (data["target_fighter_notes"] or "").strip() or None

    if "fighter_ids" in data and data["fighter_ids"] is not None:
        ids = list(dict.fromkeys(data["fighter_ids"]))
        fighters = db.scalars(select(Fighter).where(Fighter.id.in_(ids))).all() if ids else []
        missing = set(ids) - {f.id for f in fighters}
        if missing:
            raise HTTPException(422, f"Unknown fighter id(s): {', '.join(sorted(missing))}")
        video.fighters = list(fighters)
        if video.target_fighter_id and video.target_fighter_id not in ids and "target_fighter_id" not in data:
            video.target_fighter_id = None

    if "target_fighter_id" in data:
        tid = data["target_fighter_id"]
        if tid:
            fighter = db.get(Fighter, tid)
            if fighter is None:
                raise HTTPException(422, "Unknown target fighter id")
            video.target_fighter_id = tid
            if fighter not in video.fighters:
                video.fighters.append(fighter)  # the analysed fighter is always assigned
        else:
            video.target_fighter_id = None

    db.commit()
    db.expire_all()
    return vs.to_detail(_load_video(db, video_id))


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a video and all derived data")
def delete_video(
    video_id: str,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    jobs: JobManager = Depends(get_job_manager),
):
    video = _load_video(db, video_id)
    stored = video.stored_path
    if not jobs.cancel_video(video_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Video is busy processing and could not be stopped; try again shortly")
    db.delete(video)
    db.commit()
    storage.delete_video_artifacts(video_id, stored)
    jobs.forget_cancel(video_id)
    log.info("Deleted video %s", video_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{video_id}/process",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue (re)processing for a video — also used to retry failed videos",
)
def process_video(video_id: str, db: Session = Depends(get_db), jobs: JobManager = Depends(get_job_manager)):
    video = _load_video(db, video_id)
    try:
        job = jobs.create_job(db, video)
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return vs.job_to_out(job)


@router.get("/{video_id}/status", response_model=VideoStatusOut, summary="Processing status")
def video_status(video_id: str, db: Session = Depends(get_db)):
    video = _load_video(db, video_id)
    latest = video.jobs[0] if video.jobs else None
    return VideoStatusOut(
        video_id=video.id,
        status=video.status,
        error_message=video.error_message,
        processed_at=video.processed_at,
        latest_job=vs.job_to_out(latest) if latest else None,
    )


@router.get("/{video_id}/jobs", response_model=list[JobOut], summary="Processing history for a video")
def video_jobs(video_id: str, db: Session = Depends(get_db)):
    video = _load_video(db, video_id)
    return [vs.job_to_out(j) for j in video.jobs]


def _file_or_404(storage: Storage, relative: str | None, what: str):
    try:
        path = storage.resolve(relative)
    except UnsafePathError:
        path = None
    if path is None or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{what} not available")
    return path


@router.get("/{video_id}/stream", summary="Stream video (supports HTTP Range). Uses preview proxy when present.")
def stream_video(
    video_id: str,
    source: Literal["auto", "original", "preview"] = "auto",
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    if source == "preview" or (source == "auto" and video.preview_path):
        path = _file_or_404(storage, video.preview_path, "Preview")
        media_type = "video/mp4"
    else:
        path = _file_or_404(storage, video.stored_path, "Video file")
        media_type = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}.get(video.file_ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{video_id}/download", summary="Download the original file")
def download_video(video_id: str, db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    path = _file_or_404(storage, video.stored_path, "Video file")
    return FileResponse(path, filename=video.original_filename, media_type="application/octet-stream")


@router.get("/{video_id}/thumbnail", summary="Video thumbnail (JPEG)")
def get_thumbnail(video_id: str, db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    path = _file_or_404(storage, video.thumbnail_path, "Thumbnail")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/{video_id}/frames", response_model=FrameList, summary="Sampled frames with timestamps & stats")
def list_frames(
    video_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    if db.get(Video, video_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    total = db.scalar(select(func.count()).where(SampledFrame.video_id == video_id)) or 0
    rows = db.scalars(
        select(SampledFrame).where(SampledFrame.video_id == video_id).order_by(SampledFrame.frame_index).limit(limit).offset(offset)
    ).all()
    items = []
    for r in rows:
        f = FrameOut.model_validate(r, from_attributes=True)
        f.image_url = f"/api/videos/{video_id}/frames/{r.id}/image" if r.image_path else None
        items.append(f)
    return FrameList(items=items, total=total, limit=limit, offset=offset)


@router.get("/{video_id}/frames/{frame_id}/image", summary="Sampled frame image (JPEG)")
def frame_image(video_id: str, frame_id: int, db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    frame = db.get(SampledFrame, frame_id)
    if frame is None or frame.video_id != video_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Frame not found")
    path = _file_or_404(storage, frame.image_path, "Frame image")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

