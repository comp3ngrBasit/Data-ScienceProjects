"""Upload handling and response serialisation for videos."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import PurePath

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import JobStatus, ProcessingJob, Video, VideoStatus, new_id
from app.schemas.video import FighterRef, JobOut, VideoDetail, VideoSummary
from app.services.storage import Storage

log = logging.getLogger(__name__)

ISO_BMFF_BOXES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
EXT_FAMILY = {".mp4": "isobmff", ".mov": "isobmff", ".avi": "riff-avi"}


class UploadRejected(Exception):
    def __init__(self, message: str, http_status: int = 400, existing_video_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.existing_video_id = existing_video_id


def sniff_container(head: bytes) -> str | None:
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] in (b"AVI ", b"AVIX"):
        return "riff-avi"
    if len(head) >= 8 and head[4:8] in ISO_BMFF_BOXES:
        return "isobmff"
    return None


def clean_display_name(filename: str | None) -> str:
    name = filename or "video"
    name = name.replace("\\", "/")
    name = PurePath(name).name  # drop any client-side path components
    name = unicodedata.normalize("NFC", name)
    name = "".join(ch for ch in name if ch.isprintable())
    name = name.strip().strip(".") or "video"
    return name[:255]


def default_title(display_name: str) -> str:
    stem = PurePath(display_name).stem or display_name
    stem = re.sub(r"[_]+", " ", stem).strip()
    return (stem or "Untitled video")[:255]


@dataclass
class StoredUpload:
    video: Video
    job: ProcessingJob | None


async def save_upload(
    db: Session,
    storage: Storage,
    upload: UploadFile,
    max_size_bytes: int,
    auto_process: bool,
    job_manager,
) -> StoredUpload:
    settings = get_settings()
    display_name = clean_display_name(upload.filename)
    ext = PurePath(display_name).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise UploadRejected(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(settings.allowed_extensions)}", 415
        )

    tmp = storage.temp_upload_path(uuid.uuid4().hex)
    hasher = hashlib.sha256()
    size = 0
    head = b""
    try:
        with tmp.open("wb") as out:
            while True:
                chunk = await upload.read(settings.upload_chunk_size)
                if not chunk:
                    break
                if len(head) < 64:
                    head += chunk[: 64 - len(head)]
                size += len(chunk)
                if size > max_size_bytes:
                    raise UploadRejected(
                        f"File exceeds the maximum upload size of {max_size_bytes // (1024 * 1024)} MB", 413
                    )
                hasher.update(chunk)
                out.write(chunk)

        if size == 0:
            raise UploadRejected("File is empty", 400)
        family = sniff_container(head)
        if family is None:
            raise UploadRejected("File content is not a recognised MP4/MOV/AVI video container", 415)
        if family != EXT_FAMILY[ext]:
            raise UploadRejected(
                f"File extension '{ext}' does not match its content ({'AVI' if family == 'riff-avi' else 'MP4/MOV'})",
                415,
            )

        digest = hasher.hexdigest()
        existing = db.scalar(select(Video).where(Video.sha256 == digest))
        if existing is not None:
            raise UploadRejected(
                f"Duplicate: this file was already uploaded as '{existing.title}'", 409, existing.id
            )

        video_id = new_id()
        final = storage.video_path(video_id, ext)
        tmp.replace(final)
        video = Video(
            id=video_id,
            title=default_title(display_name),
            original_filename=display_name,
            stored_path=storage.to_relative(final),
            file_ext=ext,
            file_size=size,
            sha256=digest,
            content_type=upload.content_type,
            status=VideoStatus.uploaded,
        )
        db.add(video)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            storage.remove_file(final)
            dup = db.scalar(select(Video).where(Video.sha256 == digest))
            raise UploadRejected("Duplicate upload", 409, dup.id if dup else None) from exc
        db.refresh(video)
        log.info("Stored upload %s (%s, %d bytes)", video.id, display_name, size)

        job = None
        if auto_process:
            job = job_manager.create_job(db, video)
            db.refresh(video)
        return StoredUpload(video=video, job=job)
    finally:
        storage.remove_file(tmp, retries=2)
        await upload.close()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _version_token(video: Video) -> str:
    ts = video.processed_at or video.updated_at or video.uploaded_at
    return str(int(ts.timestamp())) if ts else "0"


def thumbnail_url(video: Video) -> str | None:
    if not video.thumbnail_path:
        return None
    return f"/api/videos/{video.id}/thumbnail?v={_version_token(video)}"


def to_summary(video: Video) -> VideoSummary:
    data = VideoSummary.model_validate(video, from_attributes=True)
    data.thumbnail_url = thumbnail_url(video)
    data.target_fighter = FighterRef.model_validate(video.target_fighter) if video.target_fighter else None
    data.fighters = [FighterRef.model_validate(f) for f in sorted(video.fighters, key=lambda f: f.name.lower())]
    return data


def job_to_out(job: ProcessingJob, include_title: bool = True) -> JobOut:
    out = JobOut.model_validate(job, from_attributes=True)
    if include_title and job.video is not None:
        out.video_title = job.video.title
    if job.started_at and job.finished_at:
        out.duration_seconds = round((job.finished_at - job.started_at).total_seconds(), 2)
    return out


def to_detail(video: Video) -> VideoDetail:
    summary = to_summary(video)
    detail = VideoDetail.model_validate(
        {
            **summary.model_dump(),
            **{
                k: getattr(video, k)
                for k in (
                    "content_type", "sha256", "notes", "updated_at", "frame_count", "frame_count_source",
                    "rotation", "pixel_format", "container_format", "bitrate_kbps", "audio_codec",
                    "audio_sample_rate", "audio_channels", "metadata_source", "raw_metadata", "quality",
                    "sampling_config", "target_fighter_notes",
                )
            },
            "stream_url": f"/api/videos/{video.id}/stream?v={_version_token(video)}",
            "has_preview_proxy": bool(video.preview_path),
            "jobs": [job_to_out(j, include_title=False).model_dump() for j in video.jobs],
        }
    )
    return detail


def active_job(video: Video) -> ProcessingJob | None:
    return next((j for j in video.jobs if j.status in (JobStatus.queued, JobStatus.processing)), None)
