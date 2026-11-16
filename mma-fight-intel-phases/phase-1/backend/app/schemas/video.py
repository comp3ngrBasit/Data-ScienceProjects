from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.models import JobStatus, Stance, VideoStatus


class FighterRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    nickname: str | None = None
    stance: Stance


class VideoSummary(BaseModel):
    """Lightweight representation used in the library list."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    original_filename: str
    file_ext: str
    file_size: int
    status: VideoStatus
    error_message: str | None
    uploaded_at: datetime
    processed_at: datetime | None
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    video_codec: str | None
    has_audio: bool | None
    sampled_frame_count: int
    thumbnail_url: str | None = None
    target_fighter: FighterRef | None = None
    fighters: list[FighterRef] = []


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    video_id: str
    video_title: str | None = None
    job_type: str
    status: JobStatus
    attempt: int
    progress: float
    current_stage: str | None
    error_message: str | None
    stage_log: list[dict[str, Any]] | None
    warnings: list[str] | None
    settings_snapshot: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None = None


class VideoDetail(VideoSummary):
    content_type: str | None
    sha256: str
    notes: str | None
    updated_at: datetime
    frame_count: int | None
    frame_count_source: str | None
    rotation: int | None
    pixel_format: str | None
    container_format: str | None
    bitrate_kbps: float | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    metadata_source: str | None
    raw_metadata: dict[str, Any] | None
    quality: dict[str, Any] | None
    sampling_config: dict[str, Any] | None
    target_fighter_notes: str | None
    stream_url: str
    has_preview_proxy: bool
    jobs: list[JobOut] = []


class VideoList(BaseModel):
    items: list[VideoSummary]
    total: int
    limit: int
    offset: int


class VideoUpdate(BaseModel):
    """PATCH payload. Only provided fields are changed."""

    title: str | None = Field(None, min_length=1, max_length=255)
    notes: str | None = Field(None, max_length=5000)
    target_fighter_id: str | None = None
    target_fighter_notes: str | None = Field(None, max_length=500)
    fighter_ids: list[str] | None = Field(None, max_length=10)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("Title cannot be blank")
        return v


UploadItemStatus = Literal["created", "duplicate", "rejected"]


class UploadItemResult(BaseModel):
    filename: str
    status: UploadItemStatus
    http_status: int
    video: VideoSummary | None = None
    existing_video_id: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadItemResult]
    created: int
    duplicates: int
    rejected: int


class VideoStatusOut(BaseModel):
    video_id: str
    status: VideoStatus
    error_message: str | None
    processed_at: datetime | None
    latest_job: JobOut | None


class FrameOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    frame_index: int
    timestamp_seconds: float
    width: int | None
    height: int | None
    brightness: float | None
    contrast: float | None
    sharpness: float | None
    image_url: str | None = None


class FrameList(BaseModel):
    items: list[FrameOut]
    total: int
    limit: int
    offset: int
