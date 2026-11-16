"""Database models.

Designed so later phases can attach their own tables (detections, poses,
strikes, predictions...) to ``Video`` / ``SampledFrame`` / ``Fighter`` by
foreign key without restructuring these.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.db.database import Base


class UTCDateTime(TypeDecorator):
    """Stores UTC datetimes; always returns timezone-aware values (SQLite drops tzinfo)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class VideoStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Stance(str, enum.Enum):
    orthodox = "Orthodox"
    southpaw = "Southpaw"
    unknown = "Unknown"


video_fighters = Table(
    "video_fighters",
    Base.metadata,
    Column("video_id", String(32), ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True),
    Column("fighter_id", String(32), ForeignKey("fighters.id", ondelete="CASCADE"), primary_key=True),
)


class Fighter(Base):
    __tablename__ = "fighters"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), index=True)
    nickname: Mapped[str | None] = mapped_column(String(120))
    stance: Mapped[Stance] = mapped_column(Enum(Stance, native_enum=False, length=16), default=Stance.unknown)
    height_cm: Mapped[float | None] = mapped_column(Float)
    reach_cm: Mapped[float | None] = mapped_column(Float)
    weight_class: Mapped[str | None] = mapped_column(String(60))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)

    videos: Mapped[list[Video]] = relationship(
        secondary=video_fighters, back_populates="fighters", order_by="Video.uploaded_at.desc()"
    )


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(512))  # relative to data_dir
    file_ext: Mapped[str] = mapped_column(String(8))
    file_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    content_type: Mapped[str | None] = mapped_column(String(100))
    notes: Mapped[str | None] = mapped_column(Text)

    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus, native_enum=False, length=16), default=VideoStatus.uploaded, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    # Extracted metadata (populated by processing pipeline)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    frame_count_source: Mapped[str | None] = mapped_column(String(32))
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    rotation: Mapped[int | None] = mapped_column(Integer)
    video_codec: Mapped[str | None] = mapped_column(String(64))
    pixel_format: Mapped[str | None] = mapped_column(String(32))
    container_format: Mapped[str | None] = mapped_column(String(100))
    bitrate_kbps: Mapped[float | None] = mapped_column(Float)
    has_audio: Mapped[bool | None] = mapped_column(Boolean)
    audio_codec: Mapped[str | None] = mapped_column(String(64))
    audio_sample_rate: Mapped[int | None] = mapped_column(Integer)
    audio_channels: Mapped[int | None] = mapped_column(Integer)
    metadata_source: Mapped[str | None] = mapped_column(String(64))
    raw_metadata: Mapped[dict | None] = mapped_column(JSON)
    quality: Mapped[dict | None] = mapped_column(JSON)

    thumbnail_path: Mapped[str | None] = mapped_column(String(512))
    preview_path: Mapped[str | None] = mapped_column(String(512))
    sampled_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    sampling_config: Mapped[dict | None] = mapped_column(JSON)

    # Fighter assignment (manual in Phase 1)
    target_fighter_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("fighters.id", ondelete="SET NULL"), index=True
    )
    target_fighter_notes: Mapped[str | None] = mapped_column(Text)  # e.g. "red corner, black shorts"

    target_fighter: Mapped[Fighter | None] = relationship(foreign_keys=[target_fighter_id])
    fighters: Mapped[list[Fighter]] = relationship(secondary=video_fighters, back_populates="videos")
    jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="video", cascade="all, delete-orphan", order_by="ProcessingJob.created_at.desc()"
    )
    frames: Mapped[list[SampledFrame]] = relationship(
        back_populates="video", cascade="all, delete-orphan", order_by="SampledFrame.frame_index"
    )


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    job_type: Mapped[str] = mapped_column(String(40), default="phase1_ingest")
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(Text)
    stage_log: Mapped[list | None] = mapped_column(JSON)
    warnings: Mapped[list | None] = mapped_column(JSON)
    settings_snapshot: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    video: Mapped[Video] = relationship(back_populates="jobs")


class SampledFrame(Base):
    __tablename__ = "sampled_frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    image_path: Mapped[str | None] = mapped_column(String(512))  # relative; None if images not stored
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    brightness: Mapped[float | None] = mapped_column(Float)
    contrast: Mapped[float | None] = mapped_column(Float)
    sharpness: Mapped[float | None] = mapped_column(Float)

    video: Mapped[Video] = relationship(back_populates="frames")


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
