"""Phase 3 tables (action recognition). New tables only — existing databases upgrade in place."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.models import JobStatus, UTCDateTime, new_id, utcnow


class VideoAnnotationMeta(Base):
    """Per-video dataset settings: is it fully annotated (so unlabelled windows are true negatives),
    and which split it belongs to. Splits are by video to avoid leakage."""

    __tablename__ = "video_annotation_meta"

    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True)
    fully_annotated: Mapped[bool] = mapped_column(Boolean, default=False)
    split: Mapped[str] = mapped_column(String(8), default="auto")  # auto | train | test | exclude
    split_group: Mapped[str | None] = mapped_column(String(64))  # e.g. fight id; defaults to video id
    notes: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class ActionAnnotation(Base):
    __tablename__ = "action_annotations"
    __table_args__ = (Index("ix_ann_video_time", "video_id", "start_time"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    fighter_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("fighters.id", ondelete="SET NULL"), index=True)
    track_id: Mapped[int | None] = mapped_column(Integer)  # Phase 2 track at annotation time (optional)
    category: Mapped[str] = mapped_column(String(40), index=True)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)  # annotator certainty 0-1
    opponent_reaction: Mapped[str | None] = mapped_column(String(40))  # optional category key or free text
    status: Mapped[str] = mapped_column(String(16), default="verified")  # draft | verified | needs_review | rejected
    source: Mapped[str] = mapped_column(String(16), default="manual")  # manual | import | correction
    corrects_action_id: Mapped[str | None] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class ActionModelVersion(Base):
    __tablename__ = "action_models"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(24))  # random_forest | xgboost | svm | lstm | tcn
    status: Mapped[str] = mapped_column(String(16), default="ready")  # ready | failed | archived
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    training_job_id: Mapped[str | None] = mapped_column(String(32), index=True)
    dataset_version: Mapped[str] = mapped_column(String(32))
    classes: Mapped[list] = mapped_column(JSON)  # categories the model can output (+ background)
    config: Mapped[dict] = mapped_column(JSON)  # window / feature / hyper-parameters
    metrics: Mapped[dict | None] = mapped_column(JSON)  # accuracy, precision, recall, f1, per-class, confusion
    training_info: Mapped[dict | None] = mapped_column(JSON)
    path: Mapped[str] = mapped_column(String(512))  # relative to data dir
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class ActionTrainingJob(Base):
    __tablename__ = "action_training_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(80))
    config: Mapped[dict] = mapped_column(JSON)
    dataset_version: Mapped[str | None] = mapped_column(String(32))
    dataset_summary: Mapped[dict | None] = mapped_column(JSON)
    results: Mapped[dict | None] = mapped_column(JSON)  # comparison table of algorithms
    warnings: Mapped[list | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class ActionAnalysisJob(Base):
    __tablename__ = "action_analysis_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    vision_job_id: Mapped[str | None] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(80))
    settings: Mapped[dict | None] = mapped_column(JSON)
    stats: Mapped[dict | None] = mapped_column(JSON)
    warnings: Mapped[list | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class RecognizedAction(Base):
    __tablename__ = "recognized_actions"
    __table_args__ = (Index("ix_act_job_track_time", "job_id", "track_id", "start_time"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("action_analysis_jobs.id", ondelete="CASCADE"))
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    fighter_id: Mapped[str | None] = mapped_column(String(32))  # from track assignment at analysis time
    category: Mapped[str] = mapped_column(String(40), index=True)  # model output (or 'unknown')
    action_group: Mapped[str] = mapped_column(String(16))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)  # raw model probability of the predicted class
    runner_up: Mapped[str | None] = mapped_column(String(40))
    runner_up_confidence: Mapped[float | None] = mapped_column(Float)
    validated_precision: Mapped[float | None] = mapped_column(Float)  # measured on held-out data for this class
    validated_support: Mapped[int | None] = mapped_column(Integer)
    limb: Mapped[str | None] = mapped_column(String(16))  # most active limb in the window
    windows: Mapped[int] = mapped_column(Integer, default=1)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    model_version: Mapped[str] = mapped_column(String(64))
    review_status: Mapped[str] = mapped_column(String(16), default="unreviewed")  # unreviewed | confirmed | corrected | rejected
    corrected_category: Mapped[str | None] = mapped_column(String(40))


class ActionSequence(Base):
    __tablename__ = "action_sequences"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("action_analysis_jobs.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    fighter_key: Mapped[str] = mapped_column(String(48))  # fighter id, or "track:N" when unassigned
    fighter_id: Mapped[str | None] = mapped_column(String(32))
    track_ids: Mapped[list] = mapped_column(JSON)
    sequence_index: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    actions: Mapped[list] = mapped_column(JSON)  # [{action_id, category, label, start, end, confidence}]
    intervals: Mapped[list] = mapped_column(JSON)  # seconds between consecutive actions (start_{i+1} - end_i)
    n_actions: Mapped[int] = mapped_column(Integer)
