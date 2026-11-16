"""Phase 4 tables (fighting-pattern intelligence). New tables only — existing databases upgrade in place.

All rows here are *derived* from Phase 3 actions / annotations and can be regenerated at any time
(POST /api/patterns/{fighter}/analyze). Phase 3 results are never modified.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.models import JobStatus, UTCDateTime, new_id, utcnow


class FightInfo(Base):
    """Which fight a video belongs to (several videos may show one fight) and when it took place."""

    __tablename__ = "fight_info"
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True)
    fight_key: Mapped[str | None] = mapped_column(String(64), index=True)  # same value = same fight
    fight_date: Mapped[str | None] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    event: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class PatternVideoCache(Base):
    """Per-video extracted observations; reused while the video's inputs (signature) are unchanged."""

    __tablename__ = "pattern_video_cache"
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True)
    signature: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16))  # detections | annotations
    observations: Mapped[list] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class PatternAnalysisJob(Base):
    __tablename__ = "pattern_analysis_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)  # fighter id, track key, or "*" (all)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(80))
    config: Mapped[dict] = mapped_column(JSON)
    dataset_version: Mapped[str | None] = mapped_column(String(32))
    stats: Mapped[dict | None] = mapped_column(JSON)
    warnings: Mapped[list | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class PatternSnapshot(Base):
    """Complete stored analysis of one fighter (the version served when no filters are applied)."""

    __tablename__ = "pattern_snapshots"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    dataset_version: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class ActionTransition(Base):
    __tablename__ = "action_transitions"
    __table_args__ = (Index("ix_transitions_fighter_def", "fighter_key", "definition", "order"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fighter_key: Mapped[str] = mapped_column(String(80))
    snapshot_id: Mapped[str] = mapped_column(String(32), index=True)
    definition: Mapped[str] = mapped_column(String(16))  # next | next_offensive | window
    order: Mapped[int] = mapped_column(Integer)
    previous_action: Mapped[str] = mapped_column(String(200))  # context, '>'-joined
    next_action: Mapped[str] = mapped_column(String(40))
    count: Mapped[int] = mapped_column(Integer)
    context_count: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    fights: Mapped[int] = mapped_column(Integer)
    reliability: Mapped[str] = mapped_column(String(16))
    observation_window: Mapped[float | None] = mapped_column(Float)  # seconds (window definition only)


class FightingCombination(Base):
    __tablename__ = "fighting_combinations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)
    snapshot_id: Mapped[str] = mapped_column(String(32), index=True)
    sequence: Mapped[list] = mapped_column(JSON)
    sequence_key: Mapped[str] = mapped_column(String(300), index=True)
    length: Mapped[int] = mapped_column(Integer)
    occurrences: Mapped[int] = mapped_column(Integer)
    distinct_fights: Mapped[int] = mapped_column(Integer)
    support: Mapped[float] = mapped_column(Float)  # occurrences per fight analysed
    confidence: Mapped[float | None] = mapped_column(Float)
    avg_interval: Mapped[float | None] = mapped_column(Float)
    classification: Mapped[str] = mapped_column(String(40))


class FighterTendency(Base):
    __tablename__ = "fighter_tendencies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)
    snapshot_id: Mapped[str] = mapped_column(String(32), index=True)
    category: Mapped[str] = mapped_column(String(16))  # offensive | defensive | movement
    action: Mapped[str] = mapped_column(String(40))
    count: Mapped[int] = mapped_column(Integer)
    frequency: Mapped[float] = mapped_column(Float)  # share within category
    per_minute: Mapped[float | None] = mapped_column(Float)
    fights: Mapped[int] = mapped_column(Integer)
    period_start: Mapped[str | None] = mapped_column(String(10))
    period_end: Mapped[str | None] = mapped_column(String(10))
