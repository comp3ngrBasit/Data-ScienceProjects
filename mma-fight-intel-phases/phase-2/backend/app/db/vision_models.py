"""Phase 2 tables (computer vision).

Only *new* tables are added, so an existing Phase 1 database is upgraded in
place by ``create_all`` without touching existing tables. Every result row is
tagged with the vision job that produced it; the "current" analysis of a video
is its most recent completed job. Re-analysis writes a new job's rows and only
removes the older ones once the new run succeeds.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Enum, Float, ForeignKey, Index, Integer, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.db.models import Fighter, JobStatus, UTCDateTime, Video, new_id, utcnow


class VisionJob(Base):
    __tablename__ = "vision_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list | None] = mapped_column(JSON)
    settings_snapshot: Mapped[dict | None] = mapped_column(JSON)
    model_name: Mapped[str | None] = mapped_column(String(40))
    device: Mapped[str | None] = mapped_column(String(40))
    stats: Mapped[dict | None] = mapped_column(JSON)
    annotated_video_path: Mapped[str | None] = mapped_column(String(512))
    export_path: Mapped[str | None] = mapped_column(String(512))  # npz bundle for Phase 3
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    video: Mapped[Video] = relationship()


class FighterDetection(Base):
    __tablename__ = "fighter_detections"
    __table_args__ = (
        Index("ix_det_job_frame", "job_id", "frame_number"),
        Index("ix_det_job_track", "job_id", "track_id", "frame_number"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("vision_jobs.id", ondelete="CASCADE"))
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    frame_number: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    track_id: Mapped[int] = mapped_column(Integer)
    # Bounding box, normalised to [0, 1] by frame width/height
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)


class PoseLandmark(Base):
    __tablename__ = "pose_landmarks"
    __table_args__ = (Index("ix_pose_job_track_frame", "job_id", "track_id", "frame_number"),
                      Index("ix_pose_job_frame", "job_id", "frame_number"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("vision_jobs.id", ondelete="CASCADE"))
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    frame_number: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    landmark_index: Mapped[int] = mapped_column(SmallInteger)
    landmark: Mapped[str] = mapped_column(String(16))
    x: Mapped[float] = mapped_column(Float)  # normalised 0-1 (frame width)
    y: Mapped[float] = mapped_column(Float)  # normalised 0-1 (frame height)
    confidence: Mapped[float] = mapped_column(Float)


class TrackingSession(Base):
    __tablename__ = "tracking_sessions"
    __table_args__ = (Index("ix_track_job_track", "job_id", "track_id", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("vision_jobs.id", ondelete="CASCADE"))
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    assigned_fighter_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("fighters.id", ondelete="SET NULL"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), default="unreviewed")  # fighter | referee | other | unreviewed
    suggested_role: Mapped[str] = mapped_column(String(16), default="other")  # heuristic only
    fighter_score: Mapped[float] = mapped_column(Float, default=0.0)
    first_frame: Mapped[int] = mapped_column(Integer)
    last_frame: Mapped[int] = mapped_column(Integer)
    first_timestamp: Mapped[float] = mapped_column(Float)
    last_timestamp: Mapped[float] = mapped_column(Float)
    frames_detected: Mapped[int] = mapped_column(Integer)
    coverage: Mapped[float] = mapped_column(Float)  # detected / analysed frames within [first, last]
    video_coverage: Mapped[float] = mapped_column(Float)  # detected / analysed frames in whole video
    interruptions: Mapped[int] = mapped_column(Integer, default=0)
    segments: Mapped[list | None] = mapped_column(JSON)  # [[start_frame, end_frame], ...]
    avg_confidence: Mapped[float] = mapped_column(Float)
    avg_box_height: Mapped[float] = mapped_column(Float)  # fraction of frame height
    avg_pose_confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))  # continuous | interrupted | fragment
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)

    assigned_fighter: Mapped[Fighter | None] = relationship()


class MovementMetric(Base):
    """Per-track, per-analysed-frame movement features (smoothed). Speeds are in torso lengths / second."""

    __tablename__ = "movement_metrics"
    __table_args__ = (Index("ix_mov_job_track_frame", "job_id", "track_id", "frame_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("vision_jobs.id", ondelete="CASCADE"))
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    frame_number: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    state: Mapped[str] = mapped_column(String(16))  # moving | stationary | low_confidence | missing
    valid_landmarks: Mapped[int] = mapped_column(SmallInteger)
    center_x: Mapped[float | None] = mapped_column(Float)
    center_y: Mapped[float | None] = mapped_column(Float)
    torso_px: Mapped[float | None] = mapped_column(Float)
    center_speed: Mapped[float | None] = mapped_column(Float)
    direction_deg: Mapped[float | None] = mapped_column(Float)  # 0 = screen right, 90 = screen up
    left_wrist_speed: Mapped[float | None] = mapped_column(Float)
    right_wrist_speed: Mapped[float | None] = mapped_column(Float)
    left_elbow_speed: Mapped[float | None] = mapped_column(Float)
    right_elbow_speed: Mapped[float | None] = mapped_column(Float)
    left_shoulder_speed: Mapped[float | None] = mapped_column(Float)
    right_shoulder_speed: Mapped[float | None] = mapped_column(Float)
    left_hip_speed: Mapped[float | None] = mapped_column(Float)
    right_hip_speed: Mapped[float | None] = mapped_column(Float)
    left_knee_speed: Mapped[float | None] = mapped_column(Float)
    right_knee_speed: Mapped[float | None] = mapped_column(Float)
    left_ankle_speed: Mapped[float | None] = mapped_column(Float)
    right_ankle_speed: Mapped[float | None] = mapped_column(Float)
    left_elbow_angle: Mapped[float | None] = mapped_column(Float)
    right_elbow_angle: Mapped[float | None] = mapped_column(Float)
    left_knee_angle: Mapped[float | None] = mapped_column(Float)
    right_knee_angle: Mapped[float | None] = mapped_column(Float)
    torso_lean_deg: Mapped[float | None] = mapped_column(Float)
    stance_width: Mapped[float | None] = mapped_column(Float)  # ankle distance / torso length
    guard_height: Mapped[float | None] = mapped_column(Float)  # (shoulder_y - wrist_y) / torso, mean of both hands
    posture_change: Mapped[float | None] = mapped_column(Float)  # change of normalised pose vs previous frame


class FighterDistance(Base):
    __tablename__ = "fighter_distances"
    __table_args__ = (Index("ix_dist_job_pair_frame", "job_id", "track_a", "track_b", "frame_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("vision_jobs.id", ondelete="CASCADE"))
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    frame_number: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    track_a: Mapped[int] = mapped_column(Integer)
    track_b: Mapped[int] = mapped_column(Integer)
    distance_frame: Mapped[float] = mapped_column(Float)  # centre distance as fraction of frame height
    distance_torso: Mapped[float | None] = mapped_column(Float)  # in mean torso lengths
    closing_speed: Mapped[float | None] = mapped_column(Float)  # torso lengths / s (positive = approaching)
