"""Phase 6 tables: unified pipeline runs and the model deployment history. New tables only."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.models import JobStatus, UTCDateTime, new_id, utcnow


class PipelineRun(Base):
    """Video Upload -> Detection -> Pose -> Actions -> Patterns -> Prediction, each stage on the previous one's output."""

    __tablename__ = "pipeline_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(40))
    stages: Mapped[list] = mapped_column(JSON)  # [{name, status, job_id, message, started_at, finished_at}]
    options: Mapped[dict] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class ModelDeployment(Base):
    """Every change of the active model (automatic after validation, manual, rollback)."""

    __tablename__ = "model_deployments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # action | prediction
    model_id: Mapped[str] = mapped_column(String(64))
    previous_model_id: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(20))  # training | manual | rollback
    forced: Mapped[bool] = mapped_column(Boolean, default=False)
    validation: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
