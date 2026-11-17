"""Phase 5 tables (next-move prediction). New tables only — existing databases upgrade in place."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.models import JobStatus, UTCDateTime, new_id, utcnow


class PredictionModel(Base):
    """Model registry: one row per trained architecture per training run."""

    __tablename__ = "prediction_models"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    version: Mapped[int] = mapped_column(Integer, index=True)  # training run number
    architecture: Mapped[str] = mapped_column(String(16))  # markov | xgboost | gru | lstm | transformer
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)  # chosen by the evaluation rule in its run
    experimental: Mapped[bool] = mapped_column(Boolean, default=True)
    dataset_version: Mapped[str] = mapped_column(String(32))
    dataset_info: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)  # cross-validated metrics of this architecture
    params: Mapped[dict | None] = mapped_column(JSON)
    temperature: Mapped[float] = mapped_column(Float, default=1.0)
    path: Mapped[str] = mapped_column(String(500))
    training_job_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class PredictionTrainingJob(Base):
    __tablename__ = "prediction_training_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_stage: Mapped[str | None] = mapped_column(String(120))
    config: Mapped[dict] = mapped_column(JSON)
    dataset_info: Mapped[dict | None] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON)  # full evaluation report
    version: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class PredictionVideoJob(Base):
    __tablename__ = "prediction_video_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    fighter_key: Mapped[str | None] = mapped_column(String(80))
    model_id: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, native_enum=False, length=16), default=JobStatus.queued, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stats: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class PredictionResult(Base):
    """A prediction exactly as it was made. Never edited after creation."""

    __tablename__ = "prediction_results"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)
    video_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    video_job_id: Mapped[str | None] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(String(16))  # manual | video
    at_time: Mapped[float | None] = mapped_column(Float)  # video time of the prediction moment
    source_action_id: Mapped[str | None] = mapped_column(String(32))
    sequence: Mapped[list] = mapped_column(JSON)  # current action sequence (oldest -> newest)
    context: Mapped[dict | None] = mapped_column(JSON)
    distribution: Mapped[dict] = mapped_column(JSON)  # full predicted distribution
    top3: Mapped[list] = mapped_column(JSON)
    predicted_action: Mapped[str] = mapped_column(String(40))
    predicted_probability: Mapped[float] = mapped_column(Float)
    reliability: Mapped[dict] = mapped_column(JSON)
    model_id: Mapped[str] = mapped_column(String(32), index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    in_training_data: Mapped[bool] = mapped_column(Boolean, default=False)  # the model saw this fight while training
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class PredictionEvaluation(Base):
    """Prediction history: the actual next action compared with a stored prediction."""

    __tablename__ = "prediction_evaluations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    prediction_id: Mapped[str] = mapped_column(String(32), ForeignKey("prediction_results.id", ondelete="CASCADE"), unique=True)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(20))  # evaluated | no_next_action
    actual_action: Mapped[str | None] = mapped_column(String(40))
    actual_source: Mapped[str] = mapped_column(String(20))  # video | user
    predicted_action: Mapped[str] = mapped_column(String(40))
    top3: Mapped[list] = mapped_column(JSON)
    correct_top1: Mapped[bool | None] = mapped_column(Boolean)
    correct_top3: Mapped[bool | None] = mapped_column(Boolean)
    probability_of_actual: Mapped[float | None] = mapped_column(Float)
    in_training_data: Mapped[bool] = mapped_column(Boolean, default=False)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)


class ExternalSequence(Base):
    """Imported real action sequences without video in this app (e.g. BoxingVI annotation sheets)."""

    __tablename__ = "prediction_external_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(40), index=True)
    fighter_key: Mapped[str] = mapped_column(String(80), index=True)
    fight_key: Mapped[str] = mapped_column(String(80), index=True)
    category: Mapped[str] = mapped_column(String(40))
    original_label: Mapped[str | None] = mapped_column(String(80))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    stance: Mapped[str | None] = mapped_column(String(16))
