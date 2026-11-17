"""Background queue for vision analysis jobs.

Separate from the Phase 1 ingest queue so a long detection run never delays
metadata extraction of new uploads. Jobs are persisted in ``vision_jobs``:
queued jobs resume after a restart; jobs interrupted mid-run are marked failed
and can be retried. A video can have at most one queued/processing job.
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.database import session_scope
from app.db.models import JobStatus, Video, VideoStatus
from app.db.vision_models import (
    FighterDetection,
    FighterDistance,
    MovementMetric,
    PoseLandmark,
    TrackingSession,
    VisionJob,
)
from app.pipeline.base import JobCancelled
from app.services.storage import Storage
from app.vision.analyzer import AnalysisError, VideoInfo, analyze_video, vision_dir
from app.vision.settings import VisionSettings, get_vision_settings

log = logging.getLogger(__name__)
ACTIVE = (JobStatus.queued, JobStatus.processing)
RESULT_TABLES = (PoseLandmark, FighterDetection, MovementMetric, FighterDistance, TrackingSession)


class VisionConflict(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def delete_job_results(db: Session, job_ids: list[str]) -> None:
    """Remove result rows for the given jobs (in batches of statements, not row by row)."""
    if not job_ids:
        return
    for model in RESULT_TABLES:
        db.execute(delete(model).where(model.job_id.in_(job_ids)))


def current_job(db: Session, video_id: str) -> VisionJob | None:
    """The analysis whose results are shown: the most recent completed job."""
    return db.scalar(
        select(VisionJob)
        .where(VisionJob.video_id == video_id, VisionJob.status == JobStatus.completed)
        .order_by(VisionJob.finished_at.desc())
        .limit(1)
    )


class VisionJobManager:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Condition()
        self._cancel: set[str] = set()  # video ids
        self._active: dict[str, str] = {}  # job_id -> video_id
        self._running = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="vision-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        if not self._running:
            return
        self._running = False
        with self._lock:
            self._cancel.update(self._active.values())
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=timeout)

    def _recover(self) -> None:
        with session_scope() as db:
            for job in db.scalars(select(VisionJob).where(VisionJob.status == JobStatus.processing)).all():
                job.status = JobStatus.failed
                job.error_message = "Interrupted: the application stopped during analysis. Retry to run again."
                job.finished_at = _now()
                delete_job_results(db, [job.id])
                shutil.rmtree(vision_dir(job.video_id, job.id), ignore_errors=True)
            queued = [j.id for j in db.scalars(
                select(VisionJob).where(VisionJob.status == JobStatus.queued).order_by(VisionJob.created_at)).all()]
        for jid in queued:
            self._queue.put(jid)

    # ---------------------------------------------------------------- API
    def create_job(self, db: Session, video: Video, settings: VisionSettings | None = None) -> VisionJob:
        if video.status != VideoStatus.completed:
            raise VisionConflict("Video must finish Phase 1 processing (metadata) before vision analysis")
        active = db.scalar(select(VisionJob).where(VisionJob.video_id == video.id, VisionJob.status.in_(ACTIVE)))
        if active is not None:
            raise VisionConflict(f"An analysis for this video is already {active.status.value}")
        settings = settings or get_vision_settings(db)
        attempts = db.scalar(select(VisionJob.attempt).where(VisionJob.video_id == video.id)
                             .order_by(VisionJob.attempt.desc()).limit(1)) or 0
        job = VisionJob(video_id=video.id, status=JobStatus.queued, attempt=attempts + 1,
                        settings_snapshot=settings.model_dump(), model_name=settings.model)
        db.add(job)
        db.commit()
        db.refresh(job)
        self._queue.put(job.id)
        return job

    def cancel_video(self, video_id: str, wait_seconds: float = 30.0) -> bool:
        deadline = time.monotonic() + wait_seconds
        with self._lock:
            self._cancel.add(video_id)
            while video_id in self._active.values():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lock.wait(timeout=remaining)
        return True

    def forget_cancel(self, video_id: str) -> None:
        with self._lock:
            self._cancel.discard(video_id)

    def remove_analysis(self, db: Session, video_id: str) -> int:
        """Delete every vision job, result row and file for a video (cancelling a running job first)."""
        if not self.cancel_video(video_id):
            raise VisionConflict("Analysis is still stopping; try again in a moment")
        try:
            ids = list(db.scalars(select(VisionJob.id).where(VisionJob.video_id == video_id)).all())
            delete_job_results(db, ids)
            db.execute(delete(VisionJob).where(VisionJob.video_id == video_id))
            db.commit()
            shutil.rmtree(self.storage.settings.data_dir / "vision" / video_id, ignore_errors=True)
            return len(ids)
        finally:
            self.forget_cancel(video_id)

    # ---------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                break
            try:
                self._run(job_id)
            except Exception:  # pragma: no cover - defensive
                log.exception("Vision job %s crashed outside handlers", job_id)

    def _run(self, job_id: str) -> None:
        with session_scope() as db:
            job = db.get(VisionJob, job_id)
            if job is None or job.status != JobStatus.queued:
                return
            video = db.get(Video, job.video_id)
            if video is None:
                return
            path = self.storage.resolve(video.stored_path)
            info = VideoInfo(video.id, path, video.fps or 0.0, video.frame_count or 0, video.width or 0, video.height or 0)
            settings = VisionSettings.model_validate(job.settings_snapshot or {})
            job.status = JobStatus.processing
            job.started_at = _now()
            job.progress = 0.0
            job.current_stage = "starting"
            video_id = video.id

        with self._lock:
            if not self._running:
                return
            self._cancel.discard(video_id)
            self._active[job_id] = video_id

        last = [0.0]

        def progress(stage: str, frac: float) -> None:
            now = time.monotonic()
            if now - last[0] < 0.75 and frac < 0.99:
                return
            last[0] = now
            try:
                with session_scope() as db:
                    j = db.get(VisionJob, job_id)
                    if j is not None:
                        j.progress = round(min(frac, 0.999), 4)
                        j.current_stage = stage
            except Exception:
                log.debug("progress write failed", exc_info=True)

        def cancelled() -> bool:
            return video_id in self._cancel or not self._running

        try:
            result = analyze_video(job_id, info, settings, progress, cancelled)
        except JobCancelled:
            self._finish_failed(job_id, video_id, "Cancelled" if self._running else
                                "Interrupted by application shutdown. Retry to run again.",
                                JobStatus.cancelled if self._running else JobStatus.failed)
        except AnalysisError as exc:
            self._finish_failed(job_id, video_id, str(exc), JobStatus.failed)
        except Exception as exc:
            log.exception("Vision job %s failed", job_id)
            self._finish_failed(job_id, video_id, f"Unexpected error: {type(exc).__name__}: {exc}", JobStatus.failed)
        else:
            self._finish_ok(job_id, video_id, result)
        finally:
            with self._lock:
                self._active.pop(job_id, None)
                self._lock.notify_all()

    def _finish_failed(self, job_id: str, video_id: str, message: str, status: JobStatus) -> None:
        log.warning("Vision job %s %s: %s", job_id, status.value, message)
        shutil.rmtree(vision_dir(video_id, job_id), ignore_errors=True)
        with session_scope() as db:
            delete_job_results(db, [job_id])  # partial batches from this run
            job = db.get(VisionJob, job_id)
            if job is not None:
                job.status = status
                job.error_message = message
                job.finished_at = _now()

    def _finish_ok(self, job_id: str, video_id: str, result) -> None:
        with session_scope() as db:
            job = db.get(VisionJob, job_id)
            if job is None:  # video deleted meanwhile
                shutil.rmtree(vision_dir(video_id, job_id), ignore_errors=True)
                return
            job.status = JobStatus.completed
            job.progress = 1.0
            job.current_stage = None
            job.finished_at = _now()
            job.stats = result.stats
            job.warnings = result.warnings or None
            job.device = result.device_label
            job.annotated_video_path = self.storage.to_relative(result.annotated_path) if result.annotated_path else None
            job.export_path = self.storage.to_relative(result.export_path) if result.export_path else None
            # Supersede earlier analyses of this video: drop their rows and files, keep their job records.
            older = [j for j in db.scalars(select(VisionJob).where(
                VisionJob.video_id == video_id, VisionJob.id != job_id, VisionJob.status == JobStatus.completed)).all()]
            delete_job_results(db, [j.id for j in older])
            for j in older:
                j.stats = {**(j.stats or {}), "superseded_by": job_id}
                j.annotated_video_path = None
                j.export_path = None
                shutil.rmtree(vision_dir(video_id, j.id), ignore_errors=True)
        log.info("Vision job %s completed: %s", job_id, {k: result.stats.get(k) for k in ("frames_analysed", "tracks", "processing_seconds")})
