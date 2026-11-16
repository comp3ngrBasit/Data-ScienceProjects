"""Background processing: an in-process, thread-based job queue.

Jobs are persisted in SQLite, so the queue survives restarts: on startup,
queued jobs are re-enqueued and jobs that were mid-processing when the app
stopped are marked failed (and can be retried from the UI/API).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.database import session_scope
from app.db.models import JobStatus, ProcessingJob, SampledFrame, Video, VideoStatus
from app.pipeline.base import JobCancelled, ProcessingContext, StageError
from app.pipeline.runner import JOB_TYPES, PipelineRunner
from app.schemas.settings import ProcessingSettings
from app.services.settings_service import get_processing_settings
from app.services.storage import Storage

log = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = (JobStatus.queued, JobStatus.processing)
PROGRESS_WRITE_INTERVAL = 0.5  # seconds


class JobConflictError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    def __init__(self, storage: Storage, workers: int = 1):
        self.storage = storage
        self.workers = max(1, workers)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._cancelled_videos: set[str] = set()
        self._active: dict[str, str] = {}  # job_id -> video_id
        self._idle = threading.Condition(self._lock)
        self._running = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.recover()
        for i in range(self.workers):
            t = threading.Thread(target=self._worker_loop, name=f"video-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("Job manager started with %d worker(s)", self.workers)

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        with self._lock:
            # Ask in-flight jobs to stop; they will be marked failed/interrupted on next start.
            self._cancelled_videos.update(self._active.values())
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        log.info("Job manager stopped")

    def recover(self) -> None:
        with session_scope() as db:
            interrupted = db.scalars(select(ProcessingJob).where(ProcessingJob.status == JobStatus.processing)).all()
            for job in interrupted:
                job.status = JobStatus.failed
                job.error_message = "Interrupted: the application stopped while this job was running. Retry to process again."
                job.finished_at = _now()
                if job.video:
                    job.video.status = VideoStatus.failed
                    job.video.error_message = job.error_message
            queued = db.scalars(
                select(ProcessingJob).where(ProcessingJob.status == JobStatus.queued).order_by(ProcessingJob.created_at)
            ).all()
            queued_ids = [j.id for j in queued]
        for jid in queued_ids:
            self._queue.put(jid)
        if interrupted or queued_ids:
            log.info("Recovered jobs: %d interrupted -> failed, %d re-queued", len(interrupted), len(queued_ids))

    # ------------------------------------------------------------------ API
    def create_job(self, db: Session, video: Video, job_type: str = "phase1_ingest") -> ProcessingJob:
        if job_type not in JOB_TYPES:
            raise ValueError(f"Unknown job type: {job_type}")
        active = db.scalar(
            select(ProcessingJob).where(
                ProcessingJob.video_id == video.id, ProcessingJob.status.in_(ACTIVE_JOB_STATUSES)
            )
        )
        if active is not None:
            raise JobConflictError(f"Video already has an active job ({active.status.value})")
        attempt = 1 + len(video.jobs)
        settings = get_processing_settings(db)
        job = ProcessingJob(
            video_id=video.id,
            job_type=job_type,
            status=JobStatus.queued,
            attempt=attempt,
            settings_snapshot=settings.model_dump(),
        )
        db.add(job)
        video.status = VideoStatus.queued
        video.error_message = None
        db.commit()
        db.refresh(job)
        self._queue.put(job.id)
        return job

    def is_video_active(self, video_id: str) -> bool:
        with self._lock:
            return video_id in self._active.values()

    def cancel_video(self, video_id: str, wait_seconds: float = 15.0) -> bool:
        """Signal cancellation for any running job on this video and wait for it to stop."""
        deadline = time.monotonic() + wait_seconds
        with self._idle:
            self._cancelled_videos.add(video_id)
            while video_id in self._active.values():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(timeout=remaining)
        return True

    def forget_cancel(self, video_id: str) -> None:
        with self._lock:
            self._cancelled_videos.discard(video_id)

    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ worker
    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                break
            try:
                self._process(job_id)
            except Exception:  # pragma: no cover - defensive
                log.exception("Unhandled error processing job %s", job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        # ---- claim the job
        with session_scope() as db:
            job = db.get(ProcessingJob, job_id)
            if job is None or job.status != JobStatus.queued:
                return
            video = job.video
            if video is None:
                return
            stored = self.storage.resolve(video.stored_path)
            video_id, file_ext = video.id, video.file_ext
            settings = ProcessingSettings.model_validate(job.settings_snapshot or {})
            job.status = JobStatus.processing
            job.started_at = _now()
            job.progress = 0.0
            video.status = VideoStatus.processing
            video.error_message = None

        with self._lock:
            if not self._running:
                return
            self._cancelled_videos.discard(video_id)
            self._active[job_id] = video_id

        log.info("Processing job %s for video %s", job_id, video_id)
        ctx = ProcessingContext(
            video_id=video_id,
            source_path=stored,
            file_ext=file_ext,
            settings=settings,
            storage=self.storage,
            is_cancelled=lambda: video_id in self._cancelled_videos or not self._running,
        )
        job_type = "phase1_ingest"
        with session_scope() as db:
            j = db.get(ProcessingJob, job_id)
            if j is not None:
                job_type = j.job_type
        runner = PipelineRunner(JOB_TYPES[job_type]())
        last_write = [0.0]

        def on_progress(stage: str, frac: float, stage_log: list[dict]) -> None:
            now = time.monotonic()
            if now - last_write[0] < PROGRESS_WRITE_INTERVAL and frac < 1.0:
                return
            last_write[0] = now
            try:
                with session_scope() as db:
                    j = db.get(ProcessingJob, job_id)
                    if j is not None:
                        j.progress = round(frac, 4)
                        j.current_stage = stage
                        j.stage_log = [dict(e) for e in stage_log]
            except Exception:  # progress writes are best-effort
                log.debug("Progress write failed", exc_info=True)

        try:
            runner.run(ctx, on_progress)
        except JobCancelled:
            self._handle_cancel(job_id, video_id, runner)
        except StageError as exc:
            self._handle_failure(job_id, video_id, str(exc), ctx, runner)
        except Exception as exc:
            log.exception("Job %s crashed", job_id)
            self._handle_failure(job_id, video_id, f"Unexpected error: {exc}", ctx, runner)
        else:
            self._handle_success(job_id, video_id, ctx, runner)
        finally:
            with self._idle:
                self._active.pop(job_id, None)
                self._idle.notify_all()

    # ------------------------------------------------------------------ outcomes
    def _handle_cancel(self, job_id: str, video_id: str, runner: PipelineRunner) -> None:
        log.info("Job %s cancelled", job_id)
        with session_scope() as db:
            job = db.get(ProcessingJob, job_id)
            video = db.get(Video, video_id)
            if job is not None:
                job.status = JobStatus.cancelled if self._running else JobStatus.failed
                job.error_message = "Cancelled" if self._running else "Interrupted by application shutdown. Retry to process again."
                job.finished_at = _now()
                job.stage_log = runner.stage_log
            if video is not None:
                video.status = VideoStatus.failed
                video.error_message = job.error_message if job else "Cancelled"
        # If the video was deleted, the delete endpoint removes files once we've stopped.

    def _handle_failure(self, job_id: str, video_id: str, message: str, ctx: ProcessingContext,
                        runner: PipelineRunner) -> None:
        log.warning("Job %s failed: %s", job_id, message)
        self.storage.remove_derived_artifacts(video_id)
        with session_scope() as db:
            job = db.get(ProcessingJob, job_id)
            video = db.get(Video, video_id)
            if job is not None:
                job.status = JobStatus.failed
                job.error_message = message
                job.finished_at = _now()
                job.stage_log = runner.stage_log
                job.warnings = ctx.warnings or None
            if video is not None:
                db.execute(delete(SampledFrame).where(SampledFrame.video_id == video_id))
                video.status = VideoStatus.failed
                video.error_message = message
                video.thumbnail_path = None
                video.preview_path = None
                video.sampled_frame_count = 0
                if ctx.media is not None:
                    self._apply_media(video, ctx)

    def _apply_media(self, video: Video, ctx: ProcessingContext) -> None:
        m = ctx.media
        assert m is not None
        video.duration_seconds = m.duration_seconds
        video.frame_count = m.frame_count
        video.frame_count_source = m.frame_count_source
        video.fps = round(m.fps, 3) if m.fps else None
        video.width, video.height = m.width, m.height
        video.rotation = m.rotation
        video.video_codec = m.video_codec
        video.pixel_format = m.pixel_format
        video.container_format = m.container_format
        video.bitrate_kbps = m.bitrate_kbps
        video.has_audio = m.has_audio
        video.audio_codec = m.audio_codec
        video.audio_sample_rate = m.audio_sample_rate
        video.audio_channels = m.audio_channels
        video.metadata_source = "+".join(m.sources) or None
        video.raw_metadata = m.raw

    def _handle_success(self, job_id: str, video_id: str, ctx: ProcessingContext, runner: PipelineRunner) -> None:
        with session_scope() as db:
            video = db.get(Video, video_id)
            job = db.get(ProcessingJob, job_id)
            if video is None:
                # Deleted mid-flight: don't leave files behind.
                self.storage.remove_derived_artifacts(video_id)
                return
            self._apply_media(video, ctx)
            video.thumbnail_path = self.storage.to_relative(ctx.thumbnail_path) if ctx.thumbnail_path else None
            video.preview_path = self.storage.to_relative(ctx.preview_path) if ctx.preview_path else None
            quality = dict(ctx.quality or {})
            if ctx.warnings:
                quality["warnings"] = ctx.warnings
            video.quality = quality or None
            s = ctx.settings
            video.sampling_config = {
                "mode": s.sampling_mode,
                "interval_seconds": s.sample_interval_seconds if s.sampling_mode == "interval" else None,
                "uniform_frame_count": s.uniform_frame_count if s.sampling_mode == "uniform" else None,
                "max_frames": s.max_sampled_frames,
                "images_saved": s.save_frame_images,
                "frame_max_width": s.frame_max_width,
                "jpeg_quality": s.frame_jpeg_quality,
            }
            db.execute(delete(SampledFrame).where(SampledFrame.video_id == video_id))
            db.add_all(
                SampledFrame(
                    video_id=video_id,
                    frame_index=f.frame_index,
                    timestamp_seconds=f.timestamp_seconds,
                    image_path=self.storage.to_relative(f.image_path) if f.image_path else None,
                    width=f.width,
                    height=f.height,
                    brightness=f.brightness,
                    contrast=f.contrast,
                    sharpness=f.sharpness,
                )
                for f in ctx.frames
            )
            video.sampled_frame_count = len(ctx.frames)
            video.status = VideoStatus.completed
            video.error_message = None
            video.processed_at = _now()
            if job is not None:
                job.status = JobStatus.completed
                job.progress = 1.0
                job.current_stage = None
                job.finished_at = _now()
                job.stage_log = runner.stage_log
                job.warnings = ctx.warnings or None
        log.info("Job %s completed: %d frames sampled", job_id, len(ctx.frames))
