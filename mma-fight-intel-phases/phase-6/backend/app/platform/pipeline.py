"""Unified analysis pipeline (Phase 6).

Video Upload -> (1) processing/metadata -> (2) fighter detection, tracking & pose -> (3) action recognition ->
(4) pattern analysis -> (5) next-move prediction. Each stage is the existing phase's own background job, started
only after the previous stage completed, so every stage consumes the real output of the one before.

Rules: one active run per video (duplicates return the running one); existing up-to-date results are reused
unless ``force``; a stage that cannot run because a prerequisite model does not exist is *skipped* with a reason
(never faked); a failed stage fails the run with the stage's own error; runs interrupted by a shutdown are marked
failed on restart and can be started again.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.actions.job_manager import active_model as active_action_model
from app.actions.job_manager import current_action_job
from app.actions.settings import get_action_settings
from app.db.action_models import ActionAnalysisJob
from app.db.database import session_scope
from app.db.models import JobStatus, ProcessingJob, Video, VideoStatus
from app.db.pattern_models import PatternAnalysisJob
from app.db.platform_models import PipelineRun
from app.db.prediction_models import PredictionVideoJob
from app.db.vision_models import VisionJob
from app.predictions.service import active_model as active_prediction_model
from app.vision.job_manager import current_job as current_vision_job

log = logging.getLogger("app.pipeline")
STAGES = ("processing", "vision", "actions", "patterns", "prediction")
STAGE_LABEL = {"processing": "Video processing", "vision": "Detection, tracking & pose", "actions": "Action recognition",
               "patterns": "Pattern analysis", "prediction": "Next-move prediction"}
ACTIVE = (JobStatus.queued, JobStatus.processing)


def _now():
    return datetime.now(timezone.utc)


class StageFailed(Exception):
    pass


class PipelineManager:
    def __init__(self, managers: dict, poll: float = 1.0):
        self.m = managers
        self.poll = poll
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        with session_scope() as db:
            db.execute(update(PipelineRun).where(PipelineRun.status.in_(ACTIVE)).values(
                status=JobStatus.failed, finished_at=_now(),
                error_message="Interrupted: the application stopped while this pipeline was running. Start it again to continue "
                              "(completed stages are reused)."))

    def stop(self):
        self._stop.set()
        for t in list(self._threads.values()):
            t.join(timeout=5)

    def create(self, db: Session, video: Video, force: bool = False) -> tuple[PipelineRun, bool]:
        with self._lock:
            existing = db.scalar(select(PipelineRun).where(PipelineRun.video_id == video.id, PipelineRun.status.in_(ACTIVE)))
            if existing:
                return existing, False
            run = PipelineRun(video_id=video.id, stages=[{"name": s, "label": STAGE_LABEL[s], "status": "pending"} for s in STAGES],
                              options={"force": force})
            db.add(run)
            db.commit()
            db.refresh(run)
            t = threading.Thread(target=self._run, args=(run.id,), name=f"pipeline-{run.id[:6]}", daemon=True)
            self._threads[run.id] = t
            t.start()
            return run, True

    # ------------------------------------------------------------------ helpers
    def _set_stage(self, run_id: str, name: str, **kw):
        with session_scope() as db:
            run = db.get(PipelineRun, run_id)
            stages = [dict(s) for s in run.stages]
            for s in stages:
                if s["name"] == name:
                    s.update(kw)
            run.stages = stages
            run.current_stage = name
            done = sum(1 for s in stages if s["status"] in ("completed", "skipped", "reused"))
            run.progress = round(done / len(stages), 4)

    def _wait(self, model, job_id: str, run_id: str, stage: str) -> object:
        while not self._stop.is_set():
            with session_scope() as db:
                j = db.get(model, job_id)
                status = j.status
                prog = getattr(j, "progress", None)
                err = getattr(j, "error_message", None)
            if status == JobStatus.completed:
                return status
            if status in (JobStatus.failed, JobStatus.cancelled):
                raise StageFailed(err or f"{stage} job {status.value}")
            if prog is not None:
                self._set_stage(run_id, stage, job_progress=round(float(prog), 4))
            time.sleep(self.poll)
        raise StageFailed("Application is shutting down")

    def _wait_video(self, video_id: str, run_id: str):
        while not self._stop.is_set():
            with session_scope() as db:
                v = db.get(Video, video_id)
                st, err = v.status, v.error_message
            if st == VideoStatus.completed:
                return
            if st == VideoStatus.failed:
                raise StageFailed(err or "Video processing failed")
            time.sleep(self.poll)
        raise StageFailed("Application is shutting down")

    # ------------------------------------------------------------------ the run
    def _run(self, run_id: str):
        with session_scope() as db:
            run = db.get(PipelineRun, run_id)
            run.status, run.started_at = JobStatus.processing, _now()
            video_id, force = run.video_id, bool(run.options.get("force"))
        log.info("pipeline started", extra={"run_id": run_id, "video_id": video_id})
        stage = "processing"
        try:
            # 1. Phase 1 processing
            self._set_stage(run_id, stage, status="running", started_at=_now().isoformat())
            with session_scope() as db:
                v = db.get(Video, video_id)
                if v.status == VideoStatus.completed and not force:
                    msg, job_id = "already processed — reused", None
                else:
                    active = db.scalar(select(ProcessingJob).where(ProcessingJob.video_id == video_id, ProcessingJob.status.in_(ACTIVE)))
                    job_id = active.id if active else self.m["processing"].create_job(db, v).id
                    msg = "processing"
            if job_id:
                self._set_stage(run_id, stage, job_id=job_id)
                self._wait_video(video_id, run_id)
                msg = "completed"
            self._set_stage(run_id, stage, status="reused" if "reused" in msg else "completed", message=msg, finished_at=_now().isoformat())

            # 2. Phase 2 detection, tracking, pose
            stage = "vision"
            self._set_stage(run_id, stage, status="running", started_at=_now().isoformat())
            with session_scope() as db:
                cur = current_vision_job(db, video_id)
                active = db.scalar(select(VisionJob).where(VisionJob.video_id == video_id, VisionJob.status.in_(ACTIVE)))
                if cur and not force and not active:
                    vjob, reused = cur.id, True
                else:
                    vjob, reused = (active.id if active else self.m["vision"].create_job(db, db.get(Video, video_id)).id), False
            self._set_stage(run_id, stage, job_id=vjob)
            if not reused:
                self._wait(VisionJob, vjob, run_id, stage)
            self._set_stage(run_id, stage, status="reused" if reused else "completed",
                            message="existing analysis reused" if reused else "completed", finished_at=_now().isoformat())

            # 3. Phase 3 action recognition
            stage = "actions"
            self._set_stage(run_id, stage, status="running", started_at=_now().isoformat())
            with session_scope() as db:
                model = active_action_model(db)
                cur = current_action_job(db, video_id)
                active = db.scalar(select(ActionAnalysisJob).where(ActionAnalysisJob.video_id == video_id, ActionAnalysisJob.status.in_(ACTIVE)))
                if model is None and not active:
                    ajob, mode = None, "skipped"
                elif cur and cur.vision_job_id == vjob and cur.model_version == (model.version if model else None) and not force and not active:
                    ajob, mode = cur.id, "reused"
                else:
                    ajob = active.id if active else self.m["actions"].create_analysis(db, db.get(Video, video_id), None, get_action_settings(db)).id
                    mode = "run"
            if mode == "skipped":
                self._set_stage(run_id, stage, status="skipped", message="No trained action recognition model — annotate and train one "
                                "(MMA Action Recognition → Models). Later stages use existing actions only.", finished_at=_now().isoformat())
            else:
                self._set_stage(run_id, stage, job_id=ajob)
                if mode == "run":
                    self._wait(ActionAnalysisJob, ajob, run_id, stage)
                self._set_stage(run_id, stage, status="reused" if mode == "reused" else "completed",
                                message="existing recognition reused" if mode == "reused" else "completed", finished_at=_now().isoformat())

            # 4. Phase 4 patterns (every affected fighter; unchanged fighters are skipped by the pattern job itself)
            stage = "patterns"
            self._set_stage(run_id, stage, status="running", started_at=_now().isoformat())
            with session_scope() as db:
                pjob = self.m["patterns"].create(db, "*").id
            self._set_stage(run_id, stage, job_id=pjob)
            self._wait(PatternAnalysisJob, pjob, run_id, stage)
            self._set_stage(run_id, stage, status="completed", message="pattern analysis up to date", finished_at=_now().isoformat())

            # 5. Phase 5 next-move prediction through the video
            stage = "prediction"
            self._set_stage(run_id, stage, status="running", started_at=_now().isoformat())
            with session_scope() as db:
                pm = active_prediction_model(db)
                pv = self.m["prediction"].video(db, video_id, None, None).id if pm else None
            if pv is None:
                self._set_stage(run_id, stage, status="skipped", message="No active next-move prediction model — train one "
                                "(Model Management).", finished_at=_now().isoformat())
            else:
                self._set_stage(run_id, stage, job_id=pv)
                self._wait(PredictionVideoJob, pv, run_id, stage)
                self._set_stage(run_id, stage, status="completed", message="predictions stored", finished_at=_now().isoformat())

            with session_scope() as db:
                run = db.get(PipelineRun, run_id)
                run.status, run.progress, run.finished_at, run.current_stage = JobStatus.completed, 1.0, _now(), None
            log.info("pipeline completed", extra={"run_id": run_id, "video_id": video_id})
        except Exception as exc:
            msg = str(exc) if isinstance(exc, StageFailed) else f"{type(exc).__name__}: {exc}"
            log.warning("pipeline failed", extra={"run_id": run_id, "video_id": video_id, "stage": stage, "error": msg[:500]})
            self._set_stage(run_id, stage, status="failed", message=msg[:500], finished_at=_now().isoformat())
            with session_scope() as db:
                run = db.get(PipelineRun, run_id)
                run.status, run.error_message, run.finished_at = JobStatus.failed, f"{STAGE_LABEL[stage]}: {msg}"[:2000], _now()
        finally:
            self._threads.pop(run_id, None)


def run_json(r: PipelineRun) -> dict:
    return {"id": r.id, "video_id": r.video_id, "status": r.status.value, "progress": r.progress, "current_stage": r.current_stage,
            "stages": r.stages, "options": r.options, "error_message": r.error_message,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}
