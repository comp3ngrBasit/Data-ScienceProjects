"""Background queue for Phase 3: action-analysis jobs and model-training jobs (one worker thread)."""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session

from app.actions.data import build_dataset, load_video_pose
from app.actions.features import FeatureConfig
from app.actions.inference import build_sequences, detect_video, load_model, repeated_combinations
from app.actions.settings import ActionSettings, get_action_settings
from app.actions.taxonomy import UNKNOWN, group
from app.actions.training import train_algorithms
from app.core.config import get_settings
from app.db.action_models import ActionAnalysisJob, ActionModelVersion, ActionSequence, ActionTrainingJob, RecognizedAction
from app.db.database import session_scope
from app.db.models import JobStatus, Video
from app.pipeline.base import JobCancelled
from app.services.storage import Storage
from app.vision.job_manager import current_job as current_vision_job
from app.vision.runtime import resolve_device

log = logging.getLogger(__name__)
ACTIVE = (JobStatus.queued, JobStatus.processing)


class ActionConflict(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc)


def models_root():
    d = get_settings().data_dir / "action_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def active_model(db: Session) -> ActionModelVersion | None:
    return db.scalar(select(ActionModelVersion).where(ActionModelVersion.is_active.is_(True), ActionModelVersion.status == "ready"))


def current_action_job(db: Session, video_id: str) -> ActionAnalysisJob | None:
    return db.scalar(select(ActionAnalysisJob).where(ActionAnalysisJob.video_id == video_id,
                                                     ActionAnalysisJob.status == JobStatus.completed)
                     .order_by(ActionAnalysisJob.finished_at.desc()).limit(1))


def sequence_inputs(db: Session, job_id: str) -> list[dict]:
    """Model actions of a job ready for sequencing: corrected labels applied, rejected/unknown removed."""
    rows = db.scalars(select(RecognizedAction).where(RecognizedAction.job_id == job_id)).all()
    out = []
    for r in rows:
        if r.review_status == "rejected":
            continue
        cat = r.corrected_category or r.category
        if cat == UNKNOWN or (r.low_confidence and r.review_status == "unreviewed"):
            continue
        out.append({"id": r.id, "fighter_key": r.fighter_id or f"track:{r.track_id}", "fighter_id": r.fighter_id,
                    "track_id": r.track_id, "category": cat, "start": r.start_time, "end": r.end_time,
                    "confidence": r.confidence, "source": "review" if r.review_status in ("confirmed", "corrected") else "model"})
    return out


def rebuild_sequences(db: Session, job: ActionAnalysisJob, gap: float) -> list[dict]:
    db.execute(delete(ActionSequence).where(ActionSequence.job_id == job.id))
    seqs = build_sequences(sequence_inputs(db, job.id), gap)
    if seqs:
        db.execute(insert(ActionSequence), [{"job_id": job.id, "video_id": job.video_id, **s} for s in seqs])
    st = dict(job.stats or {})
    st["sequences"] = len(seqs)
    st["repeated_combinations"] = repeated_combinations(seqs)
    job.stats = st
    return seqs


class ActionJobManager:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._lock = threading.Condition()
        self._cancel: set[str] = set()  # video ids
        self._active: dict[str, str] = {}  # job id -> video id ("" for training)
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if self._running:
            return
        self._running = True
        with session_scope() as db:
            for model in (ActionAnalysisJob, ActionTrainingJob):
                db.execute(update(model).where(model.status == JobStatus.processing).values(
                    status=JobStatus.failed, finished_at=_now(),
                    error_message="Interrupted: the application stopped while this job was running. Retry to run again."))
            queued = [("analysis", j) for j in db.scalars(select(ActionAnalysisJob.id).where(ActionAnalysisJob.status == JobStatus.queued)).all()]
            queued += [("training", j) for j in db.scalars(select(ActionTrainingJob.id).where(ActionTrainingJob.status == JobStatus.queued)).all()]
        for item in queued:
            self._queue.put(item)
        self._thread = threading.Thread(target=self._loop, name="action-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout=15.0):
        if not self._running:
            return
        self._running = False
        with self._lock:
            self._cancel.update(v for v in self._active.values() if v)
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------ API
    def create_analysis(self, db: Session, video: Video, model_version: str | None, settings: ActionSettings) -> ActionAnalysisJob:
        if current_vision_job(db, video.id) is None:
            raise ActionConflict("Run Phase 2 (fighter tracking & pose) on this video first")
        model = db.get(ActionModelVersion, model_version) if model_version else active_model(db)
        if model is None or model.status != "ready":
            raise ActionConflict("No trained action model is available. Annotate examples and train a model first.")
        if db.scalar(select(ActionAnalysisJob).where(ActionAnalysisJob.video_id == video.id, ActionAnalysisJob.status.in_(ACTIVE))):
            raise ActionConflict("An action analysis for this video is already queued or running")
        attempt = (db.scalar(select(ActionAnalysisJob.attempt).where(ActionAnalysisJob.video_id == video.id)
                             .order_by(ActionAnalysisJob.attempt.desc()).limit(1)) or 0) + 1
        job = ActionAnalysisJob(video_id=video.id, model_version=model.version, settings=settings.model_dump(), attempt=attempt)
        db.add(job)
        db.commit()
        db.refresh(job)
        self._queue.put(("analysis", job.id))
        return job

    def create_training(self, db: Session, config: dict) -> ActionTrainingJob:
        if db.scalar(select(ActionTrainingJob).where(ActionTrainingJob.status.in_(ACTIVE))):
            raise ActionConflict("A training job is already queued or running")
        job = ActionTrainingJob(config=config)
        db.add(job)
        db.commit()
        db.refresh(job)
        self._queue.put(("training", job.id))
        return job

    def cancel_video(self, video_id: str, wait_seconds: float = 30.0) -> bool:
        deadline = time.monotonic() + wait_seconds
        with self._lock:
            self._cancel.add(video_id)
            while video_id in self._active.values():
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False
                self._lock.wait(timeout=rem)
        return True

    def forget_cancel(self, video_id: str):
        with self._lock:
            self._cancel.discard(video_id)

    # ------------------------------------------------------------ worker
    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            kind, jid = item
            try:
                if kind == "analysis":
                    self._run_analysis(jid)
                else:
                    self._run_training(jid)
            except Exception:  # pragma: no cover
                log.exception("Action %s job %s crashed", kind, jid)

    def _progress_writer(self, model, job_id):
        last = [0.0]

        def write(stage: str, frac: float):
            now = time.monotonic()
            if now - last[0] < 0.75 and frac < 0.99:
                return
            last[0] = now
            try:
                with session_scope() as db:
                    j = db.get(model, job_id)
                    if j is not None:
                        j.progress, j.current_stage = round(min(frac, 0.999), 4), stage
            except Exception:
                pass

        return write

    def _run_analysis(self, job_id: str):
        with session_scope() as db:
            job = db.get(ActionAnalysisJob, job_id)
            if job is None or job.status != JobStatus.queued:
                return
            video_id = job.video_id
            job.status, job.started_at, job.progress = JobStatus.processing, _now(), 0.0
        with self._lock:
            self._cancel.discard(video_id)
            self._active[job_id] = video_id
        progress = self._progress_writer(ActionAnalysisJob, job_id)
        try:
            with session_scope() as db:
                job = db.get(ActionAnalysisJob, job_id)
                s = ActionSettings.model_validate(job.settings or {})
                mv = db.get(ActionModelVersion, job.model_version)
                if mv is None:
                    raise RuntimeError(f"Model {job.model_version} no longer exists")
                progress("loading", 0.02)
                device = resolve_device(get_action_settings(db).device, False).device
                lm = load_model(mv.version, mv.algorithm, self.storage.resolve(mv.path), device, mv.metrics)
                vp = load_video_pose(db, self.storage, video_id)
                if vp is None:
                    raise RuntimeError("Phase 2 analysis results are missing for this video")
                job.vision_job_id = vp.vision_job.id
            dets, stats, warnings = detect_video(vp, lm, s, lambda f: progress("classifying windows", 0.05 + 0.85 * f),
                                                 lambda: video_id in self._cancel or not self._running)
            progress("saving", 0.92)
            rows = []
            for d in dets:
                sess = vp.sessions.get(d.track_id)
                pc = lm.per_class.get(d.category, {})
                rows.append({
                    "id": uuid.uuid4().hex, "job_id": job_id, "video_id": video_id, "track_id": d.track_id,
                    "fighter_id": sess.assigned_fighter_id if sess else None, "category": d.category,
                    "action_group": group(d.category), "start_time": round(d.start, 3), "end_time": round(d.end, 3),
                    "duration": round(d.end - d.start, 3), "start_frame": int(round(d.start * vp.fps)),
                    "end_frame": int(round(d.end * vp.fps)), "confidence": round(d.confidence, 4),
                    "runner_up": d.runner_up, "runner_up_confidence": round(d.runner_up_confidence, 4) if d.runner_up_confidence is not None else None,
                    "validated_precision": pc.get("precision_at_threshold") if d.category != UNKNOWN else None,
                    "validated_support": pc.get("confident_predictions") if d.category != UNKNOWN else None,
                    "limb": d.limb, "windows": d.windows, "low_confidence": d.low_confidence or d.confidence < s.confidence_threshold,
                    "model_version": lm.version, "review_status": "unreviewed",
                })
            with session_scope() as db:
                if rows:
                    for i in range(0, len(rows), 2000):
                        db.execute(insert(RecognizedAction), rows[i:i + 2000])
                job = db.get(ActionAnalysisJob, job_id)
                counts = Counter(r["category"] for r in rows)
                job.stats = {**stats, "actions": len(rows), "by_category": dict(counts),
                             "by_group": dict(Counter(r["action_group"] for r in rows)),
                             "model_classes": [c for c in lm.classes if c != "background"],
                             "window_seconds": s.window_seconds or lm.window_seconds, "pose_fps": vp.fps}
                db.flush()
                rebuild_sequences(db, job, s.sequence_gap)
                job.warnings = warnings or None
                job.status, job.progress, job.current_stage, job.finished_at = JobStatus.completed, 1.0, None, _now()
                # keep only the newest results per video
                older = [j.id for j in db.scalars(select(ActionAnalysisJob).where(
                    ActionAnalysisJob.video_id == video_id, ActionAnalysisJob.id != job_id,
                    ActionAnalysisJob.status == JobStatus.completed)).all()]
                if older:
                    db.execute(delete(ActionSequence).where(ActionSequence.job_id.in_(older)))
                    db.execute(delete(RecognizedAction).where(RecognizedAction.job_id.in_(older)))
        except JobCancelled:
            self._fail(ActionAnalysisJob, job_id, "Cancelled", JobStatus.cancelled)
        except Exception as exc:
            log.exception("Action analysis %s failed", job_id)
            self._fail(ActionAnalysisJob, job_id, f"{type(exc).__name__}: {exc}", JobStatus.failed)
        finally:
            with self._lock:
                self._active.pop(job_id, None)
                self._lock.notify_all()

    def _fail(self, model, job_id, msg, status):
        with session_scope() as db:
            if model is ActionAnalysisJob:
                db.execute(delete(ActionSequence).where(ActionSequence.job_id == job_id))
                db.execute(delete(RecognizedAction).where(RecognizedAction.job_id == job_id))
            j = db.get(model, job_id)
            if j is not None:
                j.status, j.error_message, j.finished_at = status, msg, _now()

    def _run_training(self, job_id: str):
        with session_scope() as db:
            job = db.get(ActionTrainingJob, job_id)
            if job is None or job.status != JobStatus.queued:
                return
            job.status, job.started_at = JobStatus.processing, _now()
            cfg = dict(job.config)
        with self._lock:
            self._active[job_id] = ""
        progress = self._progress_writer(ActionTrainingJob, job_id)
        created_dirs = []
        try:
            fc = FeatureConfig(seq_len=cfg["seq_len"], context=cfg["context"])
            progress("Building dataset from annotations", 0.02)
            with session_scope() as db:
                statuses = ("verified", "draft") if cfg.get("include_draft_annotations") else ("verified",)
                ds = build_dataset(db, self.storage, fc, statuses, cfg["negatives_per_positive"], cfg["augment"])
                j = db.get(ActionTrainingJob, job_id)
                j.dataset_version, j.dataset_summary = ds.version, ds.summary
            device = resolve_device(cfg.get("device", "auto"), False).device
            cfg.setdefault("params", {})
            for a in ("lstm", "tcn"):
                cfg["params"].setdefault(a, {})["epochs"] = cfg.get("epochs", 60)
            out = train_algorithms(ds, cfg, device, models_root(), lambda st, f: progress(st, 0.05 + 0.9 * f))
            created_dirs = [r["dir"] for r in out["results"].values()]
            with session_scope() as db:
                comparison = {}
                best = None
                for algo, r in out["results"].items():
                    m = r["metrics"]
                    rel = self.storage.to_relative(Path(r["dir"]))
                    db.add(ActionModelVersion(
                        version=r["version"], algorithm=algo, status="ready", training_job_id=job_id,
                        dataset_version=ds.version, classes=out["classes"], path=rel,
                        config={"feature_config": fc.as_dict(), "params": cfg["params"].get(algo, {}), "threshold": cfg["threshold"]},
                        metrics={**m, "evaluation": r["evaluation"]} if "error" not in m else {"error": m["error"], "evaluation": r["evaluation"]},
                        training_info={"device": device if algo in ("lstm", "tcn") else "cpu", "train_seconds": r["train_seconds"],
                                       "final_train_samples": r["final_train_samples"], "final_train_real": r["final_train_real"],
                                       "dataset": ds.summary, "dropped_classes": out["dropped"], **(r["model_info"] or {})},
                    ))
                    f1 = m.get("f1_macro") if "error" not in m else None
                    comparison[algo] = {"version": r["version"], "accuracy": m.get("accuracy"), "f1_macro": f1,
                                        "precision_macro": m.get("precision_macro"), "recall_macro": m.get("recall_macro"),
                                        "strategy": r["evaluation"].get("strategy"), "error": m.get("error")}
                    if f1 is not None and (best is None or f1 > best[1]):
                        best = (r["version"], f1)
                db.flush()
                if best and cfg.get("activate_best", True):
                    db.execute(update(ActionModelVersion).values(is_active=False))
                    db.execute(update(ActionModelVersion).where(ActionModelVersion.version == best[0]).values(is_active=True))
                j = db.get(ActionTrainingJob, job_id)
                warn = list(ds.warnings)
                if out["dropped"]:
                    warn.append("Classes not trained (insufficient data): " + "; ".join(f"{k}: {v}" for k, v in out["dropped"].items()))
                n_real = ds.summary["annotations_used"]
                if n_real < 200:
                    warn.append(f"Only {n_real} annotated examples were used. Metrics on so little data have very wide "
                                "uncertainty and do not indicate real-world accuracy.")
                j.results = {"comparison": comparison, "best": best[0] if best else None, "classes": out["classes"],
                             "dropped": out["dropped"]}
                j.warnings = warn or None
                j.status, j.progress, j.current_stage, j.finished_at = JobStatus.completed, 1.0, None, _now()
        except Exception as exc:
            log.exception("Training job %s failed", job_id)
            for d in created_dirs:
                shutil.rmtree(d, ignore_errors=True)
            self._fail(ActionTrainingJob, job_id, str(exc), JobStatus.failed)
        finally:
            with self._lock:
                self._active.pop(job_id, None)
                self._lock.notify_all()
