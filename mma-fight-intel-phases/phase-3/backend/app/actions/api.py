"""Phase 3 REST API: /api/actions/..."""
from __future__ import annotations

import io
import logging
from collections import Counter
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.actions.data import build_dataset
from app.actions.features import FeatureConfig
from app.actions.importer import parse_rows, read_table
from app.actions.inference import build_sequences
from app.actions.job_manager import (
    ACTIVE,
    ActionConflict,
    ActionJobManager,
    active_model,
    current_action_job,
    rebuild_sequences,
    sequence_inputs,
)
from app.actions.models import ALGORITHMS
from app.actions.settings import ActionSettings, get_action_settings, save_action_settings
from app.actions.taxonomy import BACKGROUND, CATEGORIES, UNKNOWN, group, label, normalise_key
from app.api.deps import get_db, get_storage
from app.db.action_models import (
    ActionAnalysisJob,
    ActionAnnotation,
    ActionModelVersion,
    ActionSequence,
    ActionTrainingJob,
    RecognizedAction,
    VideoAnnotationMeta,
)
from app.db.models import Fighter, JobStatus, Video
from app.db.vision_models import TrackingSession
from app.services.storage import Storage
from app.vision.job_manager import current_job as current_vision_job

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/actions", tags=["action recognition (phase 3)"])


def get_action_manager(request: Request) -> ActionJobManager:
    return request.app.state.action_manager


# ============================================================================ schemas
class AnnotationIn(BaseModel):
    category: str
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    fighter_id: str | None = None
    track_id: int | None = None
    confidence: float = Field(1.0, ge=0, le=1, description="Annotator certainty")
    opponent_reaction: str | None = Field(None, max_length=40)
    status: Literal["draft", "verified", "needs_review", "rejected"] = "verified"
    notes: str | None = Field(None, max_length=2000)

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        k = normalise_key(v)
        if k is None:
            raise ValueError(f"Unknown action category '{v}'")
        return k

    @model_validator(mode="after")
    def _times(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.end_time - self.start_time > 30:
            raise ValueError("An annotated action cannot be longer than 30 seconds")
        return self


class AnnotationUpdate(BaseModel):
    category: str | None = None
    start_time: float | None = Field(None, ge=0)
    end_time: float | None = Field(None, gt=0)
    fighter_id: str | None = None
    track_id: int | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    opponent_reaction: str | None = None
    status: Literal["draft", "verified", "needs_review", "rejected"] | None = None
    notes: str | None = None

    @field_validator("category")
    @classmethod
    def _cat(cls, v):
        if v is None:
            return v
        k = normalise_key(v)
        if k is None:
            raise ValueError(f"Unknown action category '{v}'")
        return k


class ReviewIn(BaseModel):
    decision: Literal["confirm", "correct", "reject"]
    category: str | None = None
    fighter_id: str | None = None

    @model_validator(mode="after")
    def _c(self):
        if self.decision == "correct":
            if not self.category or normalise_key(self.category) is None:
                raise ValueError("A valid category is required to correct an action")
            self.category = normalise_key(self.category)
        return self


class TrainIn(BaseModel):
    algorithms: list[str] | None = None
    activate_best: bool = True
    threshold: float | None = Field(None, ge=0.05, le=0.99)

    @field_validator("algorithms")
    @classmethod
    def _a(cls, v):
        if v is None:
            return v
        bad = [a for a in v if a not in ALGORITHMS]
        if bad:
            raise ValueError(f"Unknown algorithm(s) {bad}. Choose from {ALGORITHMS}")
        return v


class MetaIn(BaseModel):
    fully_annotated: bool | None = None
    split: Literal["auto", "train", "test", "exclude"] | None = None
    split_group: str | None = Field(None, max_length=64)
    notes: str | None = None


# ============================================================================ helpers
def _video(db: Session, vid: str) -> Video:
    v = db.get(Video, vid)
    if v is None:
        raise HTTPException(404, "Video not found")
    return v


def ann_out(a: ActionAnnotation, names: dict[str, str]) -> dict:
    return {"id": a.id, "video_id": a.video_id, "fighter_id": a.fighter_id, "fighter_name": names.get(a.fighter_id or ""),
            "track_id": a.track_id, "category": a.category, "label": label(a.category), "group": group(a.category),
            "start_time": a.start_time, "end_time": a.end_time, "duration": round(a.end_time - a.start_time, 3),
            "confidence": a.confidence, "opponent_reaction": a.opponent_reaction, "status": a.status, "source": a.source,
            "corrects_action_id": a.corrects_action_id, "notes": a.notes,
            "created_at": a.created_at.isoformat(), "updated_at": a.updated_at.isoformat()}


def action_out(r: RecognizedAction, names: dict[str, str], assigned: dict[int, str | None], threshold: float) -> dict:
    fid = assigned.get(r.track_id, r.fighter_id)
    cat = r.corrected_category or r.category
    return {
        "id": r.id, "video_id": r.video_id, "track_id": r.track_id, "fighter_id": fid, "fighter_name": names.get(fid or ""),
        "category": cat, "label": label(cat), "group": group(cat), "model_category": r.category,
        "start_time": r.start_time, "end_time": r.end_time, "duration": r.duration,
        "start_frame": r.start_frame, "end_frame": r.end_frame,
        "confidence": r.confidence, "runner_up": r.runner_up, "runner_up_label": label(r.runner_up) if r.runner_up else None,
        "runner_up_confidence": r.runner_up_confidence,
        "validated_precision": r.validated_precision, "validated_support": r.validated_support,
        "low_confidence": r.low_confidence, "limb": r.limb, "windows": r.windows,
        "model_version": r.model_version, "review_status": r.review_status, "threshold": threshold,
    }


def _names(db: Session) -> dict[str, str]:
    return {f.id: f.name for f in db.scalars(select(Fighter)).all()}


def _assigned(db: Session, vision_job_id: str | None) -> dict[int, str | None]:
    if not vision_job_id:
        return {}
    return {s.track_id: s.assigned_fighter_id for s in db.scalars(
        select(TrackingSession).where(TrackingSession.job_id == vision_job_id)).all()}


def job_out(j: ActionAnalysisJob) -> dict:
    return {"id": j.id, "video_id": j.video_id, "status": j.status.value, "progress": j.progress, "current_stage": j.current_stage,
            "model_version": j.model_version, "attempt": j.attempt, "stats": j.stats, "warnings": j.warnings,
            "error_message": j.error_message, "settings": j.settings, "created_at": j.created_at.isoformat(),
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


def model_out(m: ActionModelVersion, full: bool = False) -> dict:
    met = m.metrics or {}
    d = {"version": m.version, "algorithm": m.algorithm, "status": m.status, "is_active": m.is_active,
         "dataset_version": m.dataset_version, "classes": m.classes,
         "class_labels": [label(c) for c in m.classes], "created_at": m.created_at.isoformat(),
         "accuracy": met.get("accuracy"), "precision_macro": met.get("precision_macro"),
         "recall_macro": met.get("recall_macro"), "f1_macro": met.get("f1_macro"), "error": met.get("error"),
         "evaluation_strategy": (met.get("evaluation") or {}).get("strategy"), "n_evaluated": met.get("n_evaluated"),
         "training_job_id": m.training_job_id}
    if full:
        d.update({"metrics": met, "training_info": m.training_info, "config": m.config})
    return d


def training_out(j: ActionTrainingJob) -> dict:
    return {"id": j.id, "status": j.status.value, "progress": j.progress, "current_stage": j.current_stage,
            "config": j.config, "dataset_version": j.dataset_version, "dataset_summary": j.dataset_summary,
            "results": j.results, "warnings": j.warnings, "error_message": j.error_message,
            "created_at": j.created_at.isoformat(), "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


# ============================================================================ taxonomy / settings
@router.get("/taxonomy", summary="All action categories (and which ones the active model can recognise)")
def taxonomy(db: Session = Depends(get_db)):
    m = active_model(db)
    trained = set(m.classes) if m else set()
    return {"categories": [{"key": c.key, "label": c.label, "group": c.group, "family": c.family, "coarse": c.coarse,
                            "trained": c.key in trained} for c in CATEGORIES],
            "active_model": m.version if m else None}


@router.get("/settings", response_model=ActionSettings)
def read_settings(db: Session = Depends(get_db)):
    return get_action_settings(db)


@router.put("/settings", response_model=ActionSettings)
def write_settings(payload: ActionSettings, db: Session = Depends(get_db)):
    return save_action_settings(db, payload)


# ============================================================================ models (declared before /{video_id} routes)
@router.get("/models/status", summary="Trained models, the active model, and training jobs")
def models_status(db: Session = Depends(get_db)):
    models = db.scalars(select(ActionModelVersion).order_by(ActionModelVersion.created_at.desc())).all()
    jobs = db.scalars(select(ActionTrainingJob).order_by(ActionTrainingJob.created_at.desc()).limit(20)).all()
    am = next((m for m in models if m.is_active), None)
    return {"active_model": model_out(am) if am else None, "models": [model_out(m) for m in models],
            "training_jobs": [training_out(j) for j in jobs],
            "training_active": any(j.status in ACTIVE for j in jobs)}


@router.post("/models/train", status_code=202, summary="Train & evaluate models from verified annotations (background)")
def train(body: TrainIn | None = None, db: Session = Depends(get_db), am: ActionJobManager = Depends(get_action_manager)):
    s = get_action_settings(db)
    body = body or TrainIn()
    cfg = {"algorithms": body.algorithms or s.algorithms, "activate_best": body.activate_best,
           "threshold": body.threshold or s.confidence_threshold, "min_examples": s.min_examples, "min_groups": s.min_groups,
           "augment": s.augment, "negatives_per_positive": s.negatives_per_positive, "seq_len": s.seq_len,
           "context": s.context, "include_draft_annotations": s.include_draft_annotations, "epochs": s.epochs, "device": s.device}
    try:
        job = am.create_training(db, cfg)
    except ActionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return training_out(job)


@router.get("/models/evaluation", summary="Evaluation metrics of a model (default: active model)")
def evaluation(version: str | None = None, db: Session = Depends(get_db)):
    m = db.get(ActionModelVersion, version) if version else active_model(db)
    if m is None:
        raise HTTPException(404, "No model found" if version else "No active model — train one first")
    return model_out(m, full=True)


@router.post("/models/{version}/activate", summary="Use this model for new analyses")
def activate(version: str, db: Session = Depends(get_db)):
    m = db.get(ActionModelVersion, version)
    if m is None or m.status != "ready":
        raise HTTPException(404, "Model not found")
    db.execute(update(ActionModelVersion).values(is_active=False))
    m.is_active = True
    db.commit()
    return model_out(m)


@router.delete("/models/{version}", status_code=204, summary="Delete a model version")
def delete_model(version: str, db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    m = db.get(ActionModelVersion, version)
    if m is None:
        raise HTTPException(404, "Model not found")
    import shutil

    p = storage.resolve(m.path)
    db.delete(m)
    db.commit()
    if p is not None:
        shutil.rmtree(p, ignore_errors=True)
    return Response(status_code=204)


@router.get("/dataset", summary="Preview the training dataset that would be built from current annotations")
def dataset_preview(db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    s = get_action_settings(db)
    statuses = ("verified", "draft") if s.include_draft_annotations else ("verified",)
    ds = build_dataset(db, storage, FeatureConfig(seq_len=s.seq_len, context=s.context), statuses, s.negatives_per_positive, augment=0)
    counts = ds.summary["class_counts"]
    return {**ds.summary, "warnings": ds.warnings,
            "class_table": [{"key": k, "label": label(k), "examples": v, "groups": ds.summary["groups_per_class"].get(k, 0),
                             "trainable": k == BACKGROUND or (v >= s.min_examples and ds.summary["groups_per_class"].get(k, 0) >= s.min_groups)}
                            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
            "requirements": {"min_examples": s.min_examples, "min_groups": s.min_groups}}


@router.get("/overview", summary="Dashboard statistics for action recognition")
def overview(db: Session = Depends(get_db)):
    am = active_model(db)
    ann_total = db.scalar(select(func.count()).select_from(ActionAnnotation)) or 0
    ann_verified = db.scalar(select(func.count()).select_from(ActionAnnotation).where(ActionAnnotation.status == "verified")) or 0
    by_cat = dict(db.execute(select(RecognizedAction.category, func.count()).group_by(RecognizedAction.category)).all())
    by_group = dict(db.execute(select(RecognizedAction.action_group, func.count()).group_by(RecognizedAction.action_group)).all())
    jobs = db.execute(select(ActionAnalysisJob.status, func.count()).group_by(ActionAnalysisJob.status)).all()
    analysed = db.scalar(select(func.count(func.distinct(ActionAnalysisJob.video_id))).where(ActionAnalysisJob.status == JobStatus.completed)) or 0
    active_jobs = db.execute(select(ActionAnalysisJob, Video.title).join(Video, Video.id == ActionAnalysisJob.video_id)
                             .where(ActionAnalysisJob.status.in_(ACTIVE))).all()
    tj = db.scalar(select(ActionTrainingJob).where(ActionTrainingJob.status.in_(ACTIVE)).limit(1))
    return {
        "active_model": model_out(am) if am else None,
        "annotations": {"total": ann_total, "verified": ann_verified,
                        "by_category": dict(db.execute(select(ActionAnnotation.category, func.count()).group_by(ActionAnnotation.category)).all())},
        "videos_analysed": analysed,
        "actions_detected": sum(by_cat.values()),
        "by_category": [{"key": k, "label": label(k), "group": group(k), "count": v} for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])],
        "by_group": by_group,
        "job_counts": {s.value: dict(jobs).get(s, 0) for s in JobStatus},
        "active_jobs": [{**job_out(j), "video_title": t} for j, t in active_jobs],
        "training": training_out(tj) if tj else None,
    }


# ============================================================================ annotations by id
@router.put("/annotations/{annotation_id}", summary="Update / correct an annotation")
def update_annotation(annotation_id: str, body: AnnotationUpdate, db: Session = Depends(get_db)):
    a = db.get(ActionAnnotation, annotation_id)
    if a is None:
        raise HTTPException(404, "Annotation not found")
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(a, k, v)
    if a.end_time <= a.start_time:
        raise HTTPException(422, "end_time must be after start_time")
    if a.fighter_id and db.get(Fighter, a.fighter_id) is None:
        raise HTTPException(422, "Unknown fighter id")
    db.commit()
    return ann_out(a, _names(db))


@router.delete("/annotations/{annotation_id}", status_code=204, summary="Delete an annotation")
def delete_annotation(annotation_id: str, db: Session = Depends(get_db)):
    a = db.get(ActionAnnotation, annotation_id)
    if a is None:
        raise HTTPException(404, "Annotation not found")
    db.delete(a)
    db.commit()
    return Response(status_code=204)


@router.post("/actions/{action_id}/review", summary="Confirm, correct or reject a recognised action (corrections become training data)")
def review_action(action_id: str, body: ReviewIn, db: Session = Depends(get_db)):
    r = db.get(RecognizedAction, action_id)
    if r is None:
        raise HTTPException(404, "Action not found")
    job = db.get(ActionAnalysisJob, r.job_id)
    if body.decision == "reject":
        r.review_status, r.corrected_category = "rejected", None
    elif body.decision == "confirm":
        r.review_status, r.corrected_category = "confirmed", None
    else:
        r.review_status, r.corrected_category = "corrected", body.category
    # mirror the decision into the annotation set so it is used for future training
    existing = db.scalar(select(ActionAnnotation).where(ActionAnnotation.corrects_action_id == r.id))
    fighter_id = body.fighter_id or r.fighter_id
    if body.decision in ("confirm", "correct"):
        cat = r.corrected_category or r.category
        if cat != UNKNOWN:
            if existing is None:
                existing = ActionAnnotation(video_id=r.video_id, corrects_action_id=r.id, source="correction")
                db.add(existing)
            existing.category, existing.start_time, existing.end_time = cat, r.start_time, r.end_time
            existing.track_id, existing.fighter_id, existing.status = r.track_id, fighter_id, "verified"
    elif existing is not None:
        db.delete(existing)
    db.flush()
    if job is not None:
        rebuild_sequences(db, job, (job.settings or {}).get("sequence_gap", 2.0))
    db.commit()
    return action_out(r, _names(db), _assigned(db, job.vision_job_id if job else None), (job.settings or {}).get("confidence_threshold", 0.6) if job else 0.6)


# ============================================================================ per video
@router.post("/{video_id}/analyze", status_code=202, summary="Recognise actions in a video (background job)")
def analyze(video_id: str, model_version: str | None = Query(None), db: Session = Depends(get_db),
            am: ActionJobManager = Depends(get_action_manager)):
    v = _video(db, video_id)
    try:
        job = am.create_analysis(db, v, model_version, get_action_settings(db))
    except ActionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return job_out(job)


@router.get("/{video_id}/status", summary="Action analysis status and history")
def analysis_status(video_id: str, db: Session = Depends(get_db)):
    v = _video(db, video_id)
    jobs = db.scalars(select(ActionAnalysisJob).where(ActionAnalysisJob.video_id == video_id).order_by(ActionAnalysisJob.created_at.desc())).all()
    vj = current_vision_job(db, video_id)
    am_ = active_model(db)
    active = next((j for j in jobs if j.status in ACTIVE), None)
    reason = None
    if vj is None:
        reason = "Run Phase 2 fighter tracking & pose on this video first"
    elif am_ is None:
        reason = "No trained action model yet — annotate examples and train a model"
    elif active:
        reason = f"Analysis already {active.status.value}"
    cur = current_action_job(db, video_id)
    stride = int(((vj.stats or {}).get("stride") or 1)) if vj else None
    return {"video_id": video_id, "video_title": v.title, "can_analyze": reason is None, "reason": reason,
            "active_job": job_out(active) if active else None, "current": job_out(cur) if cur else None,
            "history": [job_out(j) for j in jobs], "active_model": am_.version if am_ else None,
            "vision": {"job_id": vj.id, "stride": stride, "pose_fps": round((vj.stats or {}).get("fps", 30) / stride, 2),
                       "annotated_video": bool(vj.annotated_video_path)} if vj else None}


def _results_rows(db: Session, video_id: str, track_id=None, fighter_id=None, category=None, group_=None,
                  min_confidence=0.0, include_unknown=True, include_rejected=False):
    job = current_action_job(db, video_id)
    if job is None:
        raise HTTPException(404, "No completed action analysis for this video yet")
    q = select(RecognizedAction).where(RecognizedAction.job_id == job.id, RecognizedAction.confidence >= min_confidence)
    if track_id is not None:
        q = q.where(RecognizedAction.track_id == track_id)
    if category:
        q = q.where(RecognizedAction.category == category)
    if group_:
        q = q.where(RecognizedAction.action_group == group_)
    if not include_unknown:
        q = q.where(RecognizedAction.category != UNKNOWN)
    if not include_rejected:
        q = q.where(RecognizedAction.review_status != "rejected")
    rows = db.scalars(q.order_by(RecognizedAction.start_time)).all()
    assigned = _assigned(db, job.vision_job_id)
    if fighter_id:
        rows = [r for r in rows if assigned.get(r.track_id, r.fighter_id) == fighter_id]
    return job, rows, assigned


@router.get("/{video_id}/results", summary="Recognised actions (one structured record per action)")
def results(video_id: str, track_id: int | None = None, fighter_id: str | None = None, category: str | None = None,
            action_group: str | None = Query(None, alias="group"), min_confidence: float = Query(0.0, ge=0, le=1),
            include_unknown: bool = True, include_rejected: bool = False, db: Session = Depends(get_db)):
    _video(db, video_id)
    job, rows, assigned = _results_rows(db, video_id, track_id, fighter_id, category, action_group, min_confidence,
                                        include_unknown, include_rejected)
    names = _names(db)
    thr = (job.settings or {}).get("confidence_threshold", 0.6)
    items = [action_out(r, names, assigned, thr) for r in rows]
    return {"video_id": video_id, "job": job_out(job), "model_version": job.model_version, "total": len(items),
            "confidence_note": ("'confidence' is the model's softmax/probability output for the predicted class. "
                                "'validated_precision' is how often that class was correct on held-out videos when the model "
                                "was at least this confident (with 'validated_support' predictions). Low support means the "
                                "precision itself is uncertain."),
            "items": items}


@router.get("/{video_id}/timeline", summary="Actions per fighter for timeline display (with ground-truth annotations)")
def timeline(video_id: str, db: Session = Depends(get_db)):
    v = _video(db, video_id)
    names = _names(db)
    lanes: dict[str, dict] = {}
    job = current_action_job(db, video_id)
    if job is not None:
        _, rows, assigned = _results_rows(db, video_id)
        thr = (job.settings or {}).get("confidence_threshold", 0.6)
        for r in rows:
            fid = assigned.get(r.track_id, r.fighter_id)
            key = fid or f"track:{r.track_id}"
            lane = lanes.setdefault(key, {"key": key, "fighter_id": fid, "name": names.get(fid or "", f"Track {r.track_id}"),
                                          "track_ids": set(), "actions": [], "annotations": []})
            lane["track_ids"].add(r.track_id)
            lane["actions"].append(action_out(r, names, assigned, thr))
    anns = db.scalars(select(ActionAnnotation).where(ActionAnnotation.video_id == video_id).order_by(ActionAnnotation.start_time)).all()
    for a in anns:
        key = a.fighter_id or (f"track:{a.track_id}" if a.track_id is not None else "unassigned")
        lane = lanes.setdefault(key, {"key": key, "fighter_id": a.fighter_id,
                                      "name": names.get(a.fighter_id or "", f"Track {a.track_id}" if a.track_id is not None else "Unassigned"),
                                      "track_ids": set(), "actions": [], "annotations": []})
        lane["annotations"].append(ann_out(a, names))
    out = []
    for lane in lanes.values():
        lane["track_ids"] = sorted(lane["track_ids"])
        out.append(lane)
    out.sort(key=lambda lane: (lane["fighter_id"] is None, lane["name"]))
    return {"video_id": video_id, "duration": v.duration_seconds, "fps": v.fps, "job_id": job.id if job else None,
            "model_version": job.model_version if job else None, "lanes": out}


@router.get("/{video_id}/sequences", summary="Chronological, fighter-specific action sequences with time intervals")
def sequences(video_id: str, source: Literal["model", "annotations"] = "model", gap: float | None = Query(None, gt=0, le=30),
              db: Session = Depends(get_db)):
    _video(db, video_id)
    names = _names(db)
    if source == "annotations":
        anns = db.scalars(select(ActionAnnotation).where(ActionAnnotation.video_id == video_id,
                                                         ActionAnnotation.status == "verified",
                                                         ActionAnnotation.category != UNKNOWN)).all()
        g = gap or get_action_settings(db).sequence_gap
        seqs = build_sequences([{"id": a.id, "fighter_key": a.fighter_id or f"track:{a.track_id}", "fighter_id": a.fighter_id,
                                 "track_id": a.track_id if a.track_id is not None else -1, "category": a.category,
                                 "start": a.start_time, "end": a.end_time, "confidence": a.confidence, "source": "annotation"}
                                for a in anns], g)
        job_id, model_version = None, None
    else:
        job = current_action_job(db, video_id)
        if job is None:
            raise HTTPException(404, "No completed action analysis for this video yet")
        if gap:
            seqs = build_sequences(sequence_inputs(db, job.id), gap)
        else:
            seqs = [{k: getattr(s, k) for k in ("fighter_key", "fighter_id", "track_ids", "sequence_index", "start_time",
                                                 "end_time", "actions", "intervals", "n_actions")} | {"id": s.id}
                    for s in db.scalars(select(ActionSequence).where(ActionSequence.job_id == job.id)
                                        .order_by(ActionSequence.fighter_key, ActionSequence.sequence_index)).all()]
        job_id, model_version = job.id, job.model_version
        assigned = _assigned(db, job.vision_job_id)
        for s in seqs:  # reflect assignments made after analysis
            if not s.get("fighter_id") and len(s["track_ids"]) and assigned.get(s["track_ids"][0]):
                s["fighter_id"] = assigned[s["track_ids"][0]]
    by_fighter: dict[str, dict] = {}
    for s in seqs:
        s["fighter_name"] = names.get(s.get("fighter_id") or "", s["fighter_key"].replace("track:", "Track "))
        s["summary"] = " → ".join(a["label"] for a in s["actions"])
        f = by_fighter.setdefault(s["fighter_key"], {"fighter_key": s["fighter_key"], "fighter_id": s.get("fighter_id"),
                                                     "fighter_name": s["fighter_name"], "sequences": 0, "actions": []})
        f["sequences"] += 1
        f["actions"] += [{**a, "sequence_index": s["sequence_index"]} for a in s["actions"]]
    for f in by_fighter.values():
        f["actions"].sort(key=lambda a: a["start"])
        f["intervals"] = [round(max(0.0, f["actions"][i + 1]["start"] - f["actions"][i]["end"]), 3) for i in range(len(f["actions"]) - 1)]
    return {"video_id": video_id, "source": source, "job_id": job_id, "model_version": model_version,
            "sequences": seqs, "fighters": list(by_fighter.values()),
            "format_note": "Each sequence is a burst of actions by one fighter; a new sequence starts after a pause longer "
                           "than the sequence gap. 'intervals' are seconds from the end of one action to the start of the next. "
                           "Unknown, rejected and unreviewed low-confidence detections are excluded."}


@router.delete("/{video_id}/results", status_code=204, summary="Delete action analyses for a video (annotations are kept)")
def delete_results(video_id: str, db: Session = Depends(get_db), am: ActionJobManager = Depends(get_action_manager)):
    _video(db, video_id)
    if not am.cancel_video(video_id):
        raise HTTPException(409, "Analysis is still stopping; try again")
    try:
        ids = list(db.scalars(select(ActionAnalysisJob.id).where(ActionAnalysisJob.video_id == video_id)).all())
        if ids:
            db.execute(delete(ActionSequence).where(ActionSequence.job_id.in_(ids)))
            db.execute(delete(RecognizedAction).where(RecognizedAction.job_id.in_(ids)))
            db.execute(delete(ActionAnalysisJob).where(ActionAnalysisJob.id.in_(ids)))
        db.commit()
    finally:
        am.forget_cancel(video_id)
    return Response(status_code=204)


@router.get("/{video_id}/annotations", summary="Manual annotations for a video")
def list_annotations(video_id: str, db: Session = Depends(get_db)):
    _video(db, video_id)
    names = _names(db)
    rows = db.scalars(select(ActionAnnotation).where(ActionAnnotation.video_id == video_id).order_by(ActionAnnotation.start_time)).all()
    meta = db.get(VideoAnnotationMeta, video_id)
    return {"video_id": video_id, "items": [ann_out(a, names) for a in rows],
            "counts": dict(Counter(a.category for a in rows)),
            "meta": {"fully_annotated": meta.fully_annotated if meta else False, "split": meta.split if meta else "auto",
                     "split_group": meta.split_group if meta else None, "notes": meta.notes if meta else None}}


@router.post("/{video_id}/annotations", status_code=201, summary="Create a manual annotation")
def create_annotation(video_id: str, body: AnnotationIn, db: Session = Depends(get_db)):
    v = _video(db, video_id)
    if v.duration_seconds and body.start_time > v.duration_seconds + 0.5:
        raise HTTPException(422, "start_time is beyond the end of the video")
    if body.fighter_id and db.get(Fighter, body.fighter_id) is None:
        raise HTTPException(422, "Unknown fighter id")
    if body.fighter_id is None and body.track_id is not None:  # inherit the track's fighter if assigned
        vj = current_vision_job(db, video_id)
        if vj:
            s = db.scalar(select(TrackingSession).where(TrackingSession.job_id == vj.id, TrackingSession.track_id == body.track_id))
            if s and s.assigned_fighter_id:
                body.fighter_id = s.assigned_fighter_id
    a = ActionAnnotation(video_id=video_id, **body.model_dump(), source="manual")
    db.add(a)
    db.commit()
    return ann_out(a, _names(db))


@router.put("/{video_id}/dataset-meta", summary="Mark a video fully annotated / choose its train-test split")
def set_meta(video_id: str, body: MetaIn, db: Session = Depends(get_db)):
    _video(db, video_id)
    m = db.get(VideoAnnotationMeta, video_id) or VideoAnnotationMeta(video_id=video_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(m, k, v)
    db.add(m)
    db.commit()
    return {"video_id": video_id, "fully_annotated": m.fully_annotated, "split": m.split, "split_group": m.split_group, "notes": m.notes}


@router.post("/{video_id}/import", summary="Import annotations from CSV/XLSX (e.g. BoxingVI start_frame,end_frame,punch_class)")
async def import_annotations(
    video_id: str,
    file: UploadFile = File(...),
    units: Literal["frames", "seconds"] = Form("frames"),
    label_map: Literal["none", "boxingvi_orthodox", "boxingvi_southpaw"] = Form("none"),
    fighter_id: str | None = Form(None),
    track_id: int | None = Form(None),
    status_: Literal["verified", "draft", "needs_review"] = Form("verified", alias="status"),
    replace_existing: bool = Form(False),
    db: Session = Depends(get_db),
):
    v = _video(db, video_id)
    if units == "frames" and not v.fps:
        raise HTTPException(409, "Video frame rate unknown — process the video (Phase 1) first or use units=seconds")
    if fighter_id and db.get(Fighter, fighter_id) is None:
        raise HTTPException(422, "Unknown fighter id")
    content = await file.read()
    try:
        df = read_table(content, file.filename or "annotations.csv")
        rows, errors = parse_rows(df, units, v.fps or 30.0, label_map, None)
    except Exception as exc:
        raise HTTPException(422, f"Could not read annotation file: {exc}") from exc
    if replace_existing:
        db.execute(delete(ActionAnnotation).where(ActionAnnotation.video_id == video_id, ActionAnnotation.source == "import"))
    fighters_by_name = {f.name.lower(): f.id for f in db.scalars(select(Fighter)).all()}
    created = 0
    for r in rows:
        fid = fighter_id or (fighters_by_name.get(r.fighter_name.lower()) if r.fighter_name else None)
        db.add(ActionAnnotation(video_id=video_id, category=r.category, start_time=r.start, end_time=r.end,
                                fighter_id=fid, track_id=r.track_id if r.track_id is not None else track_id,
                                confidence=r.confidence, opponent_reaction=r.reaction, status=status_, source="import",
                                notes=f"imported: {r.raw_label}"))
        created += 1
    db.commit()
    return {"video_id": video_id, "created": created, "errors": errors[:50], "error_count": len(errors)}


@router.get("/{video_id}/export", summary="Export recognised actions or sequences (JSON or CSV) for Phase 4")
def export(video_id: str, kind: Literal["actions", "sequences"] = "sequences", format: Literal["json", "csv"] = "json",
           db: Session = Depends(get_db)):
    if kind == "sequences":
        data = sequences(video_id, "model", None, db)
        rows = [{"fighter": s["fighter_name"], "fighter_key": s["fighter_key"], "sequence_index": s["sequence_index"],
                 "start_time": s["start_time"], "end_time": s["end_time"], "n_actions": s["n_actions"],
                 "actions": " > ".join(a["category"] for a in s["actions"]),
                 "intervals": ";".join(str(i) for i in s["intervals"])} for s in data["sequences"]]
    else:
        data = results(video_id, None, None, None, None, 0.0, True, False, db)
        rows = [{k: v for k, v in a.items() if k not in ("threshold",)} for a in data["items"]]
    if format == "json":
        return data
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{video_id}_{kind}.csv"'})

