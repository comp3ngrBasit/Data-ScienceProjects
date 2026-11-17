"""Phase 6 API: dashboard, fighter intelligence, unified pipeline, model evaluation & management, complete video analysis.

All endpoints aggregate stored results of Phases 1-5; nothing is simulated. Existing phase endpoints are unchanged.
"""
from __future__ import annotations

import statistics
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.actions.job_manager import ActionConflict, current_action_job
from app.actions.job_manager import active_model as active_action_model
from app.actions.settings import get_action_settings
from app.actions.taxonomy import label
from app.api.deps import get_db
from app.db.action_models import ActionModelVersion, RecognizedAction
from app.db.models import Fighter, JobStatus, Video
from app.db.platform_models import ModelDeployment, PipelineRun
from app.db.prediction_models import PredictionEvaluation, PredictionModel, PredictionResult, PredictionTrainingJob, PredictionVideoJob
from app.patterns import engine as E
from app.patterns.extract import apply_filters, extract
from app.patterns.settings import get_pattern_settings
from app.platform import analytics as A
from app.platform.pipeline import PipelineManager, run_json
from app.platform.validation import validate_action_model, validate_prediction_model
from app.predictions.dataset import Context, Example, Step
from app.predictions.service import MODEL_CACHE, active_model, get_prediction_settings, next_sequences, predict_examples
from app.predictions.training import class_label
from app.vision.job_manager import current_job as current_vision_job

dashboard = APIRouter(prefix="/api/dashboard", tags=["dashboard (phase 6)"])
fighters = APIRouter(prefix="/api/fighters", tags=["fighter intelligence (phase 6)"])
models = APIRouter(prefix="/api/models", tags=["model evaluation & management (phase 6)"])
videos = APIRouter(prefix="/api/videos", tags=["video intelligence (phase 6)"])
pipeline = APIRouter(prefix="/api/pipeline", tags=["unified pipeline (phase 6)"])


def pipeline_manager(request: Request) -> PipelineManager:
    return request.app.state.pipeline_manager


# ============================================================================ dashboard
@dashboard.get("/overview", summary="Totals across all phases, model status and recent pipeline runs")
def dashboard_overview(db: Session = Depends(get_db)):
    return A.overview(db)


@dashboard.get("/fighters", summary="One row per fighter (profiles and unassigned tracks with actions)")
def dashboard_fighters(db: Session = Depends(get_db)):
    return A.fighters_table(db)


@dashboard.get("/analytics", summary="Pattern analytics over all fighters (or one): combinations, transitions, sequences, opponents")
def dashboard_analytics(fighter: str | None = None, db: Session = Depends(get_db)):
    return A.global_analytics(db, fighter)


# ============================================================================ fighter intelligence
def _fighter_key(db: Session, fighter_id: str):
    if db.get(Fighter, fighter_id) is None and not fighter_id.startswith(("track-", "boxingvi-", "ext-")):
        raise HTTPException(404, "Fighter not found")


@fighters.get("/{fighter_id}/intelligence", summary="Fighter profile: observed behaviour, statistical patterns, videos, movement")
def fighter_intelligence(fighter_id: str, video: list[str] | None = Query(None, description="Restrict to these videos"),
                         db: Session = Depends(get_db)):
    _fighter_key(db, fighter_id)
    return A.fighter_intelligence(db, fighter_id, video)


@fighters.get("/{fighter_id}/patterns", summary="Phase 4 patterns of the fighter (combinations, transitions, tendencies, opponents)")
def fighter_patterns(fighter_id: str, db: Session = Depends(get_db)):
    _fighter_key(db, fighter_id)
    from app.patterns.api import Filters, _analysis

    r = _analysis(db, fighter_id, Filters())
    return {k: r[k] for k in ("fighter_key", "name", "summary", "warnings", "meta", "combinations", "sequential_patterns", "tendencies",
                              "opponent_analysis", "state_labels")} | {"markov": r["markov"]["1"]}


def latest_sequence(db: Session, fighter_id: str, history: int = 5, video_id: str | None = None, at: float | None = None) -> Example | None:
    """The fighter's latest recognised actions (optionally in one video, up to a timestamp) as a prediction input."""
    ps = get_prediction_settings(db)
    from app.predictions.service import assemble

    examples, _ = assemble(db, ps, include_last=True, video_id=video_id)
    mine = [e for e in examples if e.fighter_key == fighter_id and (at is None or e.t <= at + 1e-6)]
    if not mine:
        return None
    e = max(mine, key=lambda x: (x.fight_date or "", x.video_id if video_id else "", x.t))
    e.history = e.history[-history:]
    return e


@fighters.get("/{fighter_id}/predictions", summary="Next-move prediction from the fighter's latest recognised actions")
def fighter_predictions(fighter_id: str, video_id: str | None = None, at: float | None = Query(None, ge=0, description="Video time (s)"),
                        history: int = Query(5, ge=2, le=10), db: Session = Depends(get_db)):
    _fighter_key(db, fighter_id)
    pm = active_model(db)
    hist = db.execute(select(PredictionResult, PredictionEvaluation).outerjoin(PredictionEvaluation, PredictionEvaluation.prediction_id == PredictionResult.id)
                      .where(PredictionResult.fighter_key == fighter_id).order_by(PredictionResult.created_at.desc()).limit(50)).all()
    evald = [e for _, e in hist if e and e.status == "evaluated" and not e.in_training_data]
    perf = {"held_out_evaluated": len(evald), "top1": sum(1 for e in evald if e.correct_top1), "top3": sum(1 for e in evald if e.correct_top3)}
    if pm is None:
        return {"fighter_key": fighter_id, "available": False, "reason": "No active next-move model. Train one in Model Management.",
                "performance": perf}
    e = latest_sequence(db, fighter_id, history, video_id, at)
    if e is None:
        return {"fighter_key": fighter_id, "available": False, "performance": perf,
                "reason": "No recognised actions for this fighter" + (" before this time in this video." if at is not None else ".")}
    try:
        r = predict_examples(db, pm, [e], contextual=True)[0]
        seqs = next_sequences(db, pm, e, 2, get_prediction_settings(db).beam_width)
    except Exception as exc:  # failed model inference is reported, never replaced by made-up output
        raise HTTPException(503, f"Model inference failed ({type(exc).__name__}: {exc}). Retrain or activate another model version.") from exc
    _, _, bundle, _ = MODEL_CACHE.get(pm, "cpu")
    prof, glob = bundle["profile"].get(fighter_id) or {}, bundle["profile"].get("*") or {}
    gap = ((prof.get("interval") or glob.get("interval")) or {}).get("median")  # fighter's own timing, else all fighters'
    dur = ((prof.get("duration") or glob.get("duration")) or {}).get("median")
    last = e.history[-1]
    timeline = [{"kind": "detected", "time": s.start, "end": s.end, "category": s.category, "label": label(s.category)} for s in e.history]
    if gap is not None and last.end is not None:
        t0 = last.end + max(0.0, gap)
        for i, d in enumerate(r["top3"]):
            timeline.append({"kind": "predicted" if i == 0 else "alternative", "time": round(t0, 3), "end": round(t0 + (dur or 0.3), 3),
                             "category": d["action"], "label": d["label"], "probability": d["probability"],
                             "time_basis": f"last action end + median gap ({gap:.2f} s; this fighter's if enough data, else all fighters') — an estimate, not an observation"})
    return {"fighter_key": fighter_id, "available": True, "video_id": e.video_id if not e.video_id.startswith("ext-") else None,
            "at": e.t, "context": {"opponent_action": e.context.opponent_action, "opponent_label": label(e.context.opponent_action) if e.context.opponent_action else None,
                                   "opponent_lag": e.context.opponent_lag, "distance": e.context.distance, "stance": e.context.stance,
                                   "movement": e.context.movement},
            "previous_actions": [{"category": s.category, "label": label(s.category), "start": s.start, "end": s.end} for s in e.history],
            "prediction": r, "next_sequences": seqs, "timeline": timeline, "performance": perf,
            "disclaimer": "Model predictions of what may happen next — not observed or confirmed actions."}


# ============================================================================ pipeline
@pipeline.post("/videos/{video_id}/run", status_code=202, summary="Run the full analysis pipeline for a video (reuses up-to-date stages)")
def run_pipeline(video_id: str, force: bool = False, db: Session = Depends(get_db), pm: PipelineManager = Depends(pipeline_manager)):
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "Video not found")
    run, created = pm.create(db, v, force)
    return {**run_json(run), "created": created}


@pipeline.get("/videos/{video_id}", summary="Pipeline runs of a video (latest first)")
def pipeline_runs(video_id: str, db: Session = Depends(get_db)):
    return [run_json(r) for r in db.scalars(select(PipelineRun).where(PipelineRun.video_id == video_id).order_by(PipelineRun.created_at.desc()).limit(10)).all()]


@pipeline.get("/runs", summary="Recent pipeline runs")
def all_runs(db: Session = Depends(get_db)):
    return [run_json(r) for r in db.scalars(select(PipelineRun).order_by(PipelineRun.created_at.desc()).limit(30)).all()]


# ============================================================================ complete video analysis
@videos.get("/{video_id}/complete-analysis", summary="Everything known about a video from all phases")
def complete_analysis(video_id: str, db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "Video not found")
    vj = current_vision_job(db, video_id)
    aj = current_action_job(db, video_id)
    names = {f.id: f.name for f in db.scalars(select(Fighter)).all()}
    tracks = []
    if vj:
        from app.db.vision_models import TrackingSession

        for t in db.scalars(select(TrackingSession).where(TrackingSession.job_id == vj.id).order_by(TrackingSession.track_id)).all():
            tracks.append({"track_id": t.track_id, "fighter_id": t.assigned_fighter_id, "fighter": names.get(t.assigned_fighter_id),
                           "role": t.role, "first": t.first_timestamp, "last": t.last_timestamp, "coverage": t.coverage,
                           "interruptions": t.interruptions})
    actions = []
    if aj:
        for r in db.scalars(select(RecognizedAction).where(RecognizedAction.job_id == aj.id).order_by(RecognizedAction.start_time)).all():
            fk = r.fighter_id or next((t["fighter_id"] for t in tracks if t["track_id"] == r.track_id and t["fighter_id"]), None)
            cat = r.corrected_category or r.category
            actions.append({"id": r.id, "category": cat, "label": label(cat), "model_category": r.category, "start": r.start_time,
                            "end": r.end_time, "confidence": r.confidence, "track_id": r.track_id, "fighter_id": fk,
                            "fighter": names.get(fk) if fk else (f"Track {r.track_id}" if r.track_id is not None else None),
                            "review_status": r.review_status, "low_confidence": r.low_confidence})
    # combination timeline: recurring n-grams of each fighter, with their occurrences in this video
    ps = get_pattern_settings(db)
    cfg = ps.engine_config()
    ex = extract(db, ps)
    p_all = E.prepare(apply_filters(ex), cfg)
    combos = []
    for fk in {o.fighter_key for o in p_all.obs if o.video_id == video_id}:
        seqs = E.fighter_sequences(p_all, fk)
        known = {tuple(c["sequence"]): c for c in E.combinations(seqs, cfg)}
        for s in seqs:
            if s[0].video_id != video_id:
                continue
            for n in range(min(cfg.max_n, len(s)), 1, -1):
                for i in range(len(s) - n + 1):
                    key = tuple(o.category for o in s[i:i + n])
                    if key in known:
                        c = known[key]
                        combos.append({"fighter_key": fk, "fighter": ex.names.get(fk, fk), "sequence": list(key), "labels": [label(k) for k in key],
                                       "start": round(s[i].start, 3), "end": round(s[i + n - 1].end, 3),
                                       "occurrences_total": c["occurrences"], "fights": c["distinct_fights"], "classification": c["classification"]})
    combos.sort(key=lambda c: (c["start"], -len(c["sequence"])))
    pj = db.scalar(select(PredictionVideoJob).where(PredictionVideoJob.video_id == video_id).order_by(PredictionVideoJob.created_at.desc()).limit(1))
    preds = []
    if pj:
        for r, e in db.execute(select(PredictionResult, PredictionEvaluation).outerjoin(PredictionEvaluation, PredictionEvaluation.prediction_id == PredictionResult.id)
                               .where(PredictionResult.video_job_id == pj.id).order_by(PredictionResult.at_time)).all():
            preds.append({"id": r.id, "fighter_key": r.fighter_key, "at_time": r.at_time, "top3": r.top3, "sequence": [{**x, "label": label(x["category"])} for x in r.sequence],
                          "reliability": {"level": r.reliability.get("level"), "warnings": r.reliability.get("warnings", []),
                                          "supporting_observations": r.reliability.get("supporting_observations")},
                          "model_version": r.model_version, "in_training_data": r.in_training_data,
                          "evaluation": None if e is None else {"status": e.status, "actual": e.actual_action,
                                                                "actual_label": label(e.actual_action) if e.actual_action else None,
                                                                "top1": e.correct_top1, "top3": e.correct_top3}})
    run = db.scalar(select(PipelineRun).where(PipelineRun.video_id == video_id).order_by(PipelineRun.created_at.desc()).limit(1))
    from app.services.video_service import to_detail

    return {"video": to_detail(v).model_dump(mode="json"),
            "stages": {"processing": v.status.value, "vision": vj.status.value if vj else None,
                       "actions": aj.status.value if aj else None, "action_model": aj.model_version if aj else None,
                       "predictions": pj.status.value if pj else None},
            "annotated_video_url": f"/api/vision/{video_id}/annotated-video?v={vj.id}" if vj and vj.annotated_video_path else None,
            "fps": (vj.stats or {}).get("fps") if vj else v.fps, "tracks": tracks, "actions": actions,
            "combinations": combos, "predictions": preds, "pipeline": run_json(run) if run else None,
            "fighters": sorted({a["fighter_id"] for a in actions if a["fighter_id"]} | {t["fighter_id"] for t in tracks if t["fighter_id"]})}


# ============================================================================ model evaluation & management
@models.get("/evaluation", summary="Evaluation of the action-recognition, pattern and next-move models (held-out data only)")
def evaluation(db: Session = Depends(get_db)):
    am = active_action_model(db)
    action = None
    if am:
        met = am.metrics or {}
        info = am.training_info or {}
        action = {"model": A.action_model_json(am), "metrics": met, "dataset": info.get("dataset"),
                  "samples": {"final_train_samples": info.get("final_train_samples"), "evaluated": met.get("n_evaluated"),
                              "evaluation": (met.get("evaluation") or {})},
                  "insufficient": (met.get("n_evaluated") or 0) < 200,
                  "note": "Metrics are from held-out videos / fight groups (Phase 3 split by video)."}
    pj = db.scalar(select(PredictionTrainingJob).where(PredictionTrainingJob.status == JobStatus.completed)
                   .order_by(PredictionTrainingJob.finished_at.desc()).limit(1))
    nm = None
    if pj and pj.result:
        rep = pj.result
        samples = {}
        for name, proto in rep.get("protocols", {}).items():
            folds = [f for f in proto.get("folds", []) if "test_examples" in f]
            if folds:
                samples[name] = {"folds": len(folds), "test_examples": sum(f["test_examples"] for f in folds),
                                 "mean_train_examples": round(statistics.fmean(f["train_examples"] for f in folds), 1),
                                 "mean_validation_examples": round(statistics.fmean(f.get("validation_examples", 0) for f in folds), 1)}
        nm = {"report": rep, "samples": samples, "training_job_id": pj.id, "finished_at": pj.finished_at.isoformat() if pj.finished_at else None,
              "insufficient": rep.get("status") != "ok" or rep.get("experimental", True)}
    return {"action_recognition": action, "pattern_recognition": A.pattern_evaluation(db), "next_move": nm,
            "rules": {"splits": "Action recognition: held-out videos. Next move: fight-level CV, unseen-fighter CV, temporal hold-out. "
                                "Patterns: detections vs verified annotations, split-half fights, leave-one-fight-out.",
                      "never": "No metric is computed on a model's own training data."}}


@models.get("/versions", summary="All model versions of both kinds with deployment status and history")
def versions(db: Session = Depends(get_db)):
    return {"action": [A.action_model_json(m) for m in db.scalars(select(ActionModelVersion).order_by(ActionModelVersion.created_at.desc())).all()],
            "prediction": [A.prediction_model_json(m) for m in db.scalars(select(PredictionModel).order_by(PredictionModel.version.desc(), PredictionModel.architecture)).all()],
            "deployments": A.deployments(db)}


class TrainIn(BaseModel):
    kind: Literal["action", "prediction"]


@models.post("/train", status_code=202, summary="Start training (background). New models go live only if they pass validation.")
def train(body: TrainIn, request: Request, db: Session = Depends(get_db)):
    if body.kind == "prediction":
        j = request.app.state.prediction_manager.train(db)
        return {"kind": "prediction", "job_id": j.id, "status": j.status.value}
    s = get_action_settings(db)
    cfg = {"algorithms": s.algorithms, "activate_best": True, "threshold": s.confidence_threshold, "min_examples": s.min_examples,
           "min_groups": s.min_groups, "augment": s.augment, "negatives_per_positive": s.negatives_per_positive, "seq_len": s.seq_len,
           "context": s.context, "include_draft_annotations": s.include_draft_annotations, "epochs": s.epochs, "device": s.device}
    try:
        j = request.app.state.action_manager.create_training(db, cfg)
    except ActionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"kind": "action", "job_id": j.id, "status": j.status.value}


class ActivateIn(BaseModel):
    kind: Literal["action", "prediction"]
    model_id: str
    force: bool = False
    reason: Literal["manual", "rollback"] = "manual"


@models.post("/activate", summary="Activate a model version (validation checks apply unless force=true; rollback = activate an older version)")
def activate(body: ActivateIn, db: Session = Depends(get_db)):
    if body.kind == "action":
        m = db.get(ActionModelVersion, body.model_id)
        if m is None or m.status != "ready":
            raise HTTPException(404, "Model not found or not ready")
        cur = active_action_model(db)
        val = validate_action_model(m.metrics, None)  # absolute checks; comparison with the active model is advisory here
        if not val["passed"] and not body.force:
            raise HTTPException(409, {"message": "Model failed validation checks; pass force=true to activate anyway", "validation": val})
        db.execute(update(ActionModelVersion).values(is_active=False))
        m.is_active = True
        prev = cur.version if cur else None
    else:
        m = db.get(PredictionModel, body.model_id)
        if m is None:
            raise HTTPException(404, "Model not found")
        cur = active_model(db)
        job = db.scalar(select(PredictionTrainingJob).where(PredictionTrainingJob.id == m.training_job_id))
        floor = ((job.result or {}).get("protocols", {}).get("fight_cv", {}).get("models", {}).get("class_frequency")) if job else None
        val = validate_prediction_model(m.metrics, floor, None)
        if not val["passed"] and not body.force:
            raise HTTPException(409, {"message": "Model failed validation checks; pass force=true to activate anyway", "validation": val})
        db.execute(update(PredictionModel).values(is_active=False))
        m.is_active = True
        prev = cur.id if cur else None
        MODEL_CACHE.clear()
    db.add(ModelDeployment(kind=body.kind, model_id=body.model_id, previous_model_id=prev, reason=body.reason,
                           forced=body.force and not val["passed"], validation=val))
    db.commit()
    return {"kind": body.kind, "active": body.model_id, "previous": prev, "validation": val}


@models.get("/deployments", summary="History of active-model changes")
def deployment_history(kind: str | None = None, db: Session = Depends(get_db)):
    return A.deployments(db, kind)


@models.get("/datasets", summary="What the next training run of each kind would use")
def datasets(db: Session = Depends(get_db)):
    from app.predictions.service import assemble

    ps = get_prediction_settings(db)
    ex_, _ = assemble(db, ps)
    ex_ = [e for e in ex_ if e.target]
    from collections import Counter

    return {"prediction": {"examples": len(ex_), "fights": len({e.fight_key for e in ex_}), "fighters": len({e.fighter_key for e in ex_}),
                           "sources": dict(Counter(e.source.split(":")[0] if e.source.startswith("external") else e.source for e in ex_)),
                           "targets": {class_label(k): v for k, v in Counter(e.target for e in ex_).most_common()}},
            "action": {"note": "Built from verified annotations at training time — see MMA Action Recognition → Models → dataset preview "
                               "(GET /api/actions/dataset)."}}


__all__ = ["dashboard", "fighters", "models", "videos", "pipeline", "Context", "Step"]
