"""Phase 5 REST API: /api/predictions/...

Every probability returned here is produced by a trained model from stored data; every metric is computed
on held-out predictions. Stored predictions are never edited — evaluations are separate records.
"""
from __future__ import annotations

import io
import re
from types import SimpleNamespace

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.actions.importer import parse_rows, read_table
from app.actions.taxonomy import BY_KEY, label
from app.api.deps import get_db
from app.db.models import Fighter, JobStatus, Video
from app.db.prediction_models import (
    ExternalSequence,
    PredictionEvaluation,
    PredictionModel,
    PredictionResult,
    PredictionTrainingJob,
    PredictionVideoJob,
)
from app.patterns.extract import extract
from app.patterns.settings import get_pattern_settings
from app.patterns.stats import wilson
from app.predictions.dataset import Context, Example, Step
from app.predictions.service import (
    MODEL_CACHE,
    PredictionJobManager,
    PredictionSettings,
    _make_evaluation,
    active_model,
    get_prediction_settings,
    next_sequences,
    predict_examples,
    save_prediction_settings,
)
from app.predictions.training import class_label

router = APIRouter(prefix="/api/predictions", tags=["next-move prediction (phase 5)"])
KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def mgr(request: Request) -> PredictionJobManager:
    return request.app.state.prediction_manager


# ============================================================================ schemas
class ActionIn(BaseModel):
    category: str
    start: float | None = Field(None, description="Seconds (optional; enables timing features)")
    end: float | None = None
    confidence: float = Field(1.0, ge=0, le=1)


class ContextIn(BaseModel):
    opponent_action: str | None = None
    opponent_lag: float | None = Field(None, ge=0, description="Seconds since the opponent's last action started")
    opponent_attacks_3s: int | None = Field(None, ge=0)
    distance: float | None = Field(None, ge=0, description="Distance between fighters in torso lengths")
    closing_speed: float | None = None
    stance: str | None = None
    movement: dict[str, float | None] = Field(default_factory=dict)


class PredictIn(BaseModel):
    history: list[ActionIn | str] = Field(..., min_length=1, max_length=10, description="Oldest -> newest (1-10 actions)")
    context: ContextIn | None = None
    mode: str = Field("immediate", description="immediate | contextual")
    model_id: str | None = None
    save: bool = True


class EvaluateIn(BaseModel):
    prediction_id: str
    actual_action: str


def _steps(history: list[ActionIn | str]) -> list[Step]:
    out = []
    for h in history:
        a = ActionIn(category=h) if isinstance(h, str) else h
        if a.category not in BY_KEY:
            raise HTTPException(422, f"Unknown action: {a.category}")
        out.append(Step(a.category, a.start, a.end, a.confidence))
    return out


def _context(c: ContextIn | None, fighter: Fighter | None) -> Context:
    c = c or ContextIn()
    if c.opponent_action and c.opponent_action not in BY_KEY:
        raise HTTPException(422, f"Unknown opponent action: {c.opponent_action}")
    stance = c.stance or (fighter.stance.value if fighter and fighter.stance and fighter.stance.value != "Unknown" else None)
    return Context(opponent_action=c.opponent_action, opponent_lag=c.opponent_lag, opponent_attacks_3s=c.opponent_attacks_3s,
                   distance=c.distance, closing_speed=c.closing_speed, stance=stance, movement=c.movement)


def _require_model(db: Session, model_id: str | None) -> PredictionModel:
    pm = active_model(db, model_id)
    if pm is None:
        raise HTTPException(409, "No trained prediction model yet. Train models first (POST /api/predictions/models/train).")
    return pm


def _check_fighter(db: Session, key: str) -> Fighter | None:
    if not KEY_RE.match(key):
        raise HTTPException(422, "Invalid fighter key")
    f = db.get(Fighter, key)
    if f is None and not key.startswith(("track-", "ext-")) and not db.scalar(select(ExternalSequence.id).where(ExternalSequence.fighter_key == key).limit(1)):
        raise HTTPException(404, "Fighter not found")
    return f


def _model_json(m: PredictionModel, full: bool = False) -> dict:
    d = {"id": m.id, "version": m.version, "architecture": m.architecture, "name": f"v{m.version}-{m.architecture}",
         "is_active": m.is_active, "selected": m.selected, "experimental": m.experimental, "dataset_version": m.dataset_version,
         "dataset_info": m.dataset_info, "temperature": m.temperature, "path": m.path, "created_at": m.created_at.isoformat()}
    met = m.metrics or {}
    d["summary"] = {k: met.get(k) for k in ("n", "top1_accuracy", "top1_ci95", "top3_accuracy", "macro_f1", "log_loss", "brier", "ece")}
    if full:
        d["metrics"], d["params"] = met, m.params
    return d


def _job_json(j) -> dict:
    d = {"id": j.id, "status": j.status.value, "progress": j.progress, "error_message": j.error_message,
         "created_at": j.created_at.isoformat() if j.created_at else None,
         "finished_at": j.finished_at.isoformat() if j.finished_at else None}
    for k in ("current_stage", "dataset_info", "version", "video_id", "fighter_key", "stats"):
        if hasattr(j, k):
            d[k] = getattr(j, k)
    return d


def _result_json(r: PredictionResult, ev: PredictionEvaluation | None = None) -> dict:
    return {"id": r.id, "fighter_key": r.fighter_key, "video_id": r.video_id, "mode": r.mode, "at_time": r.at_time,
            "sequence": [{**s, "label": label(s["category"])} for s in r.sequence], "context": r.context,
            "top3": r.top3, "predicted_action": r.predicted_action, "predicted_label": class_label(r.predicted_action),
            "predicted_probability": r.predicted_probability, "reliability": r.reliability, "model_version": r.model_version,
            "in_training_data": r.in_training_data, "created_at": r.created_at.isoformat(),
            "evaluation": None if ev is None else {"status": ev.status, "actual_action": ev.actual_action,
                                                   "actual_label": label(ev.actual_action) if ev.actual_action else None,
                                                   "correct_top1": ev.correct_top1, "correct_top3": ev.correct_top3,
                                                   "probability_of_actual": ev.probability_of_actual, "source": ev.actual_source,
                                                   "evaluated_at": ev.evaluated_at.isoformat()}}


# ============================================================================ overview, settings, data
@router.get("", summary="Prediction overview: active model, training state, totals")
def overview(db: Session = Depends(get_db)):
    pm = active_model(db)
    jobs = db.scalars(select(PredictionTrainingJob).order_by(PredictionTrainingJob.created_at.desc()).limit(5)).all()
    n_pred = db.scalar(select(func.count(PredictionResult.id))) or 0
    evs = db.execute(select(PredictionEvaluation.correct_top1, PredictionEvaluation.correct_top3, PredictionEvaluation.in_training_data)
                     .where(PredictionEvaluation.status == "evaluated")).all()
    held = [e for e in evs if not e[2]]
    ext = db.execute(select(ExternalSequence.dataset, func.count(ExternalSequence.id), func.count(func.distinct(ExternalSequence.fight_key)))
                     .group_by(ExternalSequence.dataset)).all()
    return {"active_model": _model_json(pm) if pm else None, "training_jobs": [_job_json(j) for j in jobs],
            "predictions": n_pred, "evaluated": len(evs),
            "held_out": {"n": len(held), "top1": sum(1 for e in held if e[0]), "top3": sum(1 for e in held if e[1])},
            "external_datasets": [{"dataset": d, "actions": n, "fights": f} for d, n, f in ext],
            "settings": get_prediction_settings(db).model_dump()}


@router.get("/settings", response_model=PredictionSettings)
def read_settings(db: Session = Depends(get_db)):
    return get_prediction_settings(db)


@router.put("/settings", response_model=PredictionSettings)
def write_settings(body: PredictionSettings, db: Session = Depends(get_db)):
    return save_prediction_settings(db, body)


@router.get("/fighters", summary="Fighters available for prediction")
def fighters(db: Session = Depends(get_db)):
    ex = extract(db, get_pattern_settings(db))
    keys = {}
    for o in ex.obs:
        d = keys.setdefault(o.fighter_key, {"fighter_key": o.fighter_key, "name": ex.names.get(o.fighter_key, o.fighter_key),
                                            "source": "app", "actions": 0})
        d["actions"] += 1
    for fk, n in db.execute(select(ExternalSequence.fighter_key, func.count(ExternalSequence.id)).group_by(ExternalSequence.fighter_key)).all():
        keys[fk] = {"fighter_key": fk, "name": fk, "source": "external", "actions": n}
    for f in db.scalars(select(Fighter)).all():
        keys.setdefault(f.id, {"fighter_key": f.id, "name": f.name, "source": "app", "actions": 0})
    return sorted(keys.values(), key=lambda d: (d["source"] != "app", -d["actions"]))


@router.post("/datasets/import", summary="Import real action sequences without video (e.g. BoxingVI annotation sheets)")
async def import_sequences(file: UploadFile = File(...), dataset: str = Form("external"), fighter_key: str = Form(...),
                           fight_key: str = Form(...), units: str = Form("frames"), fps: float = Form(30.0),
                           label_map: str = Form("boxingvi_orthodox"), stance: str | None = Form(None),
                           replace: bool = Form(True), db: Session = Depends(get_db)):
    for v, n in ((fighter_key, "fighter_key"), (fight_key, "fight_key"), (dataset, "dataset")):
        if not KEY_RE.match(v):
            raise HTTPException(422, f"{n} may contain letters, digits, '-' and '_' only")
    content = await file.read()
    if label_map.startswith("boxingvi") and (file.filename or "").lower().endswith((".xlsx", ".xls")):
        rows, errors = parse_boxingvi_sheet(content, units, fps, label_map)
        return _store_external(db, rows, errors, dataset, fighter_key, fight_key, stance, replace)
    try:
        df = read_table(content, file.filename or "x.csv")
        if all(re.fullmatch(r"-?\d+(\.\d+)?", str(c).strip()) for c in list(df.columns)[:2]):  # header-less sheet
            df = pd.read_excel(io.BytesIO(content), header=None) if (file.filename or "").lower().endswith((".xlsx", ".xls")) \
                else pd.read_csv(io.BytesIO(content), header=None, sep=None, engine="python")
        rows, errors = parse_rows(df, units, fps, label_map, None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _store_external(db, rows, errors, dataset, fighter_key, fight_key, stance, replace)


BOXINGVI_CLASSES = {"jab", "cross", "lead hook", "rear hook", "lead uppercut", "rear uppercut"}


def parse_boxingvi_sheet(content: bytes, units: str, fps: float, label_map: str):
    """BoxingVI sheets differ in layout (header names, blank columns, extra helper columns). Each data row holds a
    start frame, an end frame and one of the six punch classes: take the first two numbers and the class text."""
    from app.actions.taxonomy import BOXINGVI_MAP_ORTHODOX, BOXINGVI_MAP_SOUTHPAW

    mapping = BOXINGVI_MAP_SOUTHPAW if label_map == "boxingvi_southpaw" else BOXINGVI_MAP_ORTHODOX
    rows, errors = [], []
    for sheet in pd.read_excel(io.BytesIO(content), header=None, sheet_name=None).values():
        for i, r in enumerate(sheet.itertuples(index=False)):
            nums = [float(v) for v in r if isinstance(v, (int, float)) and not pd.isna(v)]
            lab = next((str(v).strip().lower() for v in r if isinstance(v, str) and str(v).strip().lower() in BOXINGVI_CLASSES), None)
            if lab is None:
                continue
            if len(nums) < 2 or nums[1] < nums[0]:
                errors.append(f"row {i + 1}: invalid start/end for '{lab}'")
                continue
            div = fps if units == "frames" else 1.0
            rows.append(SimpleNamespace(start=nums[0] / div, end=nums[1] / div, category=mapping[lab], raw_label=lab))
    rows.sort(key=lambda r: r.start)
    return rows, errors


def _store_external(db, rows, errors, dataset, fighter_key, fight_key, stance, replace):
    if replace:
        db.execute(delete(ExternalSequence).where(ExternalSequence.dataset == dataset, ExternalSequence.fight_key == fight_key,
                                                  ExternalSequence.fighter_key == fighter_key))
    for r in rows:
        db.add(ExternalSequence(dataset=dataset, fighter_key=fighter_key, fight_key=fight_key, category=r.category,
                                original_label=r.raw_label, start=r.start, end=r.end, stance=stance))
    db.commit()
    return {"imported": len(rows), "errors": errors[:50], "error_count": len(errors), "dataset": dataset,
            "fighter_key": fighter_key, "fight_key": fight_key}


@router.delete("/datasets/{dataset}")
def delete_dataset(dataset: str, db: Session = Depends(get_db)):
    n = db.execute(delete(ExternalSequence).where(ExternalSequence.dataset == dataset)).rowcount
    db.commit()
    return {"deleted": n}


# ============================================================================ models
@router.post("/models/train", status_code=202, summary="Train and evaluate all models in the background")
def train(db: Session = Depends(get_db), m: PredictionJobManager = Depends(mgr)):
    return _job_json(m.train(db))


@router.get("/models/jobs")
def training_jobs(db: Session = Depends(get_db)):
    return [_job_json(j) for j in db.scalars(select(PredictionTrainingJob).order_by(PredictionTrainingJob.created_at.desc()).limit(20)).all()]


@router.get("/models/versions")
def versions(db: Session = Depends(get_db)):
    return [_model_json(m) for m in db.scalars(select(PredictionModel).order_by(PredictionModel.version.desc(), PredictionModel.architecture)).all()]


@router.get("/models/evaluation", summary="Full evaluation report of a training run (latest by default)")
def evaluation(job_id: str | None = None, db: Session = Depends(get_db)):
    q = select(PredictionTrainingJob).where(PredictionTrainingJob.status == JobStatus.completed)
    j = db.get(PredictionTrainingJob, job_id) if job_id else db.scalar(q.order_by(PredictionTrainingJob.finished_at.desc()).limit(1))
    if j is None:
        raise HTTPException(404, "No completed training run")
    return {"job": _job_json(j), "report": j.result}


@router.post("/models/{model_id}/activate")
def activate(model_id: str, db: Session = Depends(get_db)):
    m = db.get(PredictionModel, model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    for o in db.scalars(select(PredictionModel).where(PredictionModel.is_active.is_(True))).all():
        o.is_active = False
    m.is_active = True
    db.commit()
    MODEL_CACHE.clear()
    return _model_json(m)


@router.get("/models/{model_id}")
def model_detail(model_id: str, db: Session = Depends(get_db)):
    m = db.get(PredictionModel, model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    return _model_json(m, full=True)


# ============================================================================ video
@router.post("/video/{video_id}/predict", status_code=202, summary="Predict the next move after every recognised action in a video")
def video_predict(video_id: str, fighter_key: str | None = None, model_id: str | None = None, db: Session = Depends(get_db),
                  m: PredictionJobManager = Depends(mgr)):
    if db.get(Video, video_id) is None:
        raise HTTPException(404, "Video not found")
    _require_model(db, model_id)
    return _job_json(m.video(db, video_id, fighter_key, model_id))


@router.get("/video/{video_id}/timeline", summary="Actual actions and stored predictions of the latest video prediction run")
def video_timeline(video_id: str, fighter_key: str | None = None, db: Session = Depends(get_db)):
    job = db.scalar(select(PredictionVideoJob).where(PredictionVideoJob.video_id == video_id).order_by(PredictionVideoJob.created_at.desc()).limit(1))
    if job is None:
        return {"job": None, "predictions": []}
    q = select(PredictionResult, PredictionEvaluation).outerjoin(PredictionEvaluation, PredictionEvaluation.prediction_id == PredictionResult.id) \
        .where(PredictionResult.video_job_id == job.id)
    if fighter_key:
        q = q.where(PredictionResult.fighter_key == fighter_key)
    rows = db.execute(q.order_by(PredictionResult.at_time)).all()
    return {"job": _job_json(job), "predictions": [_result_json(r, e) for r, e in rows]}


# ============================================================================ per fighter
def _example(fk: str, steps: list[Step], ctx: Context) -> Example:
    t = steps[-1].end if steps[-1].end is not None else 0.0
    return Example(fighter_key=fk, fight_key="__request__", video_id="__request__", fight_date=None, history=steps, context=ctx,
                   target=None, t=t)


@router.post("/{fighter_id}/predict", summary="Predict the next action from a recent action history and context")
def predict(fighter_id: str, body: PredictIn, db: Session = Depends(get_db)):
    f = _check_fighter(db, fighter_id)
    pm = _require_model(db, body.model_id)
    if body.mode not in ("immediate", "contextual"):
        raise HTTPException(422, "mode must be immediate or contextual")
    e = _example(fighter_id, _steps(body.history), _context(body.context, f))
    r = predict_examples(db, pm, [e], contextual=body.mode == "contextual")[0]
    r["fighter_key"], r["mode"] = fighter_id, body.mode
    r["sequence"] = [{"category": s.category, "label": label(s.category)} for s in e.history]
    if body.save:
        pr = PredictionResult(fighter_key=fighter_id, mode="manual", sequence=[{"category": s.category, "start": s.start, "end": s.end} for s in e.history],
                              context=body.context.model_dump() if body.context else None,
                              distribution={d["action"]: d["probability"] for d in r["distribution"]}, top3=r["top3"],
                              predicted_action=r["predicted_action"], predicted_probability=r["predicted_probability"],
                              reliability=r["reliability"], model_id=pm.id, model_version=r["model"]["version"], in_training_data=False)
        db.add(pr)
        db.commit()
        r["prediction_id"] = pr.id
    return r


@router.get("/{fighter_id}/next-sequence", summary="Possible next 2-3 actions (beam search), not guaranteed outcomes")
def next_sequence(fighter_id: str, history: list[str] = Query(..., description="Oldest -> newest, e.g. history=left_jab&history=right_cross"),
                  depth: int = Query(3, ge=2, le=3), width: int | None = Query(None, ge=1, le=20),
                  opponent_action: str | None = None, distance: float | None = Query(None, ge=0),
                  model_id: str | None = None, db: Session = Depends(get_db)):
    f = _check_fighter(db, fighter_id)
    pm = _require_model(db, model_id)
    e = _example(fighter_id, _steps(history), _context(ContextIn(opponent_action=opponent_action, distance=distance), f))
    w = width or get_prediction_settings(db).beam_width
    seqs = next_sequences(db, pm, e, depth, w)
    return {"fighter_key": fighter_id, "history": [label(s.category) for s in e.history], "depth": depth, "sequences": seqs,
            "model_version": f"v{pm.version}-{pm.architecture}",
            "note": "Joint probabilities of possible future sequences from repeated one-step predictions. Hypothetical actions use "
                    "this fighter's typical timing and the current context; they are possibilities, not outcomes."}


@router.post("/{fighter_id}/evaluate", summary="Record the actual next action for a stored prediction")
def evaluate_prediction(fighter_id: str, body: EvaluateIn, db: Session = Depends(get_db)):
    pr = db.get(PredictionResult, body.prediction_id)
    if pr is None or pr.fighter_key != fighter_id:
        raise HTTPException(404, "Prediction not found for this fighter")
    if body.actual_action not in BY_KEY:
        raise HTTPException(422, f"Unknown action: {body.actual_action}")
    if db.scalar(select(PredictionEvaluation).where(PredictionEvaluation.prediction_id == pr.id)):
        raise HTTPException(409, "This prediction has already been evaluated; evaluations are not changed afterwards")
    ev = _make_evaluation(pr, body.actual_action, "user", db.get(PredictionModel, pr.model_id))
    db.add(ev)
    db.commit()
    return _result_json(pr, ev)


@router.get("/{fighter_id}/history")
def history(fighter_id: str, limit: int = Query(200, ge=1, le=2000), db: Session = Depends(get_db)):
    rows = db.execute(select(PredictionResult, PredictionEvaluation).outerjoin(PredictionEvaluation, PredictionEvaluation.prediction_id == PredictionResult.id)
                      .where(PredictionResult.fighter_key == fighter_id).order_by(PredictionResult.created_at.desc(), PredictionResult.at_time.desc())
                      .limit(limit)).all()
    return [_result_json(r, e) for r, e in rows]


@router.get("/{fighter_id}/performance", summary="Accuracy of stored predictions against the actual next actions")
def performance(fighter_id: str, db: Session = Depends(get_db)):
    evs = db.execute(select(PredictionEvaluation, PredictionResult.model_version, PredictionResult.created_at, PredictionResult.at_time)
                     .join(PredictionResult, PredictionResult.id == PredictionEvaluation.prediction_id)
                     .where(PredictionEvaluation.fighter_key == fighter_id, PredictionEvaluation.status == "evaluated")
                     .order_by(PredictionResult.created_at, PredictionResult.at_time)).all()

    def agg(items):
        n = len(items)
        t1 = sum(1 for e in items if e.correct_top1)
        t3 = sum(1 for e in items if e.correct_top3)
        return {"n": n, "top1": t1, "top3": t3, "top1_rate": round(t1 / n, 4) if n else None, "top1_ci95": wilson(t1, n),
                "top3_rate": round(t3 / n, 4) if n else None, "top3_ci95": wilson(t3, n)}

    all_e = [e for e, *_ in evs]
    held = [e for e in all_e if not e.in_training_data]
    over, c1, c3 = [], 0, 0
    for i, (e, mv, _, _at) in enumerate([x for x in evs if not x[0].in_training_data]):
        c1 += bool(e.correct_top1)
        c3 += bool(e.correct_top3)
        over.append({"i": i + 1, "top1_rate": round(c1 / (i + 1), 4), "top3_rate": round(c3 / (i + 1), 4), "model": mv})
    by_model = {}
    for e, mv, *_ in evs:
        by_model.setdefault(mv, []).append(e)
    return {"fighter_key": fighter_id, "held_out": agg(held), "in_training_data": agg([e for e in all_e if e.in_training_data]),
            "by_model": {k: agg(v) for k, v in by_model.items()}, "over_time": over,
            "note": "Held-out = the model had not seen that fight during training. Predictions on fights the model trained on are "
                    "reported separately and are optimistic."}


@router.get("/{fighter_id}/model-status")
def model_status(fighter_id: str, db: Session = Depends(get_db)):
    pm = active_model(db)
    if pm is None:
        return {"fighter_key": fighter_id, "ready": False, "reason": "No trained model"}
    _, _, bundle, _ = MODEL_CACHE.get(pm, "cpu")
    n = bundle["fighter_examples"].get(fighter_id, 0)
    return {"fighter_key": fighter_id, "ready": True, "model": _model_json(pm), "fighter_training_examples": n,
            "known_fighter": n > 0, "classes": [{"action": c, "label": class_label(c)} for c in bundle["classes"]],
            "history_length": bundle["history_length"],
            "note": None if n else "This fighter is not in the training data; predictions use patterns of all fighters."}
