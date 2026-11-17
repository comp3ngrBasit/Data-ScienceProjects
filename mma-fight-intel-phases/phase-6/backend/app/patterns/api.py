"""Phase 4 REST API: /api/patterns/...

Every number is computed from stored Phase 3 actions / verified annotations. Responses served from a stored
snapshot say so (`meta.mode = "snapshot"`); filtered requests and fighters whose data changed since the last
analysis are computed on the fly (`meta.mode = "live"`, `meta.stale = true` when a snapshot exists but is out of date).
"""
from __future__ import annotations

import re
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.actions.taxonomy import BY_KEY, label
from app.api.deps import get_db
from app.db.models import Fighter, JobStatus, Video
from app.db.pattern_models import FightInfo, PatternAnalysisJob
from app.patterns import engine as E
from app.patterns.extract import apply_filters, extract
from app.patterns.service import (
    DEFINITIONS,
    PatternJobManager,
    analyze_fighter,
    fighter_index,
    fighter_version,
    latest_snapshot,
)
from app.patterns.settings import PatternSettings, get_pattern_settings, save_pattern_settings

router = APIRouter(prefix="/api/patterns", tags=["fighter patterns (phase 4)"])
KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def get_pattern_manager(request: Request) -> PatternJobManager:
    return request.app.state.pattern_manager


class Filters(BaseModel):
    fight: list[str] | None = None
    video: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    min_confidence: float | None = None

    @property
    def active(self) -> bool:
        return bool(self.fight or self.video or self.date_from or self.date_to or self.min_confidence is not None)


def filters(fight: list[str] | None = Query(None, description="Restrict to these fight keys"),
            video: list[str] | None = Query(None, description="Restrict to these video ids"),
            date_from: str | None = Query(None, description="YYYY-MM-DD (fights without a date are excluded)"),
            date_to: str | None = Query(None, description="YYYY-MM-DD"),
            min_confidence: float | None = Query(None, ge=0, le=1, description="Stricter detection confidence for this request")) -> Filters:
    for d in (date_from, date_to):
        if d:
            try:
                date.fromisoformat(d)
            except ValueError as err:
                raise HTTPException(422, f"Invalid date: {d}") from err
    return Filters(fight=fight, video=video, date_from=date_from, date_to=date_to, min_confidence=min_confidence)


def _check_key(db: Session, key: str, ex) -> None:
    if not KEY_RE.match(key):
        raise HTTPException(422, "Invalid fighter key")
    if any(o.fighter_key == key for o in ex.obs) or db.get(Fighter, key) is not None:
        return
    raise HTTPException(404, "Fighter not found (no fighter profile and no attributed actions with this key)")


def _analysis(db: Session, key: str, f: Filters) -> dict:
    settings = get_pattern_settings(db)
    ex = extract(db, settings)
    _check_key(db, key, ex)
    if key not in ex.names:
        fighter = db.get(Fighter, key)
        ex.names[key] = fighter.name if fighter else key
    snap = latest_snapshot(db, key)
    version = fighter_version(ex, settings, key)
    if not f.active and snap and snap.dataset_version == version:
        result = snap.result
        meta = {"mode": "snapshot", "snapshot_at": snap.created_at.isoformat(), "stale": False}
    else:
        obs = apply_filters(ex, f.fight, f.date_from, f.date_to, f.video, f.min_confidence)
        result = analyze_fighter(obs, key, settings, ex)
        meta = {"mode": "live", "snapshot_at": snap.created_at.isoformat() if snap else None,
                "stale": bool(snap and snap.dataset_version != version and not f.active)}
    meta.update({"dataset_version": version, "filters": f.model_dump(exclude_none=True),
                 "available_fights": sorted({v["fight_key"] for v in ex.videos.values() if key in v["fighters"]})})
    return {**result, "meta": meta}


# ============================================================================ index, settings, fights, jobs
@router.get("", summary="Fighters with pattern data, analysis status and dataset overview")
def overview(db: Session = Depends(get_db)):
    settings = get_pattern_settings(db)
    ex = extract(db, settings)
    fighters = fighter_index(ex)
    profiles = {f.id: f for f in db.scalars(select(Fighter)).all()}
    for d in fighters:
        snap = latest_snapshot(db, d["fighter_key"])
        version = fighter_version(ex, settings, d["fighter_key"])
        d["analysis"] = ("up_to_date" if snap and snap.dataset_version == version else "stale" if snap else "not_analysed")
        d["analysed_at"] = snap.created_at.isoformat() if snap else None
        d["stance"] = profiles[d["fighter_key"]].stance.value if d["fighter_key"] in profiles else None
    profiles_without = [{"fighter_key": f.id, "name": f.name} for f in profiles.values()
                        if f.id not in {d["fighter_key"] for d in fighters}]
    jobs = db.scalars(select(PatternAnalysisJob).order_by(PatternAnalysisJob.created_at.desc()).limit(10)).all()
    return {"fighters": fighters, "profiles_without_actions": profiles_without, "videos": list(ex.videos.values()),
            "extraction": {**ex.stats, "cache_hits": ex.cache_hits, "cache_misses": ex.cache_misses},
            "settings": settings.model_dump(), "jobs": [_job(j) for j in jobs],
            "totals": {"actions": len(ex.obs), "fighters": len(fighters), "videos": len(ex.videos),
                       "fights": len({v["fight_key"] for v in ex.videos.values()})}}


@router.get("/settings", response_model=PatternSettings)
def read_settings(db: Session = Depends(get_db)):
    return get_pattern_settings(db)


@router.put("/settings", response_model=PatternSettings)
def update_settings(body: PatternSettings, db: Session = Depends(get_db)):
    return save_pattern_settings(db, body)


class FightInfoIn(BaseModel):
    fight_key: str | None = Field(None, max_length=64)
    fight_date: str | None = None
    event: str | None = Field(None, max_length=200)
    notes: str | None = None

    @field_validator("fight_date")
    @classmethod
    def _date(cls, v):
        if v:
            date.fromisoformat(v)
        return v or None

    @field_validator("fight_key")
    @classmethod
    def _key(cls, v):
        v = (v or "").strip()
        if v and not KEY_RE.match(v):
            raise ValueError("fight_key may contain letters, digits, '-' and '_' only")
        return v or None


def _fight_row(v: Video, fi: FightInfo | None) -> dict:
    return {"video_id": v.id, "title": v.title, "duration": v.duration_seconds,
            "fight_key": fi.fight_key if fi else None, "fight_date": fi.fight_date if fi else None,
            "event": fi.event if fi else None, "notes": fi.notes if fi else None}


@router.get("/fights", summary="Fight grouping and dates per video (used for cross-fight statistics and date filters)")
def list_fights(db: Session = Depends(get_db)):
    infos = {f.video_id: f for f in db.scalars(select(FightInfo)).all()}
    return [_fight_row(v, infos.get(v.id)) for v in db.scalars(select(Video).order_by(Video.uploaded_at.desc())).all()]


@router.put("/fights/{video_id}")
def set_fight(video_id: str, body: FightInfoIn, db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "Video not found")
    fi = db.get(FightInfo, video_id) or FightInfo(video_id=video_id)
    fi.fight_key, fi.fight_date, fi.event, fi.notes = body.fight_key, body.fight_date, body.event, body.notes
    db.add(fi)
    db.commit()
    return _fight_row(v, fi)


def _job(j: PatternAnalysisJob) -> dict:
    return {"id": j.id, "fighter_key": j.fighter_key, "status": j.status.value, "progress": j.progress,
            "current_stage": j.current_stage, "dataset_version": j.dataset_version, "stats": j.stats,
            "warnings": j.warnings, "error_message": j.error_message,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


@router.get("/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    return [_job(j) for j in db.scalars(select(PatternAnalysisJob).order_by(PatternAnalysisJob.created_at.desc()).limit(limit)).all()]


@router.post("/analyze-all", status_code=202, summary="Analyse every fighter (unchanged fighters are skipped)")
def analyze_all(force: bool = False, db: Session = Depends(get_db), mgr: PatternJobManager = Depends(get_pattern_manager)):
    return _job(mgr.create(db, "*", force))


@router.get("/compare", summary="Compare two fighters")
def compare(fighter_a: str, fighter_b: str, f: Filters = Depends(filters), db: Session = Depends(get_db)):
    if fighter_a == fighter_b:
        raise HTTPException(422, "Choose two different fighters")
    settings = get_pattern_settings(db)
    ex = extract(db, settings)
    for k in (fighter_a, fighter_b):
        _check_key(db, k, ex)
    obs = apply_filters(ex, f.fight, f.date_from, f.date_to, f.video, f.min_confidence)
    cfg = settings.engine_config()
    p = E.prepare(obs, cfg)
    res = E.compare(p, fighter_a, fighter_b, cfg, ex.footage)
    res["state_labels"] = {k: label(k) for k in res["states"]} | {E.END: "End of sequence"}
    res["names"] = {k: ex.names.get(k) or (db.get(Fighter, k).name if db.get(Fighter, k) else k) for k in (fighter_a, fighter_b)}
    res["warnings"] = [f"{res['names'][k]}: only {res['fighters'][k]['actions']} actions in {res['fighters'][k]['fights']} fight(s)."
                       for k in (fighter_a, fighter_b) if res["fighters"][k]["actions"] < 30 or res["fighters"][k]["fights"] < 2]
    res["meta"] = {"mode": "live", "filters": f.model_dump(exclude_none=True)}
    return res


# ============================================================================ per fighter
@router.get("/{fighter_id}", summary="Pattern summary for one fighter")
def fighter_summary(fighter_id: str, f: Filters = Depends(filters), db: Session = Depends(get_db)):
    r = _analysis(db, fighter_id, f)
    return {k: r[k] for k in ("fighter_key", "name", "is_track", "summary", "warnings", "config", "meta", "states", "state_labels")} | {
        "top_combinations": r["combinations"][:5],
        "most_likely_next": [row for row in r["markov"]["1"]["rows"]][:10],
    }


@router.get("/{fighter_id}/combinations")
def fighter_combinations(fighter_id: str, min_length: int = Query(2, ge=2, le=8), max_length: int = Query(8, ge=2, le=8),
                         f: Filters = Depends(filters), db: Session = Depends(get_db)):
    r = _analysis(db, fighter_id, f)
    combos = [c for c in r["combinations"] if min_length <= c["length"] <= max_length]
    return {"fighter_key": fighter_id, "name": r["name"], "summary": r["summary"], "meta": r["meta"], "warnings": r["warnings"],
            "thresholds": {"min_occurrences": r["config"]["min_occurrences"], "min_fights": r["config"]["min_fights"],
                           "significance": r["config"]["significance"]},
            "combinations": combos, "sequential_patterns": r["sequential_patterns"]}


@router.get("/{fighter_id}/transitions", summary="Markov transition matrix (rows = previous action(s), columns = next)")
def fighter_transitions(fighter_id: str, order: int = Query(1, ge=1, le=3), f: Filters = Depends(filters),
                        db: Session = Depends(get_db)):
    r = _analysis(db, fighter_id, f)
    m = r["markov"].get(str(order))
    if m is None:
        raise HTTPException(422, f"Order {order} exceeds the configured maximum order ({r['config']['max_order']})")
    return {"fighter_key": fighter_id, "name": r["name"], "meta": r["meta"], "warnings": r["warnings"], "markov": m}


@router.get("/{fighter_id}/probabilities", summary="Conditional probabilities P(next | previous action(s))")
def fighter_probabilities(fighter_id: str, definition: str = Query("next", description="next | next_offensive | window"),
                          order: int = Query(1, ge=1, le=3), action: list[str] | None = Query(None, description="Context, e.g. action=left_jab&action=right_cross"),
                          f: Filters = Depends(filters), db: Session = Depends(get_db)):
    if definition not in DEFINITIONS:
        raise HTTPException(422, f"definition must be one of {', '.join(DEFINITIONS)}")
    if action:
        bad = [a for a in action if a not in BY_KEY]
        if bad:
            raise HTTPException(422, f"Unknown action(s): {', '.join(bad)}")
        order = len(action)
    r = _analysis(db, fighter_id, f)
    rows = r["transitions"][definition].get(str(order))
    if rows is None:
        raise HTTPException(422, f"Order {order} exceeds the configured maximum order ({r['config']['max_order']})")
    if action:
        rows = [x for x in rows if x["context"] == action]
    definitions = {
        "next": "Next action of the same fighter in the same sequence (END = sequence ended).",
        "next_offensive": "Next offensive action of the same fighter in the same sequence (END = none followed).",
        "window": f"Next action of the same fighter starting within {r['config']['window']} s (NONE = nothing in that window).",
    }
    return {"fighter_key": fighter_id, "name": r["name"], "meta": r["meta"], "warnings": r["warnings"],
            "definition": definition, "definition_text": definitions[definition], "order": order,
            "context": action, "context_count": rows[0]["context_count"] if rows else 0,
            "probabilities": rows, "min_observations": r["config"]["min_observations"]}


@router.get("/{fighter_id}/tendencies")
def fighter_tendencies(fighter_id: str, f: Filters = Depends(filters), db: Session = Depends(get_db)):
    r = _analysis(db, fighter_id, f)
    return {"fighter_key": fighter_id, "name": r["name"], "summary": r["summary"], "meta": r["meta"],
            "warnings": r["warnings"], "tendencies": r["tendencies"]}


@router.get("/{fighter_id}/sequences", summary="Action sequences; with ?action= the before/after explorer")
def fighter_sequences(fighter_id: str, action: str | None = None, limit: int = Query(200, ge=1, le=5000),
                      f: Filters = Depends(filters), db: Session = Depends(get_db)):
    r = _analysis(db, fighter_id, f)
    out = {"fighter_key": fighter_id, "name": r["name"], "meta": r["meta"], "warnings": r["warnings"],
           "total_sequences": len(r["sequences"]), "sequences": r["sequences"][:limit]}
    if action:
        if action not in BY_KEY:
            raise HTTPException(422, f"Unknown action: {action}")
        settings = get_pattern_settings(db)
        seqs = [[E.Obs(fighter_key=fighter_id, video_id=s["video_id"], fight_key=s["fight_key"], category=a["category"],
                       start=a["start"], end=a["end"], confidence=a["confidence"]) for a in s["actions"]] for s in r["sequences"]]
        out["explorer"] = E.explorer(seqs, action, settings.engine_config())
    out["actions_available"] = [{"action": k, "label": label(k)} for k in r["states"]]
    return out


@router.get("/{fighter_id}/opponent-analysis")
def fighter_opponents(fighter_id: str, f: Filters = Depends(filters), db: Session = Depends(get_db)):
    r = _analysis(db, fighter_id, f)
    return {"fighter_key": fighter_id, "name": r["name"], "meta": r["meta"], "warnings": r["warnings"],
            "min_actions_per_opponent": r["config"]["min_actions_per_opponent"], **r["opponent_analysis"]}


@router.post("/{fighter_id}/analyze", status_code=202, summary="Queue a background analysis (skipped if data unchanged unless force=true)")
def fighter_analyze(fighter_id: str, force: bool = False, db: Session = Depends(get_db),
                    mgr: PatternJobManager = Depends(get_pattern_manager)):
    ex = extract(db, get_pattern_settings(db))
    _check_key(db, fighter_id, ex)
    return _job(mgr.create(db, fighter_id, force))


@router.get("/{fighter_id}/analysis-status")
def fighter_status(fighter_id: str, db: Session = Depends(get_db)):
    settings = get_pattern_settings(db)
    ex = extract(db, settings)
    _check_key(db, fighter_id, ex)
    job = db.scalar(select(PatternAnalysisJob).where(PatternAnalysisJob.fighter_key.in_([fighter_id, "*"]))
                    .order_by(PatternAnalysisJob.created_at.desc()).limit(1))
    snap = latest_snapshot(db, fighter_id)
    version = fighter_version(ex, settings, fighter_id)
    return {"fighter_key": fighter_id, "job": _job(job) if job else None,
            "snapshot": {"id": snap.id, "created_at": snap.created_at.isoformat(), "dataset_version": snap.dataset_version} if snap else None,
            "current_dataset_version": version,
            "state": ("processing" if job and job.status in (JobStatus.queued, JobStatus.processing) else
                      "up_to_date" if snap and snap.dataset_version == version else "stale" if snap else "not_analysed")}
