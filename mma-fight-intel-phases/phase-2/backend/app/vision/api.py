"""Phase 2 REST API: /api/vision/..."""
from __future__ import annotations

import io
import logging
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_storage
from app.db.models import Fighter, JobStatus, Video, VideoStatus
from app.db.vision_models import (
    FighterDetection,
    FighterDistance,
    MovementMetric,
    PoseLandmark,
    TrackingSession,
    VisionJob,
)
from app.services.storage import Storage, UnsafePathError
from app.vision.constants import AVAILABLE_MODELS, KEYPOINT_NAMES, PALETTE, SKELETON
from app.vision.job_manager import ACTIVE, VisionConflict, VisionJobManager, current_job
from app.vision.runtime import downloaded_models, torch_info
from app.vision.settings import VisionSettings, get_vision_settings, save_vision_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vision", tags=["computer vision (phase 2)"])

ROLES = ("fighter", "referee", "other", "unreviewed")
MOVEMENT_FIELDS = [c.name for c in MovementMetric.__table__.columns if c.name not in ("id", "job_id", "video_id")]


def get_vision_manager(request: Request) -> VisionJobManager:
    return request.app.state.vision_manager


# ----------------------------------------------------------------------------- schemas
class VisionJobOut(BaseModel):
    id: str
    video_id: str
    video_title: str | None = None
    status: JobStatus
    attempt: int
    progress: float
    current_stage: str | None
    error_message: str | None
    warnings: list[str] | None
    model_name: str | None
    device: str | None
    stats: dict | None
    settings_snapshot: dict | None
    has_annotated_video: bool
    created_at: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None


class Assignment(BaseModel):
    track_id: int
    fighter_id: str | None = Field(None, description="Fighter profile to link, or null to unassign")
    role: Literal["fighter", "referee", "other", "unreviewed"] | None = Field(
        None, description="Defaults to 'fighter' when a fighter_id is given")


class AssignRequest(BaseModel):
    assignments: list[Assignment] = Field(..., min_length=1, max_length=200)


# ----------------------------------------------------------------------------- helpers
def job_out(job: VisionJob, title: str | None = None) -> VisionJobOut:
    dur = (job.finished_at - job.started_at).total_seconds() if job.finished_at and job.started_at else None
    return VisionJobOut(
        id=job.id, video_id=job.video_id, video_title=title, status=job.status, attempt=job.attempt,
        progress=job.progress, current_stage=job.current_stage, error_message=job.error_message,
        warnings=job.warnings, model_name=job.model_name, device=job.device, stats=job.stats,
        settings_snapshot=job.settings_snapshot, has_annotated_video=bool(job.annotated_video_path),
        created_at=job.created_at.isoformat(), started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        duration_seconds=round(dur, 2) if dur is not None else None,
    )


def _video(db: Session, video_id: str) -> Video:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found")
    return v


def _current(db: Session, video_id: str) -> VisionJob:
    _video(db, video_id)
    job = current_job(db, video_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No completed vision analysis for this video yet")
    return job


def _track_filter(db: Session, job: VisionJob, track_id: int | None, fighter_id: str | None,
                  fighters_only: bool) -> list[int] | None:
    """Resolve track filters to a list of track ids (None = no filter)."""
    ids: set[int] | None = None
    if track_id is not None:
        ids = {track_id}
    if fighter_id:
        f_ids = set(db.scalars(select(TrackingSession.track_id).where(
            TrackingSession.job_id == job.id, TrackingSession.assigned_fighter_id == fighter_id)).all())
        ids = f_ids if ids is None else ids & f_ids
    if fighters_only:
        f_ids = set(db.scalars(select(TrackingSession.track_id).where(
            TrackingSession.job_id == job.id, TrackingSession.role == "fighter")).all())
        ids = f_ids if ids is None else ids & f_ids
    return None if ids is None else sorted(ids)


def _frame_range(q, model, start_frame: int | None, end_frame: int | None):
    if start_frame is not None:
        q = q.where(model.frame_number >= start_frame)
    if end_frame is not None:
        q = q.where(model.frame_number <= end_frame)
    return q


def _dims(job: VisionJob, video: Video) -> tuple[int, int]:
    st = job.stats or {}
    return int(st.get("width") or video.width or 1), int(st.get("height") or video.height or 1)


def _tracking_payload(db: Session, job: VisionJob) -> dict:
    sessions = db.scalars(select(TrackingSession).where(TrackingSession.job_id == job.id)
                          .order_by(TrackingSession.fighter_score.desc())).all()
    fighters = {f.id: f for f in db.scalars(select(Fighter).where(
        Fighter.id.in_([s.assigned_fighter_id for s in sessions if s.assigned_fighter_id]))).all()}
    tracks = []
    for s in sessions:
        f = fighters.get(s.assigned_fighter_id) if s.assigned_fighter_id else None
        tracks.append({
            "track_id": s.track_id, "color": PALETTE[(s.track_id - 1) % len(PALETTE)],
            "role": s.role, "suggested_role": s.suggested_role, "fighter_score": s.fighter_score,
            "assigned_fighter": {"id": f.id, "name": f.name, "stance": f.stance.value} if f else None,
            "first_frame": s.first_frame, "last_frame": s.last_frame,
            "first_timestamp": s.first_timestamp, "last_timestamp": s.last_timestamp,
            "frames_detected": s.frames_detected, "coverage": s.coverage, "video_coverage": s.video_coverage,
            "interruptions": s.interruptions, "segments": s.segments, "avg_confidence": s.avg_confidence,
            "avg_box_height": s.avg_box_height, "avg_pose_confidence": s.avg_pose_confidence,
            "status": s.status, "reviewed": s.reviewed,
        })
    st = job.stats or {}
    main = [t for t in tracks if t["status"] != "fragment"]
    fighter_tracks = [t for t in tracks if t["role"] == "fighter"] or [t for t in tracks if t["suggested_role"] == "fighter"]
    return {
        "job_id": job.id,
        "analysed_frames": st.get("frames_analysed"),
        "stride": st.get("stride"),
        "fps": st.get("fps"),
        "scene_cuts": st.get("scene_cuts", []),
        "tracks": tracks,
        "consistency": {
            "tracks_total": len(tracks),
            "tracks_non_fragment": len(main),
            "fragment_tracks": len(tracks) - len(main),
            "max_people_in_frame": st.get("max_people_in_frame"),
            "fighter_tracks_basis": "assigned roles" if any(t["role"] == "fighter" for t in tracks) else "suggested",
            "fighter_coverage": round(sum(t["coverage"] for t in fighter_tracks) / len(fighter_tracks), 4) if fighter_tracks else None,
            "fighter_video_coverage": round(sum(t["video_coverage"] for t in fighter_tracks), 4) if fighter_tracks else None,
            "fighter_interruptions": sum(t["interruptions"] for t in fighter_tracks),
            "note": "Track IDs can switch when fighters cross, clinch, overlap or leave the frame, and always reset at "
                    "camera cuts. Assign every track of an athlete to the same fighter profile.",
        },
    }


# ----------------------------------------------------------------------------- jobs
@router.post("/{video_id}/analyze", response_model=VisionJobOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Queue detection + tracking + pose analysis (background). Also used to re-run/retry.")
def analyze(video_id: str, db: Session = Depends(get_db), vm: VisionJobManager = Depends(get_vision_manager)):
    video = _video(db, video_id)
    try:
        job = vm.create_job(db, video)
    except VisionConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return job_out(job, video.title)


@router.get("/{video_id}/status", summary="Analysis status, progress, current results summary and history")
def analysis_status(video_id: str, db: Session = Depends(get_db)):
    video = _video(db, video_id)
    jobs = db.scalars(select(VisionJob).where(VisionJob.video_id == video_id).order_by(VisionJob.created_at.desc())).all()
    active = next((j for j in jobs if j.status in ACTIVE), None)
    cur = current_job(db, video_id)
    can, reason = True, None
    if video.status != VideoStatus.completed:
        can, reason = False, "Phase 1 processing must complete first"
    elif active:
        can, reason = False, f"Analysis already {active.status.value}"
    return {
        "video_id": video_id,
        "video_title": video.title,
        "video_status": video.status.value,
        "can_analyze": can,
        "reason": reason,
        "active_job": job_out(active, video.title) if active else None,
        "current": job_out(cur, video.title) if cur else None,
        "annotated_video_url": f"/api/vision/{video_id}/annotated-video?v={cur.id}" if cur and cur.annotated_video_path else None,
        "history": [job_out(j, video.title) for j in jobs],
    }


@router.delete("/{video_id}/analysis", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete all vision analyses, results and annotated videos for a video (cancels a running job)")
def delete_analysis(video_id: str, db: Session = Depends(get_db), vm: VisionJobManager = Depends(get_vision_manager)):
    _video(db, video_id)
    try:
        vm.remove_analysis(db, video_id)
    except VisionConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/jobs", summary="List vision jobs (all videos)")
def list_jobs(status_filter: JobStatus | None = Query(None, alias="status"), limit: int = Query(100, ge=1, le=500),
              db: Session = Depends(get_db)):
    q = select(VisionJob, Video.title).join(Video, Video.id == VisionJob.video_id)
    if status_filter:
        q = q.where(VisionJob.status == status_filter)
    rows = db.execute(q.order_by(VisionJob.created_at.desc()).limit(limit)).all()
    counts = dict(db.execute(select(VisionJob.status, func.count()).group_by(VisionJob.status)).all())
    return {"items": [job_out(j, t).model_dump() for j, t in rows], "counts": {s.value: counts.get(s, 0) for s in JobStatus}}


@router.post("/jobs/{job_id}/retry", response_model=VisionJobOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Retry a failed or cancelled vision job (same settings)")
def retry_job(job_id: str, db: Session = Depends(get_db), vm: VisionJobManager = Depends(get_vision_manager)):
    job = db.get(VisionJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status not in (JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Only failed or cancelled jobs can be retried (status: {job.status.value})")
    video = _video(db, job.video_id)
    try:
        settings = VisionSettings.model_validate(job.settings_snapshot or {})
        new = vm.create_job(db, video, settings)
    except VisionConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return job_out(new, video.title)


# ----------------------------------------------------------------------------- results
@router.get("/{video_id}/tracking", summary="Tracking sessions (one per track ID) with consistency statistics")
def tracking(video_id: str, db: Session = Depends(get_db)):
    return _tracking_payload(db, _current(db, video_id))


@router.post("/{video_id}/assign-fighter", summary="Assign track IDs to fighter profiles and mark roles")
def assign_fighter(video_id: str, body: AssignRequest, db: Session = Depends(get_db)):
    job = _current(db, video_id)
    video = _video(db, video_id)
    sessions = {s.track_id: s for s in db.scalars(select(TrackingSession).where(TrackingSession.job_id == job.id)).all()}
    for a in body.assignments:
        s = sessions.get(a.track_id)
        if s is None:
            raise HTTPException(422, f"Track {a.track_id} does not exist in the current analysis")
        if a.fighter_id:
            fighter = db.get(Fighter, a.fighter_id)
            if fighter is None:
                raise HTTPException(422, f"Unknown fighter id {a.fighter_id}")
            s.assigned_fighter_id = fighter.id
            s.role = a.role or "fighter"
            if fighter not in video.fighters:  # keep Phase 1's video <-> fighter link in sync
                video.fighters.append(fighter)
        else:
            s.assigned_fighter_id = None
            s.role = a.role or "unreviewed"
        s.reviewed = s.role != "unreviewed"
    db.commit()
    return _tracking_payload(db, job)


@router.get("/{video_id}/detections", summary="Fighter detections (bounding boxes, confidence, track id)")
def detections(
    video_id: str,
    track_id: int | None = None,
    fighter_id: str | None = Query(None, description="Only tracks assigned to this fighter"),
    fighters_only: bool = Query(False, description="Only tracks whose role is 'fighter'"),
    start_frame: int | None = Query(None, ge=0),
    end_frame: int | None = Query(None, ge=0),
    min_confidence: float = Query(0.0, ge=0, le=1),
    limit: int = Query(1000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    job = _current(db, video_id)
    w, h = _dims(job, _video(db, video_id))
    q = select(FighterDetection).where(FighterDetection.job_id == job.id, FighterDetection.confidence >= min_confidence)
    ids = _track_filter(db, job, track_id, fighter_id, fighters_only)
    if ids is not None:
        q = q.where(FighterDetection.track_id.in_(ids))
    q = _frame_range(q, FighterDetection, start_frame, end_frame)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(FighterDetection.frame_number, FighterDetection.track_id).limit(limit).offset(offset)).all()
    return {
        "video_id": video_id, "job_id": job.id, "frame_width": w, "frame_height": h, "total": total,
        "limit": limit, "offset": offset,
        "items": [{
            "frame_number": r.frame_number, "timestamp": r.timestamp, "track_id": r.track_id,
            "confidence": r.confidence,
            "bbox": [round(r.x1 * w, 1), round(r.y1 * h, 1), round(r.x2 * w, 1), round(r.y2 * h, 1)],
            "bbox_norm": [round(r.x1, 5), round(r.y1, 5), round(r.x2, 5), round(r.y2, 5)],
        } for r in rows],
    }


@router.get("/{video_id}/pose", summary="Pose landmarks (normalised x/y, confidence) per tracked person")
def pose(
    video_id: str,
    track_id: int | None = None,
    fighter_id: str | None = None,
    fighters_only: bool = False,
    start_frame: int | None = Query(None, ge=0),
    end_frame: int | None = Query(None, ge=0),
    landmarks: str | None = Query(None, description="Comma-separated landmark names, e.g. left_wrist,right_wrist"),
    min_confidence: float = Query(0.0, ge=0, le=1, description="Drop landmarks below this confidence"),
    format: Literal["grouped", "long"] = Query("grouped", description="grouped: one item per person per frame; long: one row per landmark"),
    limit: int = Query(500, ge=1, le=20000, description="grouped: persons-frames; long: landmark rows"),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    job = _current(db, video_id)
    names = None
    if landmarks:
        names = [n.strip() for n in landmarks.split(",") if n.strip()]
        bad = [n for n in names if n not in KEYPOINT_NAMES]
        if bad:
            raise HTTPException(422, f"Unknown landmark(s): {', '.join(bad)}. Valid: {', '.join(KEYPOINT_NAMES)}")
    ids = _track_filter(db, job, track_id, fighter_id, fighters_only)

    base = select(PoseLandmark).where(PoseLandmark.job_id == job.id)
    if ids is not None:
        base = base.where(PoseLandmark.track_id.in_(ids))
    base = _frame_range(base, PoseLandmark, start_frame, end_frame)

    def lm_row(r: PoseLandmark) -> dict:
        return {"landmark": r.landmark, "x": r.x, "y": r.y, "confidence": r.confidence}

    if format == "long":
        q = base
        if names:
            q = q.where(PoseLandmark.landmark.in_(names))
        if min_confidence:
            q = q.where(PoseLandmark.confidence >= min_confidence)
        total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
        rows = db.scalars(q.order_by(PoseLandmark.frame_number, PoseLandmark.track_id, PoseLandmark.landmark_index)
                          .limit(limit).offset(offset)).all()
        return {"video_id": video_id, "job_id": job.id, "coordinates": "normalised [0,1]", "total": total,
                "items": [{"frame_number": r.frame_number, "timestamp": r.timestamp, "track_id": r.track_id, **lm_row(r)} for r in rows]}

    # grouped: paginate over (frame, track) pairs using the detections table
    dq = select(FighterDetection.frame_number, FighterDetection.track_id, FighterDetection.timestamp).where(
        FighterDetection.job_id == job.id)
    if ids is not None:
        dq = dq.where(FighterDetection.track_id.in_(ids))
    dq = _frame_range(dq, FighterDetection, start_frame, end_frame)
    total = db.scalar(select(func.count()).select_from(dq.subquery())) or 0
    keys = db.execute(dq.order_by(FighterDetection.frame_number, FighterDetection.track_id).limit(limit).offset(offset)).all()
    items = []
    if keys:
        lo, hi = keys[0][0], keys[-1][0]
        q = base.where(PoseLandmark.frame_number >= lo, PoseLandmark.frame_number <= hi)
        grouped: dict[tuple[int, int], list[dict]] = {}
        for r in db.scalars(q.order_by(PoseLandmark.landmark_index)).all():
            if names and r.landmark not in names:
                continue
            if r.confidence < min_confidence:
                continue
            grouped.setdefault((r.frame_number, r.track_id), []).append(lm_row(r))
        items = [{"frame_number": f, "timestamp": ts, "track_id": t, "keypoints": grouped.get((f, t), [])} for f, t, ts in keys]
    return {"video_id": video_id, "job_id": job.id, "coordinates": "normalised [0,1] (x by width, y by height)",
            "total": total, "limit": limit, "offset": offset, "items": items}


@router.get("/{video_id}/movement", summary="Smoothed movement features per track and analysed frame")
def movement(
    video_id: str,
    track_id: int | None = None,
    fighter_id: str | None = None,
    fighters_only: bool = False,
    start_frame: int | None = Query(None, ge=0),
    end_frame: int | None = Query(None, ge=0),
    limit: int = Query(5000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    job = _current(db, video_id)
    q = select(MovementMetric).where(MovementMetric.job_id == job.id)
    ids = _track_filter(db, job, track_id, fighter_id, fighters_only)
    if ids is not None:
        q = q.where(MovementMetric.track_id.in_(ids))
    q = _frame_range(q, MovementMetric, start_frame, end_frame)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(MovementMetric.track_id, MovementMetric.frame_number).limit(limit).offset(offset)).all()
    return {"video_id": video_id, "job_id": job.id, "total": total,
            "units": {"speeds": "torso lengths per second", "center": "pixels", "angles": "degrees",
                      "direction_deg": "0 = screen right, 90 = screen up", "stance_width": "torso lengths",
                      "guard_height": "torso lengths above shoulders (negative = below)"},
            "items": [{k: getattr(r, k) for k in MOVEMENT_FIELDS} for r in rows]}


@router.get("/{video_id}/distances", summary="Approximate distance between two tracked fighters over time")
def distances(video_id: str, track_a: int | None = None, track_b: int | None = None, db: Session = Depends(get_db)):
    job = _current(db, video_id)
    pairs = db.execute(select(FighterDistance.track_a, FighterDistance.track_b, func.count())
                       .where(FighterDistance.job_id == job.id).group_by(FighterDistance.track_a, FighterDistance.track_b)).all()
    if track_a is None or track_b is None:
        return {"video_id": video_id, "job_id": job.id, "pairs": [{"track_a": a, "track_b": b, "frames": n} for a, b, n in pairs],
                "items": [], "note": "Pass track_a and track_b to get the time series"}
    a, b = sorted((track_a, track_b))
    rows = db.scalars(select(FighterDistance).where(FighterDistance.job_id == job.id, FighterDistance.track_a == a,
                                                     FighterDistance.track_b == b).order_by(FighterDistance.frame_number)).all()
    return {"video_id": video_id, "job_id": job.id, "track_a": a, "track_b": b,
            "units": {"distance_frame": "fraction of frame height", "distance_torso": "mean torso lengths",
                      "closing_speed": "torso lengths / s, positive = approaching"},
            "pairs": [{"track_a": x, "track_b": y, "frames": n} for x, y, n in pairs],
            "items": [{"frame_number": r.frame_number, "timestamp": r.timestamp, "distance_frame": r.distance_frame,
                       "distance_torso": r.distance_torso, "closing_speed": r.closing_speed} for r in rows]}


@router.get("/{video_id}/overlay", summary="Compact per-frame boxes + keypoints for drawing overlays in the player")
def overlay(video_id: str, start_frame: int = Query(0, ge=0), end_frame: int = Query(..., ge=0),
            db: Session = Depends(get_db)):
    job = _current(db, video_id)
    if end_frame - start_frame > 20000:
        raise HTTPException(422, "Range too large (max 20000 frames per request)")
    dets = db.execute(select(FighterDetection.frame_number, FighterDetection.track_id, FighterDetection.confidence,
                             FighterDetection.x1, FighterDetection.y1, FighterDetection.x2, FighterDetection.y2)
                      .where(FighterDetection.job_id == job.id, FighterDetection.frame_number.between(start_frame, end_frame))
                      .order_by(FighterDetection.frame_number)).all()
    kps = db.execute(select(PoseLandmark.frame_number, PoseLandmark.track_id, PoseLandmark.landmark_index,
                            PoseLandmark.x, PoseLandmark.y, PoseLandmark.confidence)
                     .where(PoseLandmark.job_id == job.id, PoseLandmark.frame_number.between(start_frame, end_frame))).all()
    kp_map: dict[tuple[int, int], list[float]] = {}
    for f, t, i, x, y, c in kps:
        arr = kp_map.setdefault((f, t), [0.0] * 51)
        arr[i * 3: i * 3 + 3] = [round(x, 4), round(y, 4), round(c, 3)]
    frames: dict[int, list] = {}
    for f, t, c, x1, y1, x2, y2 in dets:
        frames.setdefault(f, []).append([t, round(c, 3), round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4), kp_map.get((f, t), [])])
    return {"job_id": job.id, "start_frame": start_frame, "end_frame": end_frame,
            "format": "frames: [[frame, [[track_id, conf, x1, y1, x2, y2, [x,y,c]*17], ...]], ...] (normalised)",
            "frames": [[f, people] for f, people in sorted(frames.items())]}


@router.get("/{video_id}/frame/{frame_number}", summary="Everything known about one frame (nearest analysed frame)")
def frame_details(video_id: str, frame_number: int, db: Session = Depends(get_db)):
    job = _current(db, video_id)
    stride = int((job.stats or {}).get("stride") or 1)
    f = db.scalar(select(FighterDetection.frame_number).where(
        FighterDetection.job_id == job.id, FighterDetection.frame_number <= frame_number,
        FighterDetection.frame_number > frame_number - stride).order_by(FighterDetection.frame_number.desc()).limit(1))
    analysed = frame_number - (frame_number % stride)
    if f is None:
        return {"requested_frame": frame_number, "analysed_frame": analysed, "people": [], "distances": []}
    w, h = _dims(job, _video(db, video_id))
    dets = db.scalars(select(FighterDetection).where(FighterDetection.job_id == job.id, FighterDetection.frame_number == f)).all()
    lms = db.scalars(select(PoseLandmark).where(PoseLandmark.job_id == job.id, PoseLandmark.frame_number == f)
                     .order_by(PoseLandmark.landmark_index)).all()
    mov = {m.track_id: m for m in db.scalars(select(MovementMetric).where(MovementMetric.job_id == job.id,
                                                                         MovementMetric.frame_number == f)).all()}
    sessions = {s.track_id: s for s in db.scalars(select(TrackingSession).where(TrackingSession.job_id == job.id)).all()}
    names = {x.id: x.name for x in db.scalars(select(Fighter)).all()}
    people = []
    for d in dets:
        s = sessions.get(d.track_id)
        people.append({
            "track_id": d.track_id, "confidence": d.confidence,
            "role": s.role if s else None, "fighter_name": names.get(s.assigned_fighter_id) if s else None,
            "bbox": [round(d.x1 * w), round(d.y1 * h), round(d.x2 * w), round(d.y2 * h)],
            "keypoints": [{"landmark": r.landmark, "x": r.x, "y": r.y, "confidence": r.confidence}
                          for r in lms if r.track_id == d.track_id],
            "movement": {k: getattr(mov[d.track_id], k) for k in MOVEMENT_FIELDS} if d.track_id in mov else None,
        })
    dists = db.scalars(select(FighterDistance).where(FighterDistance.job_id == job.id, FighterDistance.frame_number == f)).all()
    return {"requested_frame": frame_number, "analysed_frame": f, "timestamp": dets[0].timestamp if dets else None,
            "people": people,
            "distances": [{"track_a": x.track_a, "track_b": x.track_b, "distance_frame": x.distance_frame,
                           "distance_torso": x.distance_torso, "closing_speed": x.closing_speed} for x in dists]}


@router.get("/{video_id}/annotated-video", summary="Annotated MP4 (boxes, IDs, skeletons) with HTTP Range support")
def annotated_video(video_id: str, download: bool = False, db: Session = Depends(get_db),
                    storage: Storage = Depends(get_storage)):
    job = _current(db, video_id)
    try:
        path = storage.resolve(job.annotated_video_path)
    except UnsafePathError:
        path = None
    if path is None or not path.exists():
        raise HTTPException(404, "No annotated video for this analysis (disabled in settings or generation failed)")
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}_annotated.mp4" if download else None)


@router.get("/{video_id}/export", summary="Download results as CSV (pandas) or the NumPy bundle for Phase 3")
def export(video_id: str, kind: Literal["detections", "pose", "movement", "distances", "tracks", "npz"] = "pose",
           db: Session = Depends(get_db), storage: Storage = Depends(get_storage)):
    job = _current(db, video_id)
    if kind == "npz":
        path = storage.resolve(job.export_path) if job.export_path else None
        if path is None or not path.exists():
            raise HTTPException(404, "Export bundle not available")
        return FileResponse(path, media_type="application/octet-stream", filename=f"{video_id}_pose_tracks.npz")
    model = {"detections": FighterDetection, "pose": PoseLandmark, "movement": MovementMetric,
             "distances": FighterDistance, "tracks": TrackingSession}[kind]
    cols = [c for c in model.__table__.columns if c.name not in ("id", "job_id", "video_id")]
    stmt = select(*cols).where(model.job_id == job.id)
    df = pd.read_sql(stmt, db.connection())
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{video_id}_{kind}.csv"'})


# ----------------------------------------------------------------------------- dashboard / settings
@router.get("/overview", summary="Computer-vision dashboard statistics")
def vision_overview(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(VisionJob.status, func.count()).group_by(VisionJob.status)).all())
    # current analysis per video = latest completed
    sub = (select(VisionJob.video_id, func.max(VisionJob.finished_at).label("fin"))
           .where(VisionJob.status == JobStatus.completed).group_by(VisionJob.video_id).subquery())
    current = db.execute(select(VisionJob, Video.title).join(sub, (VisionJob.video_id == sub.c.video_id) & (VisionJob.finished_at == sub.c.fin))
                         .join(Video, Video.id == VisionJob.video_id).order_by(VisionJob.finished_at.desc())).all()
    job_ids = [j.id for j, _ in current]
    frames = sum(int((j.stats or {}).get("frames_analysed") or 0) for j, _ in current)
    det_conf = [(j.stats or {}).get("avg_detection_confidence") for j, _ in current]
    det_conf = [c for c in det_conf if c is not None]
    pose_conf = [c for c in ((j.stats or {}).get("avg_pose_confidence") for j, _ in current) if c is not None]
    fighter_tracks = db.scalar(select(func.count()).select_from(TrackingSession).where(
        TrackingSession.job_id.in_(job_ids), TrackingSession.role == "fighter")) or 0 if job_ids else 0
    suggested = db.scalar(select(func.count()).select_from(TrackingSession).where(
        TrackingSession.job_id.in_(job_ids), TrackingSession.suggested_role == "fighter")) or 0 if job_ids else 0
    distinct_fighters = db.scalar(select(func.count(func.distinct(TrackingSession.assigned_fighter_id))).where(
        TrackingSession.job_id.in_(job_ids), TrackingSession.assigned_fighter_id.is_not(None))) or 0 if job_ids else 0
    cov = db.execute(select(func.avg(TrackingSession.coverage), func.sum(TrackingSession.interruptions), func.count())
                     .where(TrackingSession.job_id.in_(job_ids), TrackingSession.suggested_role == "fighter")).one() if job_ids else (None, 0, 0)
    active = db.execute(select(VisionJob, Video.title).join(Video, Video.id == VisionJob.video_id)
                        .where(VisionJob.status.in_(ACTIVE)).order_by(VisionJob.created_at)).all()
    eligible = db.scalar(select(func.count()).select_from(Video).where(Video.status == VideoStatus.completed)) or 0
    latest = next(((j, t) for j, t in current if j.annotated_video_path), None)
    return {
        "videos_analysed": len(current),
        "videos_eligible": eligible,
        "job_counts": {s.value: counts.get(s, 0) for s in JobStatus},
        "total_frames_analysed": frames,
        "fighter_tracks_assigned": fighter_tracks,
        "fighter_tracks_suggested": suggested,
        "distinct_fighters_identified": distinct_fighters,
        "avg_detection_confidence": round(sum(det_conf) / len(det_conf), 4) if det_conf else None,
        "avg_pose_confidence": round(sum(pose_conf) / len(pose_conf), 4) if pose_conf else None,
        "tracking": {
            "avg_fighter_track_coverage": round(float(cov[0]), 4) if cov[0] is not None else None,
            "fighter_track_interruptions": int(cov[1] or 0),
            "fighter_tracks": int(cov[2] or 0),
        },
        "active_jobs": [job_out(j, t).model_dump() for j, t in active],
        "recent": [job_out(j, t).model_dump() for j, t in current[:6]],
        "latest_annotated": {"video_id": latest[0].video_id, "title": latest[1],
                             "url": f"/api/vision/{latest[0].video_id}/annotated-video?v={latest[0].id}"} if latest else None,
    }


@router.get("/settings", response_model=VisionSettings, summary="Vision analysis settings")
def read_settings(db: Session = Depends(get_db)):
    return get_vision_settings(db)


@router.put("/settings", response_model=VisionSettings, summary="Update vision settings (apply to new analyses)")
def write_settings(payload: VisionSettings, db: Session = Depends(get_db)):
    return save_vision_settings(db, payload)


@router.get("/system", summary="PyTorch / CUDA / model availability")
def system():
    return {"torch": torch_info(), "available_models": AVAILABLE_MODELS, "downloaded_models": downloaded_models(),
            "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON, "palette": PALETTE}
