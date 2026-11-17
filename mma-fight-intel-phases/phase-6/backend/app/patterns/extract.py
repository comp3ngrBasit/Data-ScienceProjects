"""Database -> pattern observations (Phase 4).

Reads Phase 3 recognised actions (with human corrections applied) or verified annotations, attributes
each action to a fighter, attaches fight / opponent / distance / stance context and caches the result per
video. Nothing here writes to Phase 2/3 tables.
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.actions.job_manager import current_action_job
from app.actions.taxonomy import BACKGROUND, BY_KEY, UNKNOWN
from app.db.action_models import ActionAnnotation, RecognizedAction, VideoAnnotationMeta
from app.db.models import Fighter, Video
from app.db.pattern_models import FightInfo, PatternVideoCache
from app.db.vision_models import FighterDistance, TrackingSession
from app.patterns.engine import Obs
from app.patterns.settings import PatternSettings
from app.vision.job_manager import current_job as current_vision_job

OBS_FIELDS = ("fighter_key", "video_id", "fight_key", "category", "start", "end", "confidence", "source", "action_id",
              "track_id", "fighter_id", "opponent_key", "opponent_id", "distance", "stance", "fight_date")


def track_key(video_id: str, track_id: int) -> str:
    return f"track-{video_id}-{track_id}"


def is_track_key(key: str) -> bool:
    return key.startswith("track-")


@dataclass
class Extraction:
    obs: list[Obs]
    videos: dict[str, dict] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    footage: dict[str, float] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0

    def version(self, settings: PatternSettings) -> str:
        h = hashlib.sha1()
        h.update(json.dumps(settings.model_dump(), sort_keys=True).encode())
        for o in sorted(self.obs, key=lambda o: (o.video_id, o.fighter_key, o.start, o.category)):
            h.update(f"{o.video_id}|{o.fighter_key}|{o.fight_key}|{o.fight_date}|{o.category}|{o.start:.3f}|{o.end:.3f}|{o.confidence:.3f}|{o.opponent_key}\n".encode())
        return h.hexdigest()[:16]


def _fighter_names(db: Session) -> dict[str, str]:
    return {f.id: f.name for f in db.scalars(select(Fighter)).all()}


def _video_inputs(db: Session, video: Video, s: PatternSettings) -> tuple[str | None, list[dict], dict]:
    """Choose the source for one video and return raw action rows + context needed for a signature."""
    meta = db.get(VideoAnnotationMeta, video.id)
    anns = db.scalars(select(ActionAnnotation).where(ActionAnnotation.video_id == video.id,
                                                     ActionAnnotation.status == "verified")).all()
    job = current_action_job(db, video.id)
    use = None
    if s.source == "annotations":
        use = "annotations" if anns else None
    elif s.source == "detections":
        use = "detections" if job else None
    else:  # best
        if meta and meta.fully_annotated and anns:
            use = "annotations"
        elif job:
            use = "detections"
    rows: list[dict] = []
    info = {"action_job_id": job.id if job else None, "fully_annotated": bool(meta and meta.fully_annotated),
            "verified_annotations": len(anns), "excluded": defaultdict(int)}
    if use == "annotations":
        for a in anns:
            if a.category in (UNKNOWN, BACKGROUND) or a.category not in BY_KEY:
                info["excluded"]["unknown_category"] += 1
                continue
            rows.append({"id": a.id, "category": a.category, "start": a.start_time, "end": a.end_time,
                         "confidence": float(a.confidence if a.confidence is not None else 1.0),
                         "track_id": a.track_id, "fighter_id": a.fighter_id, "source": "annotation"})
    elif use == "detections":
        for r in db.scalars(select(RecognizedAction).where(RecognizedAction.job_id == job.id)).all():
            if r.review_status == "rejected":
                info["excluded"]["rejected"] += 1
                continue
            cat = r.corrected_category or r.category
            if cat in (UNKNOWN, BACKGROUND) or cat not in BY_KEY:
                info["excluded"]["unknown_category"] += 1
                continue
            reviewed = r.review_status in ("confirmed", "corrected")
            if not reviewed:
                if r.low_confidence and not s.include_low_confidence:
                    info["excluded"]["low_confidence"] += 1
                    continue
                if r.confidence < s.min_confidence:
                    info["excluded"]["below_min_confidence"] += 1
                    continue
            rows.append({"id": r.id, "category": cat, "start": r.start_time, "end": r.end_time,
                         "confidence": 1.0 if reviewed else float(r.confidence), "track_id": r.track_id,
                         "fighter_id": r.fighter_id, "source": "review" if reviewed else "detection"})
    info["excluded"] = dict(info["excluded"])
    return use, rows, info


def _signature(video: Video, use: str | None, rows: list[dict], vjob_id: str | None, tracks: dict, fi: FightInfo | None,
               meta: VideoAnnotationMeta | None, stances: dict) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"use": use, "rows": sorted(rows, key=lambda r: r["id"]), "vjob": vjob_id,
                         "tracks": sorted(tracks.items()), "fight": [fi.fight_key, fi.fight_date] if fi else None,
                         "group": meta.split_group if meta else None, "stances": sorted(stances.items()),
                         "duration": video.duration_seconds, "target": video.target_fighter_id, "v": 2}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _distance_lookup(db: Session, vjob_id: str | None):
    if not vjob_id:
        return lambda *_: None
    rows = db.execute(select(FighterDistance.track_a, FighterDistance.track_b, FighterDistance.timestamp,
                             FighterDistance.distance_torso).where(FighterDistance.job_id == vjob_id)
                      .order_by(FighterDistance.timestamp)).all()
    pairs: dict[tuple[int, int], tuple[list[float], list[float | None]]] = defaultdict(lambda: ([], []))
    for a, b, t, d in rows:
        ts, ds = pairs[(min(a, b), max(a, b))]
        ts.append(t)
        ds.append(d)

    def lookup(track: int | None, other_tracks: list[int], t: float) -> float | None:
        if track is None:
            return None
        best = None
        for ot in other_tracks:
            ts, ds = pairs.get((min(track, ot), max(track, ot)), ([], []))
            if not ts:
                continue
            i = bisect_left(ts, t)
            for j in (i - 1, i):
                if 0 <= j < len(ts) and abs(ts[j] - t) <= 0.5 and ds[j] is not None:
                    if best is None or abs(ts[j] - t) < best[0]:
                        best = (abs(ts[j] - t), ds[j])
        return round(best[1], 3) if best else None

    return lookup


def _build_video_obs(db: Session, video: Video, use: str, rows: list[dict], tracks: dict[int, str | None],
                     fight_key: str, fight_date: str | None, stances: dict[str, str], vjob_id: str | None) -> tuple[list[dict], int]:
    unattributed = 0
    items = []
    for r in rows:
        fid = (tracks.get(r["track_id"]) if r["track_id"] is not None else None) or r["fighter_id"]
        if not fid and r["track_id"] is None and video.target_fighter_id:
            fid = video.target_fighter_id  # annotation without track: the video's designated target fighter
        if fid:
            key = fid
        elif r["track_id"] is not None:
            key = track_key(video.id, r["track_id"])
        else:
            unattributed += 1
            continue
        items.append({**r, "fighter_key": key, "fighter_id": fid})
    keys = sorted({i["fighter_key"] for i in items} | {f for f in tracks.values() if f})
    key_tracks: dict[str, set[int]] = defaultdict(set)
    for i in items:
        if i["track_id"] is not None:
            key_tracks[i["fighter_key"]].add(i["track_id"])
    for t, f in tracks.items():
        if f:
            key_tracks[f].add(t)
    dist = _distance_lookup(db, vjob_id)
    out = []
    for i in items:
        others = [k for k in keys if k != i["fighter_key"]]
        opp = others[0] if len(others) == 1 else None
        if len(others) > 1:  # pick the closest other fighter at that moment
            best = None
            for k in others:
                d = dist(i["track_id"], sorted(key_tracks[k]), i["start"])
                if d is not None and (best is None or d < best[0]):
                    best = (d, k)
            opp = best[1] if best else None
        d = dist(i["track_id"], sorted(key_tracks[opp]), i["start"]) if opp else None
        out.append({"fighter_key": i["fighter_key"], "video_id": video.id, "fight_key": fight_key, "category": i["category"],
                    "start": float(i["start"]), "end": float(i["end"]), "confidence": i["confidence"], "source": i["source"],
                    "action_id": i["id"], "track_id": i["track_id"], "fighter_id": i["fighter_id"], "opponent_key": opp,
                    "opponent_id": opp if opp and not is_track_key(opp) else None, "distance": d,
                    "stance": stances.get(i["fighter_id"]) if i["fighter_id"] else None, "fight_date": fight_date})
    return out, unattributed


def extract(db: Session, settings: PatternSettings, use_cache: bool = True) -> Extraction:
    names = _fighter_names(db)
    stances = {f.id: (f.stance.value if hasattr(f.stance, "value") else str(f.stance))
               for f in db.scalars(select(Fighter)).all()}
    stances = {k: v for k, v in stances.items() if v and v.lower() != "unknown"}
    ex = Extraction(obs=[], names=dict(names))
    stats = defaultdict(int)
    for video in db.scalars(select(Video).order_by(Video.uploaded_at)).all():
        use, rows, info = _video_inputs(db, video, settings)
        for k, v in info["excluded"].items():
            stats[f"excluded_{k}"] += v
        if not use:
            continue
        vjob = current_vision_job(db, video.id)
        tracks: dict[int, str | None] = {}
        if vjob:
            for ts in db.scalars(select(TrackingSession).where(TrackingSession.job_id == vjob.id)).all():
                tracks[ts.track_id] = ts.assigned_fighter_id
        fi = db.get(FightInfo, video.id)
        meta = db.get(VideoAnnotationMeta, video.id)
        fight_key = (fi.fight_key if fi and fi.fight_key else None) or (meta.split_group if meta and meta.split_group else None) or video.id
        fight_date = fi.fight_date if fi else None
        sig = _signature(video, use, rows, vjob.id if vjob else None, tracks, fi, meta, stances)
        cache = db.get(PatternVideoCache, video.id)
        if use_cache and cache and cache.signature == sig:
            obs_dicts = cache.observations
            ex.cache_hits += 1
        else:
            obs_dicts, unattributed = _build_video_obs(db, video, use, rows, tracks, fight_key, fight_date, stances,
                                                       vjob.id if vjob else None)
            stats["excluded_unattributed"] += unattributed
            ex.cache_misses += 1
            if not use_cache:
                pass  # alternative-source extraction (evaluation): never overwrite the regular cache
            elif cache is None:
                db.add(PatternVideoCache(video_id=video.id, signature=sig, source=use, observations=obs_dicts))
            else:
                cache.signature, cache.source, cache.observations = sig, use, obs_dicts
        obs = [Obs(**{k: d.get(k) for k in OBS_FIELDS}) for d in obs_dicts]
        ex.obs.extend(obs)
        ex.footage[video.id] = float(video.duration_seconds or 0.0)
        ex.videos[video.id] = {"video_id": video.id, "title": video.title, "source": use, "actions": len(obs),
                               "fight_key": fight_key, "fight_date": fight_date, "event": fi.event if fi else None,
                               "duration": video.duration_seconds, "fighters": sorted({o.fighter_key for o in obs})}
        for o in obs:
            if o.fighter_key not in ex.names:
                ex.names[o.fighter_key] = f"Unassigned track {o.track_id} · {video.title}"
    db.commit()
    stats["videos_used"] = len(ex.videos)
    stats["actions_extracted"] = len(ex.obs)
    ex.stats = dict(stats)
    return ex


def apply_filters(ex: Extraction, fights: list[str] | None = None, date_from: str | None = None,
                  date_to: str | None = None, videos: list[str] | None = None, min_confidence: float | None = None) -> list[Obs]:
    out = []
    for o in ex.obs:
        if fights and o.fight_key not in fights:
            continue
        if videos and o.video_id not in videos:
            continue
        if (date_from or date_to) and not o.fight_date:
            continue  # undated fights can't satisfy a date filter
        if date_from and o.fight_date < date_from:
            continue
        if date_to and o.fight_date > date_to:
            continue
        if min_confidence is not None and o.source == "detection" and o.confidence < min_confidence:
            continue
        out.append(o)
    # fresh copies: prepare() annotates objects in place
    return [Obs(**{k: getattr(o, k) for k in OBS_FIELDS}) for o in out]


def obs_to_dict(o: Obs) -> dict:
    d = asdict(o)
    d["group"] = o.group
    return d
