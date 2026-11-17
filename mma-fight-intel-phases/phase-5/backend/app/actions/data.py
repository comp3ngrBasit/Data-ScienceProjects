"""Loading Phase 2 pose data and turning annotations into a training dataset."""
from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.actions.features import FeatureConfig, TrackPose, window_frames
from app.actions.taxonomy import BACKGROUND
from app.db.action_models import ActionAnnotation, VideoAnnotationMeta
from app.db.models import Video
from app.db.vision_models import PoseLandmark, TrackingSession, VisionJob
from app.services.storage import Storage
from app.vision.job_manager import current_job

log = logging.getLogger(__name__)


@dataclass
class VideoPose:
    video_id: str
    vision_job: VisionJob
    fps: float
    width: int
    height: int
    tracks: dict[int, TrackPose]
    sessions: dict[int, TrackingSession]
    scene_cuts: list[int]


def load_video_pose(db: Session, storage: Storage, video_id: str) -> VideoPose | None:
    """Pose sequences for every track of the video's current Phase 2 analysis (None if not analysed)."""
    job = current_job(db, video_id)
    if job is None:
        return None
    st = job.stats or {}
    video = db.get(Video, video_id)
    fps = float(st.get("fps") or (video.fps if video else 0) or 30.0)
    w = int(st.get("width") or (video.width if video else 1) or 1)
    h = int(st.get("height") or (video.height if video else 1) or 1)
    tracks: dict[int, TrackPose] = {}

    path = storage.resolve(job.export_path) if job.export_path else None
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as z:
            ids = sorted({int(k.split("_")[0][1:]) for k in z.files if k.startswith("t") and k.endswith("_frames")})
            for tid in ids:
                kp = z[f"t{tid}_keypoints"].astype(np.float64) * np.array([w, h])
                tracks[tid] = TrackPose(tid, z[f"t{tid}_frames"].astype(np.int64), kp, z[f"t{tid}_keypoint_conf"], fps)
    else:  # fall back to the database (slower, same data)
        rows = db.execute(select(PoseLandmark.track_id, PoseLandmark.frame_number, PoseLandmark.landmark_index,
                                 PoseLandmark.x, PoseLandmark.y, PoseLandmark.confidence)
                          .where(PoseLandmark.job_id == job.id)
                          .order_by(PoseLandmark.track_id, PoseLandmark.frame_number)).all()
        acc: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
        for tid, f, i, x, y, c in rows:
            k, cf = acc[tid].setdefault(f, (np.zeros((17, 2)), np.zeros(17)))
            k[i] = (x * w, y * h)
            cf[i] = c
        for tid, by_f in acc.items():
            fr = np.array(sorted(by_f))
            tracks[tid] = TrackPose(tid, fr, np.stack([by_f[f][0] for f in fr]), np.stack([by_f[f][1] for f in fr]), fps)

    sessions = {s.track_id: s for s in db.scalars(select(TrackingSession).where(TrackingSession.job_id == job.id)).all()}
    return VideoPose(video_id, job, fps, w, h, tracks, sessions, list(st.get("scene_cuts") or []))


def resolve_track(vp: VideoPose, db: Session, ann: ActionAnnotation) -> int | None:
    """Which tracked person an annotation refers to.

    1. explicit track id, 2. tracks assigned to the annotated fighter that overlap the time,
    3. if only one person is tracked in that interval, that person. Otherwise ambiguous (None).
    """
    t0, t1 = ann.start_time, ann.end_time
    f0, f1 = int(t0 * vp.fps), int(t1 * vp.fps) + 1

    def overlap(tid: int) -> int:
        tp = vp.tracks.get(tid)
        return 0 if tp is None else int(((tp.frames >= f0) & (tp.frames <= f1)).sum())

    if ann.track_id is not None and overlap(ann.track_id) > 0:
        return ann.track_id
    if ann.fighter_id:
        cands = [tid for tid, s in vp.sessions.items() if s.assigned_fighter_id == ann.fighter_id and overlap(tid) > 0]
        if cands:
            return max(cands, key=overlap)
    present = [tid for tid in vp.tracks if overlap(tid) > 0]
    if len(present) == 1:
        return present[0]
    if ann.track_id is None and not ann.fighter_id and present:
        # 1) exactly one confirmed fighter in view
        fighters = [tid for tid in present if vp.sessions.get(tid) and vp.sessions[tid].role == "fighter"]
        if len(fighters) == 1:
            return fighters[0]
        # 2) a clearly dominant person (single-athlete clips such as BoxingVI): largest on screen by >= 1.5x
        def size(tid: int) -> float:
            tp = vp.tracks[tid]
            sel = (tp.frames >= f0) & (tp.frames <= f1)
            k = tp.keypoints[sel]
            if not len(k):
                return 0.0
            return float(np.median(k[:, :, 1].max(1) - k[:, :, 1].min(1))) * overlap(tid)
        ranked = sorted(present, key=size, reverse=True)
        if len(ranked) == 1 or size(ranked[0]) >= 1.5 * size(ranked[1]):
            return ranked[0]
    return None


@dataclass
class Sample:
    X: np.ndarray  # (T, F)
    label: str
    group: str  # split group (video or fight)
    video_id: str
    split: str  # train | test
    source: str  # annotation id or "negative"
    t0: float
    t1: float
    augmented: bool = False  # temporal jitter copy — used for training only, never evaluated


@dataclass
class Dataset:
    samples: list[Sample]
    version: str
    summary: dict
    warnings: list[str] = field(default_factory=list)


def dataset_version(anns: list[ActionAnnotation], metas: list[VideoAnnotationMeta], cfg: FeatureConfig, vision_ids: list[str]) -> str:
    h = hashlib.sha1()
    for a in sorted(anns, key=lambda a: a.id):
        h.update(f"{a.id}|{a.category}|{a.start_time:.3f}|{a.end_time:.3f}|{a.status}|{a.track_id}|{a.fighter_id}".encode())
    for m in sorted(metas, key=lambda m: m.video_id):
        h.update(f"{m.video_id}|{m.fully_annotated}|{m.split}|{m.split_group}".encode())
    h.update(repr(sorted(cfg.as_dict().items())).encode())
    h.update("|".join(sorted(vision_ids)).encode())
    return h.hexdigest()[:12]


def build_dataset(db: Session, storage: Storage, cfg: FeatureConfig, include_status=("verified",),
                  negatives_per_positive: float = 1.5, augment: int = 4, seed: int = 13) -> Dataset:
    rng = random.Random(seed)
    anns = db.scalars(select(ActionAnnotation).where(ActionAnnotation.status.in_(include_status))).all()
    metas = {m.video_id: m for m in db.scalars(select(VideoAnnotationMeta)).all()}
    by_video: dict[str, list[ActionAnnotation]] = defaultdict(list)
    for a in anns:
        if a.category not in ("unknown",):
            by_video[a.video_id].append(a)

    for vid, m in metas.items():  # fully annotated videos with no actions are pure background footage
        if m.fully_annotated and vid not in by_video and m.split != "exclude":
            by_video[vid] = []
    durs = [a.end_time - a.start_time + 2 * cfg.context for items in by_video.values() for a in items]
    global_dur = float(np.median(durs)) if durs else 0.8

    samples: list[Sample] = []
    warnings: list[str] = []
    skipped = Counter()
    vision_ids = []
    for vid, items in by_video.items():
        meta = metas.get(vid)
        if meta and meta.split == "exclude":
            skipped["excluded_video"] += len(items)
            continue
        vp = load_video_pose(db, storage, vid)
        if vp is None:
            skipped["video_not_analysed"] += len(items)
            warnings.append(f"Video {vid}: {len(items)} annotation(s) skipped — run Phase 2 analysis first")
            continue
        if not vp.tracks:
            continue
        vision_ids.append(vp.vision_job.id)
        group = (meta.split_group if meta and meta.split_group else vid)
        split = meta.split if meta and meta.split in ("train", "test") else "auto"
        stride_s = 1.0 / vp.fps * int((vp.vision_job.stats or {}).get("stride") or 1)
        if stride_s > 1 / 15:
            warnings.append(f"Video {vid}: pose sampled every {stride_s:.2f}s — fast strikes are under-sampled; "
                            "re-run Phase 2 with analysis rate ≥ 25 fps for action work")
        used: dict[int, list[tuple[float, float]]] = defaultdict(list)
        durations = []
        for a in items:
            tid = resolve_track(vp, db, a)
            if tid is None:
                skipped["ambiguous_fighter"] += 1
                continue
            t0, t1 = a.start_time - cfg.context, a.end_time + cfg.context
            res = window_frames(vp.tracks[tid], t0, t1, cfg)
            if res is None:
                skipped["no_pose_in_window"] += 1
                continue
            X, info = res
            if info["visible_frac"] < 0.3:
                skipped["low_pose_confidence"] += 1
                continue
            samples.append(Sample(X, a.category, group, vid, split, a.id, t0, t1))
            used[tid].append((t0, t1))
            durations.append(t1 - t0)
            d = t1 - t0
            for _ in range(augment):  # temporal jitter: shift +-20 % and stretch 0.85-1.15
                c = (t0 + t1) / 2 + rng.uniform(-0.2, 0.2) * d
                half = d * rng.uniform(0.85, 1.15) / 2
                r2 = window_frames(vp.tracks[tid], c - half, c + half, cfg)
                if r2 is not None:
                    samples.append(Sample(r2[0], a.category, group, vid, split, a.id, c - half, c + half, augmented=True))

        # Negatives only from fully-annotated videos, on the same tracks, away from any labelled action
        if meta and meta.fully_annotated:
            all_marks = [(a.start_time - 0.2, a.end_time + 0.2) for a in items]
            if durations:
                want = int(round(len(durations) * negatives_per_positive))
                dur = float(np.median(durations))
            else:  # no actions in this video: sample background windows across its tracks
                total = sum(tp.times[-1] - tp.times[0] for tp in vp.tracks.values() if len(tp.frames) > 1)
                want = int(min(40, max(4, total / max(global_dur, 0.1) / 2)))
                dur = global_dur
            tracks = list(used) or [t for t, tp in vp.tracks.items() if len(tp.frames) > 5]
            if not tracks:
                continue
            attempts = 0
            got = 0
            while got < want and attempts < want * 40:
                attempts += 1
                tp = vp.tracks[rng.choice(tracks)]
                if len(tp.frames) < 3:
                    continue
                lo, hi = tp.times[0], tp.times[-1] - dur
                if hi <= lo:
                    continue
                s = rng.uniform(lo, hi)
                e = s + dur
                if any(not (e < a or s > b) for a, b in all_marks):
                    continue
                if any(s < c / vp.fps < e for c in vp.scene_cuts):
                    continue
                res = window_frames(tp, s, e, cfg)
                if res is None or res[1]["visible_frac"] < 0.3:
                    continue
                samples.append(Sample(res[0], BACKGROUND, group, vid, split, "negative", s, e))
                got += 1
            if got < want:
                warnings.append(f"Video {vid}: only {got}/{want} background windows could be sampled")

    real = [s for s in samples if not s.augmented]
    counts = Counter(s.label for s in real)
    groups_per_class = {c: len({s.group for s in real if s.label == c}) for c in counts}
    version = dataset_version(anns, list(metas.values()), cfg, vision_ids)
    summary = {
        "version": version,
        "samples": len(real),
        "augmented_copies": len(samples) - len(real),
        "class_counts": dict(counts),
        "groups_per_class": groups_per_class,
        "groups": len({s.group for s in real}),
        "videos": len({s.video_id for s in real}),
        "annotations_used": sum(1 for s in real if s.source != "negative"),
        "annotations_total": len(anns),
        "skipped": dict(skipped),
        "feature_config": cfg.as_dict(),
    }
    return Dataset(samples, version, summary, warnings)
