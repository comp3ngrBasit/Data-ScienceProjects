"""Phase 2 analysis of one video: detection -> tracking -> pose -> movement -> annotated video.

Streams the video frame by frame (never loads it into memory), runs the pose
model on every N-th frame (configurable analysis rate), writes results to the
database in batches, and pipes an annotated copy to FFmpeg.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sqlalchemy import insert

from app.core.config import get_settings
from app.db.database import session_scope
from app.db.vision_models import FighterDetection, FighterDistance, MovementMetric, PoseLandmark, TrackingSession
from app.pipeline.base import JobCancelled
from app.vision.annotate import AnnotatedVideoWriter, draw_hud, draw_people
from app.vision.constants import CORE_LANDMARKS, KEYPOINT_NAMES, KP
from app.vision.movement import TrackSequence, compute_pair_distances, compute_track_movement
from app.vision.pose_tracker import PoseTracker, TrackedPerson
from app.vision.settings import VisionSettings

log = logging.getLogger(__name__)

FLUSH_EVERY = 150  # analysed frames per DB batch
SCENE_CUT_THRESHOLD = 0.55  # HSV histogram correlation below this between analysed frames = camera cut
MAX_CONSECUTIVE_READ_FAILURES = 30
CORE_IDX = [KP[n] for n in CORE_LANDMARKS]
MOVEMENT_COLUMNS = [c.name for c in MovementMetric.__table__.columns if c.name not in ("id", "job_id", "video_id")]


class AnalysisError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    video_id: str
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class AnalysisResult:
    stats: dict
    warnings: list[str] = field(default_factory=list)
    annotated_path: Path | None = None
    export_path: Path | None = None
    device_label: str = ""


def vision_dir(video_id: str, job_id: str) -> Path:
    return get_settings().data_dir / "vision" / video_id / job_id


class _BatchWriter:
    """Accumulates rows and bulk-inserts them (executemany) to avoid per-row inserts."""

    def __init__(self, job_id: str, video_id: str):
        self.job_id, self.video_id = job_id, video_id
        self.det: list[dict] = []
        self.pose: list[dict] = []
        self.totals = {"detections": 0, "landmarks": 0}

    def add(self, frame: int, ts: float, people: list[TrackedPerson], w: int, h: int) -> None:
        for p in people:
            x1, y1, x2, y2 = (float(v) for v in p.box)
            self.det.append(dict(job_id=self.job_id, video_id=self.video_id, frame_number=frame, timestamp=ts,
                                 track_id=p.track_id, x1=x1 / w, y1=y1 / h, x2=x2 / w, y2=y2 / h,
                                 confidence=round(p.confidence, 4)))
            for i, name in enumerate(KEYPOINT_NAMES):
                self.pose.append(dict(job_id=self.job_id, video_id=self.video_id, track_id=p.track_id,
                                      frame_number=frame, timestamp=ts, landmark_index=i, landmark=name,
                                      x=round(float(p.keypoints[i, 0]) / w, 5), y=round(float(p.keypoints[i, 1]) / h, 5),
                                      confidence=round(float(p.keypoint_conf[i]), 4)))

    def flush(self) -> None:
        if not self.det and not self.pose:
            return
        with session_scope() as db:
            if self.det:
                db.execute(insert(FighterDetection), self.det)
            if self.pose:
                db.execute(insert(PoseLandmark), self.pose)
        self.totals["detections"] += len(self.det)
        self.totals["landmarks"] += len(self.pose)
        self.det, self.pose = [], []


def _bulk(model, rows: list[dict], chunk: int = 5000) -> None:
    for i in range(0, len(rows), chunk):
        with session_scope() as db:
            db.execute(insert(model), rows[i : i + chunk])


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _suggest_fighters(ranked: list[dict], sequences: dict[int, TrackSequence]) -> set[int]:
    """Greedy suggestion of fighter tracks: at most two suggested tracks visible at the same time.

    Because IDs change at camera cuts and after long occlusions, one athlete is often split into
    several consecutive tracks; a track is accepted when, for most of its frames, fewer than two
    already-accepted tracks are present. Large, confidently detected, long tracks go first.
    This is only a suggestion — referees and corner staff can look identical to the heuristic.
    """
    if not ranked:
        return set()
    top = ranked[0]["fighter_score"]
    max_h = max(r["avg_box_height"] for r in ranked)
    occupancy: dict[int, int] = {}
    accepted: set[int] = set()
    for r in ranked:
        if r["fighter_score"] < 0.5 * top or r["avg_box_height"] < 0.5 * max_h:
            continue
        frames = sequences[r["track_id"]].frames
        crowded = sum(1 for f in frames if occupancy.get(f, 0) >= 2)
        if crowded > 0.2 * len(frames):
            continue
        accepted.add(r["track_id"])
        for f in frames:
            occupancy[f] = occupancy.get(f, 0) + 1
    return accepted


def analyze_video(
    job_id: str,
    info: VideoInfo,
    settings: VisionSettings,
    progress: Callable[[str, float], None],
    is_cancelled: Callable[[], bool],
) -> AnalysisResult:
    t_start = time.monotonic()
    warnings: list[str] = []
    out_dir = vision_dir(info.video_id, job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = info.fps if info.fps and info.fps > 0 else 30.0
    stride = 1 if settings.analysis_fps <= 0 or settings.analysis_fps >= fps else max(1, int(round(fps / settings.analysis_fps)))
    analysis_rate = fps / stride

    # ---- model
    progress("loading_model", 0.01)
    try:
        tracker = PoseTracker(settings, analysis_rate, out_dir)
    except Exception as exc:
        raise AnalysisError(f"Could not load pose model '{settings.model}': {exc}") from exc
    if tracker.device.warning:
        warnings.append(tracker.device.warning)
    log.info("Vision job %s: model=%s device=%s stride=%d (%.2f fps analysed)", job_id, settings.model,
             tracker.device.label, stride, analysis_rate)

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise AnalysisError("OpenCV could not open the video (missing, corrupted or unsupported)")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or info.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or info.height
    total = info.frame_count or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer: AnnotatedVideoWriter | None = None
    annotated_path = out_dir / "annotated.mp4"
    scale = 1.0
    if settings.generate_annotated_video:
        scale = min(1.0, settings.annotated_max_width / max(width, 1))
        try:
            writer = AnnotatedVideoWriter(annotated_path, int(width * scale), int(height * scale), fps)
        except RuntimeError as exc:
            warnings.append(f"Annotated video disabled: {exc}")

    batch = _BatchWriter(job_id, info.video_id)
    sequences: dict[int, TrackSequence] = {}
    analysed_frames: list[int] = []
    missing_frames: list[int] = []
    people_per_frame: list[int] = []
    last_people: list[TrackedPerson] = []
    last_analysed = -10**9
    last_annot: np.ndarray | None = None
    infer_seconds = 0.0
    scene_cuts: list[int] = []
    prev_hist: np.ndarray | None = None
    idx = 0
    consecutive_fail = 0
    progress("analyzing", 0.03)

    try:
        while True:
            if is_cancelled():
                raise JobCancelled()
            analyse = idx % stride == 0
            if not analyse and writer is None:
                ok = cap.grab()  # skip decode work we don't need
                frame = None
            else:
                ok, frame = cap.read()

            if not ok:
                if total and idx < total - 1 and consecutive_fail < MAX_CONSECUTIVE_READ_FAILURES:
                    consecutive_fail += 1
                    missing_frames.append(idx)
                    if writer is not None and last_annot is not None:
                        writer.write(last_annot)  # keep the annotated video in sync with the original
                    idx += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    continue
                break
            consecutive_fail = 0
            ts = round(idx / fps, 4)

            if analyse:
                t0 = time.monotonic()
                people = tracker.process(frame)
                infer_seconds += time.monotonic() - t0
                analysed_frames.append(idx)
                people_per_frame.append(len(people))
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < SCENE_CUT_THRESHOLD:
                    scene_cuts.append(idx)  # camera cut: track IDs are expected to change here
                prev_hist = hist
                batch.add(idx, ts, people, width, height)
                for p in people:
                    sequences.setdefault(p.track_id, TrackSequence(p.track_id)).add(
                        idx, p.box, p.keypoints, p.keypoint_conf, p.confidence)
                last_people, last_analysed = people, idx
                if len(analysed_frames) % FLUSH_EVERY == 0:
                    batch.flush()

            if writer is not None and frame is not None:
                img = cv2.resize(frame, (writer.width, writer.height), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
                held = idx - last_analysed < stride
                if held:
                    draw_people(img, last_people, scale, settings.keypoint_confidence, stale=not analyse)
                draw_hud(img, idx, ts, len(last_people) if held else 0, analyse)
                writer.write(img)
                last_annot = img

            idx += 1
            if total and idx % 10 == 0:
                progress("analyzing", 0.03 + 0.82 * min(1.0, idx / total))
        batch.flush()
    except BaseException:
        cap.release()
        if writer is not None:
            writer.abort()
        raise
    cap.release()

    if not analysed_frames:
        if writer is not None:
            writer.abort()
        raise AnalysisError("No frames could be decoded — the video may be corrupted")
    if missing_frames:
        warnings.append(f"{len(missing_frames)} frame(s) could not be decoded and were skipped")
    if scene_cuts:
        warnings.append(f"{len(scene_cuts)} camera cut(s) detected — tracking IDs usually change across cuts; "
                        "assign every track of the same athlete to their fighter profile")

    if writer is not None:
        progress("encoding", 0.86)
        try:
            writer.close()
        except RuntimeError as exc:
            warnings.append(str(exc))
            writer = None

    # ---- tracking sessions
    progress("tracking_summary", 0.88)
    af = np.asarray(analysed_frames)
    min_frames = max(3, int(round(0.5 * analysis_rate)))
    track_rows: list[dict] = []
    for tid, seq in sequences.items():
        pos = np.searchsorted(af, seq.frames)
        gaps = np.where(np.diff(pos) > 1)[0]
        segments, start = [], 0
        for g in gaps:
            segments.append([seq.frames[start], seq.frames[g]])
            start = g + 1
        segments.append([seq.frames[start], seq.frames[-1]])
        span = int(pos[-1] - pos[0] + 1)
        kc = np.asarray(seq.keypoint_conf)[:, CORE_IDX]
        heights = [(b[3] - b[1]) / height for b in seq.boxes]
        n = len(seq.frames)
        track_rows.append(dict(
            track_id=tid, first_frame=seq.frames[0], last_frame=seq.frames[-1],
            first_timestamp=round(seq.frames[0] / fps, 4), last_timestamp=round(seq.frames[-1] / fps, 4),
            frames_detected=n, coverage=round(n / span, 4), video_coverage=round(n / len(af), 4),
            interruptions=len(gaps), segments=segments,
            avg_confidence=round(float(np.mean(seq.det_conf)), 4), avg_box_height=round(float(np.mean(heights)), 4),
            avg_pose_confidence=round(float(kc.mean()), 4),
            status="fragment" if n < min_frames else ("interrupted" if len(gaps) else "continuous"),
        ))
    # Fighter likelihood: present most of the time, large on screen, confidently detected. A suggestion only —
    # a referee can score as high as a fighter, so the user confirms roles manually.
    max_h = max((r["avg_box_height"] for r in track_rows), default=1.0) or 1.0
    for r in track_rows:
        r["fighter_score"] = round(0.45 * r["video_coverage"] + 0.35 * (r["avg_box_height"] / max_h) + 0.2 * r["avg_confidence"], 4)
    ranked = sorted((r for r in track_rows if r["status"] != "fragment"), key=lambda r: r["fighter_score"], reverse=True)
    suggested = _suggest_fighters(ranked, sequences)
    for r in track_rows:
        r["suggested_role"] = "fighter" if r["track_id"] in suggested else "other"
    if track_rows:
        with session_scope() as db:
            db.execute(insert(TrackingSession), [
                dict(job_id=job_id, video_id=info.video_id, role="unreviewed", reviewed=False, **r) for r in track_rows
            ])

    # ---- movement
    progress("movement", 0.90)
    movement_frames: dict[int, pd.DataFrame] = {}
    mov_rows: list[dict] = []
    for tid, seq in sequences.items():
        if is_cancelled():
            raise JobCancelled()
        if len(seq.frames) < 2:
            continue
        df = compute_track_movement(seq, af, fps, settings.keypoint_confidence, settings.smoothing_window,
                                    settings.max_gap_interpolation, settings.stationary_threshold)
        movement_frames[tid] = df
        for rec in df.to_dict("records"):
            mov_rows.append({"job_id": job_id, "video_id": info.video_id,
                             **{k: _clean(rec.get(k)) for k in MOVEMENT_COLUMNS}})
    _bulk(MovementMetric, mov_rows)

    candidates = [r["track_id"] for r in ranked[:4]]
    dist_rows: list[dict] = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if a not in movement_frames or b not in movement_frames:
                continue
            ta, tb = sorted((a, b))
            d = compute_pair_distances(movement_frames[ta], movement_frames[tb], height, fps)
            for rec in d.to_dict("records"):
                dist_rows.append({"job_id": job_id, "video_id": info.video_id, "track_a": ta, "track_b": tb,
                                  **{k: _clean(v) for k, v in rec.items()}})
    _bulk(FighterDistance, dist_rows)

    # ---- export bundle for Phase 3 (fast NumPy loading without the database)
    progress("export", 0.97)
    export_path = out_dir / "pose_tracks.npz"
    arrays: dict[str, np.ndarray] = {}
    for tid, seq in sequences.items():
        arrays[f"t{tid}_frames"] = np.asarray(seq.frames, np.int32)
        arrays[f"t{tid}_keypoints"] = (np.asarray(seq.keypoints) / np.array([width, height])).astype(np.float32)
        arrays[f"t{tid}_keypoint_conf"] = np.asarray(seq.keypoint_conf, np.float32)
        arrays[f"t{tid}_boxes"] = (np.asarray(seq.boxes) / np.array([width, height, width, height])).astype(np.float32)
        arrays[f"t{tid}_det_conf"] = np.asarray(seq.det_conf, np.float32)
    meta = {"video_id": info.video_id, "job_id": job_id, "fps": fps, "width": width, "height": height,
            "keypoint_names": KEYPOINT_NAMES, "coordinates": "normalised [0,1] by frame width/height",
            "analysed_frames": len(af), "stride": stride, "track_ids": sorted(sequences)}
    np.savez_compressed(export_path, meta=np.array(json.dumps(meta)), analysed_frames=af.astype(np.int32), **arrays)

    det_conf = [c for s in sequences.values() for c in s.det_conf]
    pose_conf = [float(np.mean(np.asarray(s.keypoint_conf)[:, CORE_IDX])) for s in sequences.values() if s.frames]
    elapsed = time.monotonic() - t_start
    fighter_tracks = [r for r in track_rows if r["track_id"] in suggested]
    stats = {
        "frames_total": total or idx,
        "frames_read": idx,
        "frames_analysed": len(af),
        "frames_missing": len(missing_frames),
        "stride": stride,
        "analysis_fps": round(analysis_rate, 3),
        "frames_with_people": int(sum(1 for c in people_per_frame if c > 0)),
        "max_people_in_frame": int(max(people_per_frame) if people_per_frame else 0),
        "detections": batch.totals["detections"],
        "landmarks": batch.totals["landmarks"],
        "tracks": len(track_rows),
        "tracks_non_fragment": len(ranked),
        "fragment_tracks": len(track_rows) - len(ranked),
        "suggested_fighter_tracks": sorted(suggested),
        "avg_detection_confidence": round(float(np.mean(det_conf)), 4) if det_conf else None,
        "avg_pose_confidence": round(float(np.mean(pose_conf)), 4) if pose_conf else None,
        "fighter_track_coverage": round(float(np.mean([r["coverage"] for r in fighter_tracks])), 4) if fighter_tracks else None,
        "fighter_track_interruptions": int(sum(r["interruptions"] for r in fighter_tracks)),
        "scene_cuts": scene_cuts,
        "movement_rows": len(mov_rows),
        "distance_rows": len(dist_rows),
        "processing_seconds": round(elapsed, 2),
        "inference_ms_per_frame": round(1000 * infer_seconds / len(af), 1),
        "realtime_factor": round((idx / fps) / elapsed, 3) if elapsed > 0 else None,
        "model": settings.model,
        "device": tracker.device.label,
        "tracker": settings.tracker,
        "width": width,
        "height": height,
        "fps": fps,
    }
    progress("finalizing", 0.99)
    return AnalysisResult(stats=stats, warnings=warnings,
                          annotated_path=annotated_path if writer is not None and annotated_path.exists() else None,
                          export_path=export_path, device_label=tracker.device.label)
