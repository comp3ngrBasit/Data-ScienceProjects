"""Movement feature extraction from tracked pose sequences (NumPy + Pandas).

Pure functions with no database dependency so Phase 3 (action recognition)
can call them on any pose sequence.

Conventions
-----------
* Input keypoints are in **pixels** of the source video; conversions keep the
  geometry correct for non-square frames.
* Speeds are expressed in **torso lengths per second** (shoulder-midpoint to
  hip-midpoint distance), which makes them comparable across resolutions,
  zoom levels and camera distance.
* Landmarks below the keypoint-confidence threshold are treated as *missing*
  (NaN), never as zero movement. Short gaps (<= ``max_gap`` analysed frames)
  are linearly interpolated; longer gaps stay NaN.
* Smoothing is a centred rolling mean; values are not smoothed *into* gaps.
* ``state`` separates the cases the spec asks to distinguish:
  ``missing`` (person not detected in that analysed frame),
  ``low_confidence`` (detected but fewer than half the core landmarks visible),
  ``stationary`` (visible and moving less than the threshold) and ``moving``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.vision.constants import KP

CORE_IDX = [KP[n] for n in (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)]
SPEED_JOINTS = [
    "left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]
MIN_VALID_CORE = 7  # of 13


@dataclass
class TrackSequence:
    """Raw per-track observations at analysed frames (only frames where the track was detected)."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)  # (4,) pixels
    keypoints: list[np.ndarray] = field(default_factory=list)  # (17, 2) pixels
    keypoint_conf: list[np.ndarray] = field(default_factory=list)  # (17,)
    det_conf: list[float] = field(default_factory=list)

    def add(self, frame: int, box: np.ndarray, kps: np.ndarray, kconf: np.ndarray, conf: float) -> None:
        self.frames.append(frame)
        self.boxes.append(box)
        self.keypoints.append(kps)
        self.keypoint_conf.append(kconf)
        self.det_conf.append(conf)


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    a = np.where(np.isfinite(a), a, np.nan)
    cnt = np.isfinite(a).sum(axis=1)
    return np.where(cnt > 0, np.nansum(a, axis=1) / np.maximum(cnt, 1), np.nan)


def _nanmedian_rows(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    for i, row in enumerate(a):
        row = row[np.isfinite(row)]
        if row.size:
            out[i] = float(np.median(row))
    return out


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of shape (n, 2)."""
    v1, v2 = a - b, c - b
    dot = np.einsum("ij,ij->i", v1, v2)
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / n, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_track_movement(
    seq: TrackSequence,
    analysed_frames: np.ndarray,
    fps: float,
    kp_threshold: float,
    smoothing_window: int,
    max_gap: int,
    stationary_threshold: float,
) -> pd.DataFrame:
    """Return one row per analysed frame in the track's lifespan (including frames where it was missing)."""
    if not seq.frames:
        return pd.DataFrame()
    frames = np.asarray(seq.frames)
    span = analysed_frames[(analysed_frames >= frames.min()) & (analysed_frames <= frames.max())]
    n = len(span)
    pos = {f: i for i, f in enumerate(span)}
    present = np.zeros(n, bool)
    xy = np.full((n, 17, 2), np.nan)
    kc = np.zeros((n, 17))
    box = np.full((n, 4), np.nan)
    for f, b, k, c in zip(seq.frames, seq.boxes, seq.keypoints, seq.keypoint_conf, strict=True):
        i = pos.get(f)
        if i is None:
            continue
        present[i] = True
        box[i] = b
        kc[i] = c
        xy[i] = k

    raw_valid = kc >= kp_threshold
    xy[~raw_valid] = np.nan  # low-confidence landmark = missing, not zero

    # --- interpolate short gaps, then smooth (without bleeding into long gaps)
    cols = {}
    for j in range(17):
        cols[f"x{j}"], cols[f"y{j}"] = xy[:, j, 0], xy[:, j, 1]
    df = pd.DataFrame(cols, index=span)
    if max_gap > 0:
        df = df.interpolate(method="index", limit=max_gap, limit_area="inside")
    filled_mask = df.notna()
    if smoothing_window > 1:
        df = df.rolling(smoothing_window, center=True, min_periods=1).mean().where(filled_mask)
    sm = np.stack([np.stack([df[f"x{j}"].to_numpy(), df[f"y{j}"].to_numpy()], axis=1) for j in range(17)], axis=1)

    bdf = pd.DataFrame(box, index=span)
    if max_gap > 0:
        bdf = bdf.interpolate(method="index", limit=max_gap, limit_area="inside")
    box_s = bdf.to_numpy()

    def J(name: str) -> np.ndarray:
        return sm[:, KP[name], :]

    mid_sh = (J("left_shoulder") + J("right_shoulder")) / 2
    mid_hip = (J("left_hip") + J("right_hip")) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    box_h = box_s[:, 3] - box_s[:, 1]
    torso = pd.Series(torso, index=span).rolling(15, center=True, min_periods=1).median().to_numpy()
    torso = np.where(np.isfinite(torso) & (torso > 1), torso, 0.3 * box_h)  # fallback when shoulders/hips hidden

    box_c = np.stack([(box_s[:, 0] + box_s[:, 2]) / 2, (box_s[:, 1] + box_s[:, 3]) / 2], axis=1)
    center = np.where(np.isfinite(mid_hip).all(axis=1, keepdims=True), mid_hip, box_c)
    center = pd.DataFrame(center, index=span).rolling(smoothing_window, center=True, min_periods=1).mean().where(
        pd.DataFrame(np.isfinite(center), index=span)
    ).to_numpy()

    t = span / fps
    dt = np.diff(t, prepend=np.nan)

    def speed(points: np.ndarray) -> np.ndarray:
        d = np.diff(points, axis=0, prepend=np.full((1, 2), np.nan))
        return np.linalg.norm(d, axis=1) / dt / torso

    out = pd.DataFrame(index=span)
    out["frame_number"] = span
    out["timestamp"] = np.round(t, 4)
    out["present"] = present
    out["valid_landmarks"] = raw_valid[:, CORE_IDX].sum(axis=1)
    out["center_x"] = center[:, 0]
    out["center_y"] = center[:, 1]
    out["torso_px"] = torso
    cs = speed(center)
    out["center_speed"] = cs
    dvec = np.diff(center, axis=0, prepend=np.full((1, 2), np.nan))
    direction = (np.degrees(np.arctan2(-dvec[:, 1], dvec[:, 0])) + 360) % 360
    out["direction_deg"] = np.where(cs > stationary_threshold, direction, np.nan)
    for name in SPEED_JOINTS:
        out[f"{name}_speed"] = speed(J(name))
    out["left_elbow_angle"] = _angle(J("left_shoulder"), J("left_elbow"), J("left_wrist"))
    out["right_elbow_angle"] = _angle(J("right_shoulder"), J("right_elbow"), J("right_wrist"))
    out["left_knee_angle"] = _angle(J("left_hip"), J("left_knee"), J("left_ankle"))
    out["right_knee_angle"] = _angle(J("right_hip"), J("right_knee"), J("right_ankle"))
    tv = mid_sh - mid_hip
    out["torso_lean_deg"] = np.degrees(np.arctan2(tv[:, 0], -tv[:, 1]))
    out["stance_width"] = np.linalg.norm(J("left_ankle") - J("right_ankle"), axis=1) / torso
    guard = np.stack([
        (J("left_shoulder")[:, 1] - J("left_wrist")[:, 1]) / torso,
        (J("right_shoulder")[:, 1] - J("right_wrist")[:, 1]) / torso,
    ], axis=1)
    out["guard_height"] = _nanmean_rows(guard)

    # Posture vector: core joints relative to hip centre, scaled by torso -> RMS change between frames
    rel = (sm[:, CORE_IDX, :] - mid_hip[:, None, :]) / torso[:, None, None]
    prev = np.roll(rel, 1, axis=0)
    prev[0] = np.nan
    diff = np.linalg.norm(rel - prev, axis=2)  # (n, 13)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(diff).sum(axis=1)
        pc = np.where(cnt >= 4, np.sqrt(np.nansum(np.where(np.isfinite(diff), diff, 0) ** 2, axis=1) / np.maximum(cnt, 1)), np.nan)
    out["posture_change"] = pc

    # --- state
    joint_speeds = out[[f"{n}_speed" for n in SPEED_JOINTS]].to_numpy()
    motion = np.fmax(cs, _nanmedian_rows(joint_speeds))
    state = np.where(
        ~present, "missing",
        np.where(out["valid_landmarks"].to_numpy() < MIN_VALID_CORE, "low_confidence",
                 np.where(~np.isfinite(motion), "low_confidence",
                          np.where(motion < stationary_threshold, "stationary", "moving"))),
    )
    out["state"] = state
    out["track_id"] = seq.track_id
    # Never report computed motion for frames where the person wasn't detected (only interpolated values).
    num_cols = [c for c in out.columns if c not in ("frame_number", "timestamp", "present", "state", "track_id", "valid_landmarks")]
    out.loc[~present, num_cols] = np.nan
    return out.reset_index(drop=True)


def compute_pair_distances(
    a: pd.DataFrame, b: pd.DataFrame, frame_height: int, fps: float
) -> pd.DataFrame:
    """Centre-to-centre distance between two tracks on the analysed frames where both are present."""
    if a.empty or b.empty:
        return pd.DataFrame()
    m = a[a["present"]][["frame_number", "timestamp", "center_x", "center_y", "torso_px"]].merge(
        b[b["present"]][["frame_number", "center_x", "center_y", "torso_px"]], on="frame_number", suffixes=("_a", "_b")
    )
    if m.empty:
        return pd.DataFrame()
    px = np.hypot(m["center_x_a"] - m["center_x_b"], m["center_y_a"] - m["center_y_b"])
    torso = (m["torso_px_a"] + m["torso_px_b"]) / 2
    out = pd.DataFrame({
        "frame_number": m["frame_number"],
        "timestamp": m["timestamp"],
        "distance_frame": px / frame_height,
        "distance_torso": px / torso,
    })
    dt = out["frame_number"].diff() / fps
    out["closing_speed"] = -(out["distance_torso"].diff() / dt)
    out.loc[dt > 1.0, "closing_speed"] = np.nan  # don't differentiate across long gaps
    return out.dropna(subset=["distance_frame"])
