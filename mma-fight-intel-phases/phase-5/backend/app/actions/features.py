"""Pose-sequence features for action recognition (pure NumPy; reusable outside the app).

Input per track: analysed frame numbers, keypoints in *pixels* (n, 17, 2) and keypoint
confidences (n, 17). Output: per-frame feature matrix (T, F) for a time window, resampled
to a fixed sequence length, plus a summary vector for classical models.

Geometry is normalised so features are comparable across resolutions, zoom levels and
fighters: coordinates are relative to the hip centre and divided by torso length.
Low-confidence landmarks are treated as missing, interpolated inside the window when
possible, otherwise zero-filled with an explicit visibility channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.vision.constants import KP

CORE = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
CORE_IDX = [KP[n] for n in CORE]
LIMBS = {
    "left_arm": KP["left_wrist"], "right_arm": KP["right_wrist"],
    "left_leg": KP["left_ankle"], "right_leg": KP["right_ankle"],
}


@dataclass
class FeatureConfig:
    seq_len: int = 16  # frames per window after resampling
    context: float = 0.15  # seconds added before/after an annotated action
    kp_threshold: float = 0.3

    def as_dict(self) -> dict:
        return {"seq_len": self.seq_len, "context": self.context, "kp_threshold": self.kp_threshold}


@dataclass
class TrackPose:
    """Pose observations of one track at analysed frames."""

    track_id: int
    frames: np.ndarray  # (n,) int
    keypoints: np.ndarray  # (n, 17, 2) pixels
    conf: np.ndarray  # (n, 17)
    fps: float

    @property
    def times(self) -> np.ndarray:
        return self.frames / self.fps


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(np.sum(v1 * v2, axis=-1) / n, -1, 1)
    return np.arccos(cos) / np.pi  # 0..1


def _interp_missing(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs along axis 0 (per column); edges held. All-NaN columns stay NaN."""
    out = x.copy()
    flat = out.reshape(len(t), -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if ok.sum() == 0 or ok.all():
            continue
        flat[:, j] = np.interp(t, t[ok], col[ok])
    return flat.reshape(out.shape)


def window_frames(pose: TrackPose, t0: float, t1: float, cfg: FeatureConfig) -> tuple[np.ndarray, dict] | None:
    """Per-frame features for [t0, t1] resampled to cfg.seq_len. Returns (T, F) and quality info."""
    times = pose.times
    sel = (times >= t0 - 1e-6) & (times <= t1 + 1e-6)
    if sel.sum() < 2:
        return None
    t = times[sel]
    k = pose.keypoints[sel].astype(np.float64).copy()
    c = pose.conf[sel]
    vis = c >= cfg.kp_threshold
    k[~vis] = np.nan
    visible_frac = float(vis[:, CORE_IDX].mean())

    # Resample to a fixed number of steps across the window (so duration is normalised away,
    # velocities below are still in real units because we keep dt).
    T = cfg.seq_len
    tt = np.linspace(t0, t1, T)
    k = _interp_missing(k, t)
    kr = np.empty((T, 17, 2))
    for j in range(17):
        for d in range(2):
            col = k[:, j, d]
            kr[:, j, d] = np.interp(tt, t, col) if np.isfinite(col).all() else np.nan
    visr = np.stack([np.interp(tt, t, vis[:, j].astype(float)) for j in range(17)], axis=1)

    def J(n):
        return kr[:, KP[n], :]

    mid_sh = (J("left_shoulder") + J("right_shoulder")) / 2
    mid_hip = (J("left_hip") + J("right_hip")) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    torso_med = np.nanmedian(torso) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(torso_med) or torso_med < 1:
        return None  # can't normalise without a visible torso
    centre = np.where(np.isfinite(mid_hip), mid_hip, np.nanmean(mid_hip, axis=0))
    rel = (kr[:, CORE_IDX, :] - centre[:, None, :]) / torso_med  # (T, 13, 2)
    dt = (t1 - t0) / max(T - 1, 1)
    vel = np.diff(rel, axis=0, prepend=rel[:1]) / max(dt, 1e-3)  # torso lengths / s

    feats = [rel.reshape(T, -1), vel.reshape(T, -1)]
    angles = np.stack([
        _angle(J("left_shoulder"), J("left_elbow"), J("left_wrist")),
        _angle(J("right_shoulder"), J("right_elbow"), J("right_wrist")),
        _angle(J("left_hip"), J("left_knee"), J("left_ankle")),
        _angle(J("right_hip"), J("right_knee"), J("right_ankle")),
        _angle(mid_hip, J("left_shoulder"), J("left_elbow")),
        _angle(mid_hip, J("right_shoulder"), J("right_elbow")),
        _angle(mid_sh, J("left_hip"), J("left_knee")),
        _angle(mid_sh, J("right_hip"), J("right_knee")),
    ], axis=1)
    ext = np.stack([
        np.linalg.norm(J("left_wrist") - J("left_shoulder"), axis=1) / torso_med,
        np.linalg.norm(J("right_wrist") - J("right_shoulder"), axis=1) / torso_med,
        np.linalg.norm(J("left_ankle") - J("left_hip"), axis=1) / torso_med,
        np.linalg.norm(J("right_ankle") - J("right_hip"), axis=1) / torso_med,
        (J("left_shoulder")[:, 1] - J("left_wrist")[:, 1]) / torso_med,
        (J("right_shoulder")[:, 1] - J("right_wrist")[:, 1]) / torso_med,
        (mid_hip[:, 1] - J("left_ankle")[:, 1]) / torso_med,
        (mid_hip[:, 1] - J("right_ankle")[:, 1]) / torso_med,
    ], axis=1)
    feats += [angles, ext, visr[:, CORE_IDX]]
    X = np.concatenate(feats, axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X = np.clip(X, -50, 50)

    speeds = {name: float(np.nanmax(np.linalg.norm(np.diff(kr[:, idx, :], axis=0), axis=1)) / torso_med / max(dt, 1e-3))
              if np.isfinite(kr[:, idx, :]).all() else 0.0 for name, idx in LIMBS.items()}
    return X, {"visible_frac": visible_frac, "limb_speeds": speeds, "frames": int(sel.sum())}


FEATURE_DIM = 26 + 26 + 8 + 8 + 13


def summarise(X: np.ndarray) -> np.ndarray:
    """Window summary for classical models: mean, std, min, max, end-start, and peak position per feature."""
    T = X.shape[0]
    return np.concatenate([
        X.mean(0), X.std(0), X.min(0), X.max(0), X[-1] - X[0], np.abs(X).argmax(0) / max(T - 1, 1),
    ]).astype(np.float32)


def limb_speed_profile(pose: TrackPose, t0: float, t1: float, limb: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unsmoothed) speed of a limb end point over time, in torso lengths / s, for boundary refinement."""
    times = pose.times
    sel = (times >= t0) & (times <= t1)
    t = times[sel]
    k = pose.keypoints[sel].astype(float)
    c = pose.conf[sel]
    idx = LIMBS[limb]
    p = k[:, idx, :].copy()
    p[c[:, idx] < 0.3] = np.nan
    sh = (k[:, KP["left_shoulder"]] + k[:, KP["right_shoulder"]]) / 2
    hp = (k[:, KP["left_hip"]] + k[:, KP["right_hip"]]) / 2
    torso = np.nanmedian(np.linalg.norm(sh - hp, axis=1)) if len(t) else np.nan
    if len(t) < 2 or not np.isfinite(torso) or torso < 1:
        return t, np.zeros(len(t))
    sp = np.linalg.norm(np.diff(p, axis=0, prepend=p[:1]), axis=1) / torso / np.maximum(np.diff(t, prepend=t[0] - 1 / pose.fps), 1e-3)
    return t, np.nan_to_num(sp)
