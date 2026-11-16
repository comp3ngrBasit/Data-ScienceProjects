"""Unit tests for movement maths on controlled pose sequences (no model involved)."""
from __future__ import annotations

import numpy as np

from app.vision.constants import KP
from app.vision.movement import TrackSequence, compute_pair_distances, compute_track_movement

FPS = 30.0
FRAMES = np.arange(0, 90, 3)  # analysed every 3rd frame -> 10 Hz


def standing_pose(cx: float = 500.0, top: float = 200.0) -> np.ndarray:
    """A simple upright skeleton: torso length 150 px."""
    k = np.zeros((17, 2), np.float32)
    k[KP["nose"]] = (cx, top)
    k[KP["left_eye"]] = (cx - 5, top - 5)
    k[KP["right_eye"]] = (cx + 5, top - 5)
    k[KP["left_ear"]] = (cx - 10, top)
    k[KP["right_ear"]] = (cx + 10, top)
    k[KP["left_shoulder"]] = (cx - 40, top + 50)
    k[KP["right_shoulder"]] = (cx + 40, top + 50)
    k[KP["left_elbow"]] = (cx - 50, top + 110)
    k[KP["right_elbow"]] = (cx + 50, top + 110)
    k[KP["left_wrist"]] = (cx - 40, top + 60)
    k[KP["right_wrist"]] = (cx + 40, top + 60)
    k[KP["left_hip"]] = (cx - 30, top + 200)
    k[KP["right_hip"]] = (cx + 30, top + 200)
    k[KP["left_knee"]] = (cx - 35, top + 300)
    k[KP["right_knee"]] = (cx + 35, top + 300)
    k[KP["left_ankle"]] = (cx - 40, top + 400)
    k[KP["right_ankle"]] = (cx + 40, top + 400)
    return k


def make_seq(track_id=1, frames=FRAMES, pose_fn=lambda i, f: standing_pose(), conf_fn=lambda i, f: np.full(17, 0.9)):
    s = TrackSequence(track_id)
    for i, f in enumerate(frames):
        k = pose_fn(i, f)
        s.add(int(f), np.array([k[:, 0].min() - 20, k[:, 1].min() - 20, k[:, 0].max() + 20, k[:, 1].max() + 20]),
              k, conf_fn(i, f).astype(np.float32), 0.9)
    return s


def run(seq, frames=FRAMES, window=1, gap=0):
    return compute_track_movement(seq, np.asarray(frames), FPS, 0.35, window, gap, 0.15)


def test_stationary_body_is_stationary_not_missing():
    df = run(make_seq())
    assert (df["state"].iloc[1:] == "stationary").all()
    assert np.allclose(df["center_speed"].iloc[1:], 0)
    assert np.allclose(df["torso_px"], 150)


def test_wrist_speed_in_torso_lengths_per_second():
    # right wrist moves 15 px per analysed frame (0.1 s) -> 150 px/s -> 1.0 torso/s
    def pose(i, f):
        k = standing_pose()
        k[KP["right_wrist"], 0] += 15 * i
        return k

    df = run(make_seq(pose_fn=pose))
    assert np.allclose(df["right_wrist_speed"].iloc[1:], 1.0, atol=1e-3)
    assert np.allclose(df["left_wrist_speed"].iloc[1:], 0.0)


def test_low_confidence_landmarks_are_missing_not_zero_motion():
    def conf(i, f):
        c = np.full(17, 0.9)
        if 10 <= i < 15:
            c[KP["right_wrist"]] = 0.1  # occluded wrist
        return c

    df = run(make_seq(conf_fn=conf))
    assert df.loc[10:15, "right_wrist_speed"].isna().all()  # unknown, not 0
    assert df.loc[2:9, "right_wrist_speed"].notna().all()


def test_whole_body_low_confidence_flagged():
    def conf(i, f):
        return np.full(17, 0.1 if i == 12 else 0.9)

    df = run(make_seq(conf_fn=conf))
    assert df.loc[12, "state"] == "low_confidence"
    assert df.loc[12, "valid_landmarks"] == 0


def test_missing_frames_are_reported_as_missing():
    frames = [f for f in FRAMES if not (30 <= f <= 45)]
    df = run(make_seq(frames=frames), frames=FRAMES)
    missing = df[df["frame_number"].between(30, 45)]
    assert (missing["state"] == "missing").all()
    assert missing["center_speed"].isna().all()


def test_short_gaps_interpolated_but_never_reported_for_missing_frames():
    frames = [f for f in FRAMES if f != 30]
    df = run(make_seq(frames=frames), frames=FRAMES, gap=2)
    assert df.loc[df["frame_number"] == 30, "state"].item() == "missing"
    # neighbours keep valid speeds thanks to interpolation
    assert df.loc[df["frame_number"] == 33, "center_speed"].notna().item()


def test_direction_and_angles():
    def pose(i, f):
        return standing_pose(cx=500 + 30 * i)  # walking right: 300 px/s = 2 torso/s

    df = run(make_seq(pose_fn=pose))
    assert np.allclose(df["center_speed"].iloc[1:], 2.0, atol=1e-3)
    assert np.allclose(df["direction_deg"].iloc[1:], 0.0, atol=1e-3)  # screen right
    assert (df["state"].iloc[1:] == "moving").all()
    assert np.allclose(df["torso_lean_deg"], 0.0, atol=1e-3)
    assert (df["left_knee_angle"] > 150).all()  # nearly straight leg


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)

    def pose(i, f):
        k = standing_pose()
        return k + rng.normal(0, 3, k.shape).astype(np.float32)

    raw = run(make_seq(pose_fn=pose), window=1)
    smooth = run(make_seq(pose_fn=pose), window=5)
    assert smooth["left_wrist_speed"].mean() < raw["left_wrist_speed"].mean()


def test_pair_distance_and_closing_speed():
    a = run(make_seq(1, pose_fn=lambda i, f: standing_pose(cx=300 + 15 * i)))
    b = run(make_seq(2, pose_fn=lambda i, f: standing_pose(cx=900 - 15 * i)))
    d = compute_pair_distances(a, b, 1080, FPS)
    assert len(d) == len(FRAMES)
    assert np.isclose(d["distance_torso"].iloc[0], 600 / 150)
    assert np.isclose(d["distance_frame"].iloc[0], 600 / 1080)
    # approaching at 2 x 150 px/s = 2 torso/s
    assert np.allclose(d["closing_speed"].iloc[1:20], 2.0, atol=1e-3)  # approaching until they meet at i=20
    assert np.allclose(d["closing_speed"].iloc[21:], -2.0, atol=1e-3)  # then moving apart
