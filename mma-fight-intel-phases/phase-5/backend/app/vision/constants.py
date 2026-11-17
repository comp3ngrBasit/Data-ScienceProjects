"""COCO-17 keypoint layout used by YOLO Pose, skeleton edges and track colours."""
from __future__ import annotations

KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Landmarks the spec requires; all 17 are stored, these are the ones movement analysis relies on.
CORE_LANDMARKS = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]
SKELETON_IDX = [(KP[a], KP[b]) for a, b in SKELETON]

# Track colours (RGB hex). The frontend uses the same list: colour = PALETTE[(track_id - 1) % len].
PALETTE = [
    "#ef4444",  # red
    "#38bdf8",  # sky
    "#facc15",  # yellow
    "#4ade80",  # green
    "#c084fc",  # purple
    "#fb923c",  # orange
    "#f472b6",  # pink
    "#2dd4bf",  # teal
    "#a3e635",  # lime
    "#818cf8",  # indigo
]


def track_color_bgr(track_id: int) -> tuple[int, int, int]:
    h = PALETTE[(max(track_id, 1) - 1) % len(PALETTE)].lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


AVAILABLE_MODELS = [
    "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose",
    "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
]
