"""Reusable person detection + multi-object tracking + pose estimation.

One YOLO *pose* model forward pass yields person boxes, confidences and 17
COCO keypoints per person; Ultralytics' ByteTrack / BoT-SORT then assigns
persistent track IDs, and keypoints stay aligned with each tracked box. This
module has no database or web dependencies so Phase 3 can reuse it directly::

    tracker = PoseTracker(settings, analysis_fps=10)
    for frame in frames:
        people = tracker.process(frame)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app.vision.runtime import DeviceChoice, load_pose_model, resolve_device
from app.vision.settings import VisionSettings

log = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    track_id: int
    confidence: float
    box: np.ndarray  # (4,) x1, y1, x2, y2 in pixels
    keypoints: np.ndarray  # (17, 2) pixels
    keypoint_conf: np.ndarray  # (17,)

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


def write_tracker_config(settings: VisionSettings, analysis_fps: float, out_dir: Path) -> Path:
    """Write a tracker YAML based on Ultralytics' defaults with our buffer/threshold overrides.

    ``track_buffer`` is counted in *processed* frames, so it is scaled by the analysis rate
    to mean "keep a lost track alive for N seconds" regardless of sampling.
    """
    from ultralytics.utils import ROOT

    base = yaml.safe_load((ROOT / "cfg" / "trackers" / f"{settings.tracker}.yaml").read_text(encoding="utf-8"))
    base["track_buffer"] = max(3, int(math.ceil(settings.track_buffer_seconds * max(analysis_fps, 1.0))))
    # Track thresholds follow the detection threshold so low-confidence noise doesn't spawn tracks.
    base["track_high_thresh"] = max(0.1, settings.detection_confidence)
    base["new_track_thresh"] = max(0.1, settings.detection_confidence)
    base["track_low_thresh"] = min(0.1, settings.detection_confidence * 0.5)
    if settings.tracker == "botsort":
        base["with_reid"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tracker_{settings.tracker}.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def _precision_kwargs(half: bool) -> dict:
    """FP16 on CUDA. Newer Ultralytics versions replaced ``half`` with ``quantize``."""
    if not half:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


class PoseTracker:
    def __init__(self, settings: VisionSettings, analysis_fps: float, work_dir: Path):
        self.settings = settings
        self.device: DeviceChoice = resolve_device(settings.device, settings.half_precision)
        if self.device.device == "cpu":
            import os

            import torch

            cores = os.cpu_count() or 1
            torch.set_num_threads(cores - 1 if cores >= 4 else cores)  # leave a core for the web server
        self.model = load_pose_model(settings.model)
        self.tracker_cfg = write_tracker_config(settings, analysis_fps, work_dir)
        self.frames_seen = 0
        self._precision_kwargs = _precision_kwargs(self.device.half)

    def process(self, frame: np.ndarray) -> list[TrackedPerson]:
        s = self.settings
        result = self.model.track(
            frame,
            persist=True,
            tracker=str(self.tracker_cfg),
            conf=s.detection_confidence,
            iou=s.iou_threshold,
            imgsz=s.image_size,
            classes=[0],
            device=self.device.device,
            verbose=False,
            **self._precision_kwargs,
        )[0]
        self.frames_seen += 1
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []  # nothing tracked yet (tracks need confirmation) or no people

        h = frame.shape[0]
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        kps = result.keypoints
        kxy = kps.xy.cpu().numpy() if kps is not None else np.zeros((len(ids), 17, 2))
        kcf = kps.conf.cpu().numpy() if kps is not None and kps.conf is not None else np.zeros((len(ids), 17))

        people = [
            TrackedPerson(int(ids[i]), float(conf[i]), xyxy[i].astype(np.float32), kxy[i].astype(np.float32), kcf[i].astype(np.float32))
            for i in range(len(ids))
        ]
        # Filter background people (crowd, cage-side staff far away) by apparent size.
        min_h = s.min_box_height * h
        people = [p for p in people if p.height >= min_h]
        people.sort(key=lambda p: p.height, reverse=True)
        return people[: s.max_people_per_frame]
