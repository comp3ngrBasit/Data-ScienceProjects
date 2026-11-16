"""Drawing of boxes / track IDs / skeletons, and an H.264 writer that streams frames to FFmpeg."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.services.ffmpeg_tools import SUBPROCESS_FLAGS, get_ffmpeg_tools
from app.vision.constants import SKELETON_IDX, track_color_bgr
from app.vision.pose_tracker import TrackedPerson

log = logging.getLogger(__name__)


def draw_people(img: np.ndarray, people: list[TrackedPerson], scale: float, kp_thr: float, stale: bool = False) -> None:
    """Draw in place. ``scale`` maps original pixel coords to ``img``. ``stale`` = result held from an earlier frame."""
    thick = max(2, int(round(img.shape[0] / 360)))
    for p in people:
        color = track_color_bgr(p.track_id)
        x1, y1, x2, y2 = (p.box * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"ID {p.track_id}  {p.confidence:.2f}"
        fs = max(0.45, img.shape[0] / 1400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
        cv2.putText(img, label, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 15, 15), 1, cv2.LINE_AA)

        pts = p.keypoints * scale
        ok = p.keypoint_conf >= kp_thr
        for a, b in SKELETON_IDX:
            if ok[a] and ok[b]:
                cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, thick, cv2.LINE_AA)
        for i in range(len(pts)):
            if ok[i]:
                c = tuple(pts[i].astype(int))
                cv2.circle(img, c, thick + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, thick + 1, color, -1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, frame_no: int, ts: float, n_people: int, analysed: bool) -> None:
    txt = f"frame {frame_no}  {ts:7.2f}s  people {n_people}" + ("" if analysed else "  (held)")
    cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


class AnnotatedVideoWriter:
    """Pipes raw BGR frames into FFmpeg -> browser-playable H.264 MP4 at the source frame rate."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        tools = get_ffmpeg_tools()
        if not tools.ffmpeg:
            raise RuntimeError("FFmpeg is required to write the annotated video")
        self.width, self.height = width - width % 2, height - height % 2
        self.path = path
        self.tmp = path.with_suffix(".part.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            tools.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.tmp),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=SUBPROCESS_FLAGS)
        self.frames = 0

    def write(self, img: np.ndarray) -> None:
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.frames += 1

    def close(self) -> Path:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
        code = self.proc.wait(timeout=600)
        if code != 0 or not self.tmp.exists():
            raise RuntimeError(f"FFmpeg failed writing annotated video: {err.strip()[-300:]}")
        self.tmp.replace(self.path)
        return self.path

    def abort(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait(timeout=30)
        if self.tmp.exists():
            try:
                self.tmp.unlink()
            except OSError:
                pass
