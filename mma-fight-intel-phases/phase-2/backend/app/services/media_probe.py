"""Video metadata extraction using FFprobe/FFmpeg and OpenCV.

The two sources are merged: FFprobe is authoritative for container/codec/audio
information, OpenCV verifies that frames can actually be decoded and provides
frame count / fps / resolution as a fallback.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

from app.services.ffmpeg_tools import SUBPROCESS_FLAGS, get_ffmpeg_tools

log = logging.getLogger(__name__)


class ProbeError(RuntimeError):
    """The file could not be read as a video."""


@dataclass
class MediaInfo:
    duration_seconds: float | None = None
    frame_count: int | None = None
    frame_count_source: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    rotation: int | None = None
    video_codec: str | None = None
    pixel_format: str | None = None
    container_format: str | None = None
    bitrate_kbps: float | None = None
    has_audio: bool | None = None
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    sources: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_rate(value: str | None) -> float | None:
    if not value or value in ("0/0", "0"):
        return None
    try:
        f = float(Fraction(value))
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _container_name(fmt: dict[str, Any]) -> str | None:
    """FFmpeg reports MP4 and MOV with the same demuxer; use the brand to tell them apart."""
    name = fmt.get("format_name") or ""
    if "mp4" in name and "mov" in name:
        brand = ((fmt.get("tags") or {}).get("major_brand") or "").strip().lower()
        if brand == "qt":
            return "QuickTime (MOV)"
        if brand:
            return f"MPEG-4 (MP4, brand {brand})"
        return "MP4 / QuickTime"
    return fmt.get("format_long_name") or name or None


# ---------------------------------------------------------------------------
# FFprobe
# ---------------------------------------------------------------------------
def probe_ffprobe(path: Path, ffprobe: str) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=SUBPROCESS_FLAGS)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = (msg[-1] if msg else "unknown error").replace(str(path), "file").replace(path.name, "file")
        raise ProbeError(f"FFprobe could not read the file ({detail.split(': ', 1)[-1]})")
    try:
        return json.loads(r.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON: {exc}") from exc


def apply_ffprobe(info: MediaInfo, data: dict[str, Any]) -> None:
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if v is None:
        raise ProbeError("No video stream found in file")

    info.sources.append("ffprobe")
    info.container_format = _container_name(fmt)
    info.duration_seconds = _to_float(fmt.get("duration")) or _to_float(v.get("duration"))
    br = _to_float(fmt.get("bit_rate"))
    info.bitrate_kbps = round(br / 1000, 1) if br else None

    info.video_codec = v.get("codec_name")
    info.pixel_format = v.get("pix_fmt")
    info.width = _to_int(v.get("width"))
    info.height = _to_int(v.get("height"))
    info.fps = _parse_rate(v.get("avg_frame_rate")) or _parse_rate(v.get("r_frame_rate"))
    nb = _to_int(v.get("nb_frames"))
    if nb and nb > 0:
        info.frame_count, info.frame_count_source = nb, "container"

    rotation = None
    tags = v.get("tags") or {}
    if "rotate" in tags:
        rotation = _to_int(tags.get("rotate"))
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = _to_int(sd.get("rotation"))
    info.rotation = (rotation % 360) if rotation else 0

    info.has_audio = a is not None
    if a is not None:
        info.audio_codec = a.get("codec_name")
        info.audio_sample_rate = _to_int(a.get("sample_rate"))
        info.audio_channels = _to_int(a.get("channels"))

    info.raw["ffprobe"] = {
        "format": {k: fmt.get(k) for k in ("format_name", "format_long_name", "duration", "bit_rate", "nb_streams", "size")},
        "format_tags": fmt.get("tags") or {},
        "video_stream": {k: v.get(k) for k in (
            "codec_name", "codec_long_name", "profile", "level", "pix_fmt", "width", "height",
            "avg_frame_rate", "r_frame_rate", "nb_frames", "bit_rate", "color_space", "field_order",
        )},
        "audio_stream": None if a is None else {k: a.get(k) for k in (
            "codec_name", "codec_long_name", "sample_rate", "channels", "channel_layout", "bit_rate",
        )},
        "stream_count": len(streams),
    }


# ---------------------------------------------------------------------------
# `ffmpeg -i` fallback (used with the imageio-ffmpeg binary, which has no ffprobe)
# ---------------------------------------------------------------------------
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_BITRATE_RE = re.compile(r"Duration:.*bitrate:\s*(\d+)\s*kb/s")
_INPUT_RE = re.compile(r"Input #0,\s*([^,]+(?:,[^,]+)*?),\s*from")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?: Video:\s*([a-zA-Z0-9_]+)([^\n]*)")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:\s*([a-zA-Z0-9_]+)([^\n]*)")
_RES_RE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_HZ_RE = re.compile(r"(\d+)\s*Hz")
_PIX_RE = re.compile(r"\b(yuv\w+|rgb\w+|bgr\w+|gray\w*|nv12|nv21)\b")


def probe_ffmpeg_banner(path: Path, ffmpeg: str) -> str:
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True, timeout=120, creationflags=SUBPROCESS_FLAGS,
    )
    # ffmpeg exits non-zero because no output is given; the info is on stderr.
    return r.stderr.decode("utf-8", "replace")


def apply_ffmpeg_banner(info: MediaInfo, text: str) -> None:
    vm = _VIDEO_RE.search(text)
    if not vm:
        if "Invalid data found" in text or "moov atom not found" in text:
            raise ProbeError("FFmpeg reports invalid or truncated video data")
        raise ProbeError("No video stream found in file")
    info.sources.append("ffmpeg")
    if m := _DURATION_RE.search(text):
        h, mi, se = m.groups()
        info.duration_seconds = int(h) * 3600 + int(mi) * 60 + float(se)
    if m := _BITRATE_RE.search(text):
        info.bitrate_kbps = float(m.group(1))
    if m := _INPUT_RE.search(text):
        info.container_format = m.group(1).strip()
    info.video_codec = vm.group(1)
    details = vm.group(2)
    if m := _RES_RE.search(details):
        info.width, info.height = int(m.group(1)), int(m.group(2))
    if m := _FPS_RE.search(details):
        info.fps = float(m.group(1))
    if m := _PIX_RE.search(details):
        info.pixel_format = m.group(1)
    am = _AUDIO_RE.search(text)
    info.has_audio = am is not None
    if am:
        info.audio_codec = am.group(1)
        if m := _HZ_RE.search(am.group(2)):
            info.audio_sample_rate = int(m.group(1))
        ad = am.group(2)
        info.audio_channels = 2 if "stereo" in ad else 1 if "mono" in ad else 6 if "5.1" in ad else None
    rot = re.search(r"rotate\s*:\s*(-?\d+)", text) or re.search(r"rotation of (-?\d+(?:\.\d+)?)", text)
    info.rotation = int(float(rot.group(1))) % 360 if rot else 0
    info.raw["ffmpeg_banner"] = text[-4000:]


# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
def _fourcc_to_str(v: float) -> str | None:
    code = int(v)
    if code <= 0:
        return None
    s = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    s = s.strip("\x00 ").strip()
    return s if s.isprintable() and s else None


def apply_opencv(info: MediaInfo, path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ProbeError("OpenCV could not open the video (corrupted file or unsupported codec)")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ProbeError("OpenCV could not decode any frame (corrupted file or unsupported codec)")
        info.sources.append("opencv")
        cv_fps = cap.get(cv2.CAP_PROP_FPS) or None
        cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cv_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        cv_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        info.raw["opencv"] = {
            "fps": cv_fps,
            "frame_count": cv_frames,
            "width": cv_w,
            "height": cv_h,
            "fourcc": fourcc,
            "decoded_frame_shape": list(frame.shape),
            "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else None,
        }
        if info.fps is None and cv_fps and 0 < cv_fps < 1000:
            info.fps = float(cv_fps)
        if not info.width or not info.height:
            info.width, info.height = cv_w or frame.shape[1], cv_h or frame.shape[0]
        if info.frame_count is None and cv_frames > 0:
            info.frame_count, info.frame_count_source = cv_frames, "opencv"
        if info.video_codec is None and fourcc:
            info.video_codec = fourcc.lower()
        if info.duration_seconds is None and cv_frames > 0 and info.fps:
            info.duration_seconds = cv_frames / info.fps
    finally:
        cap.release()


def probe_media(path: Path) -> MediaInfo:
    """Extract metadata. Raises ProbeError if the file is not a decodable video."""
    if not path.exists():
        raise ProbeError("Stored video file is missing")
    if path.stat().st_size == 0:
        raise ProbeError("Video file is empty")

    info = MediaInfo()
    tools = get_ffmpeg_tools()
    ff_error: str | None = None
    try:
        if tools.ffprobe:
            apply_ffprobe(info, probe_ffprobe(path, tools.ffprobe))
        elif tools.ffmpeg:
            apply_ffmpeg_banner(info, probe_ffmpeg_banner(path, tools.ffmpeg))
        else:
            info.warnings.append("FFmpeg not available: codec/audio details limited to OpenCV")
    except ProbeError as exc:
        ff_error = str(exc)
    except subprocess.TimeoutExpired:
        ff_error = "FFmpeg metadata probe timed out"

    try:
        apply_opencv(info, path)
    except ProbeError as exc:
        # Both sources failing (or OpenCV unable to decode) means the file is unusable.
        detail = f"{exc}" + (f"; {ff_error}" if ff_error else "")
        raise ProbeError(detail) from exc

    if ff_error:
        info.warnings.append(f"FFmpeg probe failed, used OpenCV only: {ff_error}")

    # Estimate frame count if neither container nor OpenCV knew it.
    if info.frame_count is None and info.duration_seconds and info.fps:
        info.frame_count = int(round(info.duration_seconds * info.fps))
        info.frame_count_source = "estimated"

    if info.duration_seconds is None or info.duration_seconds <= 0:
        info.warnings.append("Duration could not be determined")
    return info
