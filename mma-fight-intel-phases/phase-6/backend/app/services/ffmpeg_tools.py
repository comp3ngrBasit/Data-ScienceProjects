"""Locate FFmpeg / FFprobe binaries.

Resolution order:
1. Explicit ``MMA_FFMPEG_PATH`` / ``MMA_FFPROBE_PATH``
2. Binaries on PATH (e.g. installed with ``winget install Gyan.FFmpeg``)
3. The FFmpeg binary bundled by the ``imageio-ffmpeg`` pip package
   (this one has no ffprobe, so metadata falls back to parsing ``ffmpeg -i``)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import get_settings

log = logging.getLogger(__name__)

# Prevent console windows popping up for each subprocess on Windows.
SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str | None
    ffprobe: str | None
    source: str | None  # "configured" | "system" | "imageio-ffmpeg" | None


def _works(binary: str) -> bool:
    try:
        r = subprocess.run(
            [binary, "-version"], capture_output=True, timeout=15, creationflags=SUBPROCESS_FLAGS
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache
def get_ffmpeg_tools() -> FFmpegTools:
    s = get_settings()
    ffmpeg = ffprobe = None
    source = None

    if s.ffmpeg_path and _works(s.ffmpeg_path):
        ffmpeg, source = s.ffmpeg_path, "configured"
    if s.ffprobe_path and _works(s.ffprobe_path):
        ffprobe = s.ffprobe_path
        source = source or "configured"

    if ffmpeg is None:
        found = shutil.which("ffmpeg")
        if found and _works(found):
            ffmpeg, source = found, source or "system"
    if ffprobe is None:
        found = shutil.which("ffprobe")
        if found and _works(found):
            ffprobe = found
            source = source or "system"

    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and _works(exe):
                ffmpeg, source = exe, source or "imageio-ffmpeg"
        except Exception as exc:  # pragma: no cover - depends on environment
            log.warning("imageio-ffmpeg unavailable: %s", exc)

    if ffmpeg is None:
        log.warning("FFmpeg not found. Preview transcoding is disabled and metadata uses OpenCV only.")
    if ffprobe is None:
        log.info("ffprobe not found; metadata will be parsed from `ffmpeg -i` output when possible.")
    return FFmpegTools(ffmpeg=ffmpeg, ffprobe=ffprobe, source=source)
