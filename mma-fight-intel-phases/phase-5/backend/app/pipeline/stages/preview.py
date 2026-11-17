"""Browser-playable preview proxy.

Browsers cannot play AVI and often fail on MOV/HEVC. When enabled, those files
are transcoded once to an H.264/AAC MP4 (max 1280px wide). The original file is
never modified and remains the source for all analysis.
"""
from __future__ import annotations

import logging
import subprocess
import time

from app.core.config import get_settings
from app.pipeline.base import JobCancelled, ProcessingContext
from app.services.ffmpeg_tools import SUBPROCESS_FLAGS, get_ffmpeg_tools

log = logging.getLogger(__name__)

BROWSER_SAFE_PIX = {"yuv420p", "yuvj420p", None}


def needs_preview(ext: str, codec: str | None, pix_fmt: str | None) -> bool:
    if ext.lower() != ".mp4":
        return True
    return (codec or "").lower() not in ("h264", "avc1") or pix_fmt not in BROWSER_SAFE_PIX


class PreviewProxyStage:
    name = "preview"
    label = "Creating browser preview"
    weight = 0.3
    required = False

    def run(self, ctx: ProcessingContext) -> None:
        media = ctx.media
        assert media is not None
        out = ctx.storage.preview_path(ctx.video_id)
        ctx.storage.remove_file(out)

        if not ctx.settings.generate_preview_proxy:
            return
        if not needs_preview(ctx.file_ext, media.video_codec, media.pixel_format):
            return
        tools = get_ffmpeg_tools()
        if not tools.ffmpeg:
            ctx.warnings.append("Preview proxy skipped: FFmpeg not available (file may not play in the browser)")
            return

        tmp = out.with_suffix(".tmp.mp4")
        cmd = [
            tools.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(ctx.source_path),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-vf", "scale='min(1280,iw)':-2",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats",
            str(tmp),
        ]
        duration = media.duration_seconds or 0
        timeout = get_settings().preview_transcode_timeout_seconds
        started = time.monotonic()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=SUBPROCESS_FLAGS,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if ctx.is_cancelled():
                    proc.kill()
                    raise JobCancelled()
                if time.monotonic() - started > timeout:
                    proc.kill()
                    raise TimeoutError("preview transcode timed out")
                if line.startswith("out_time_us=") and duration > 0:
                    try:
                        us = int(line.split("=", 1)[1])
                        ctx.report(self.name, us / 1e6 / duration)
                    except ValueError:
                        pass
            proc.wait(timeout=30)
            stderr = proc.stderr.read() if proc.stderr else ""
            if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
                tail = stderr.strip().splitlines()[-1:] if stderr else []
                ctx.storage.remove_file(tmp)
                ctx.warnings.append(f"Preview transcode failed: {tail[0] if tail else 'unknown error'}")
                return
            tmp.replace(out)
            ctx.preview_path = out
        except JobCancelled:
            ctx.storage.remove_file(tmp)
            raise
        except Exception:
            ctx.storage.remove_file(tmp)
            raise
        finally:
            if proc.poll() is None:
                proc.kill()
            for stream in (proc.stdout, proc.stderr):
                if stream:
                    stream.close()
        ctx.report(self.name, 1.0)
