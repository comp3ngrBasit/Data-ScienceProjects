"""The Windows-without-FFmpeg path: only the imageio-ffmpeg binary (no ffprobe) is available."""
from __future__ import annotations

import imageio_ffmpeg
import pytest

from app.services import media_probe
from app.services.ffmpeg_tools import FFmpegTools
from tests.conftest import make_ffmpeg_video, make_opencv_video


@pytest.fixture()
def bundled_only(monkeypatch):
    tools = FFmpegTools(ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(), ffprobe=None, source="imageio-ffmpeg")
    monkeypatch.setattr(media_probe, "get_ffmpeg_tools", lambda: tools)


def test_banner_probe_h264_audio(bundled_only, tmp_path):
    info = media_probe.probe_media(make_ffmpeg_video(tmp_path / "a.mp4", seconds=3, audio=True))
    assert info.sources == ["ffmpeg", "opencv"]
    assert info.video_codec == "h264" and info.has_audio is True and info.audio_codec == "aac"
    assert (info.width, info.height) == (640, 360)
    assert abs(info.fps - 30) < 0.01 and abs(info.duration_seconds - 3) < 0.2
    assert info.audio_sample_rate == 44100


def test_banner_probe_avi(bundled_only, tmp_path):
    info = media_probe.probe_media(make_opencv_video(tmp_path / "b.avi", fourcc="MJPG"))
    assert info.video_codec == "mjpeg" and info.has_audio is False
    assert info.frame_count == 50


def test_opencv_only(monkeypatch, tmp_path):
    monkeypatch.setattr(media_probe, "get_ffmpeg_tools", lambda: FFmpegTools(None, None, None))
    info = media_probe.probe_media(make_opencv_video(tmp_path / "c.mp4"))
    assert info.sources == ["opencv"] and info.width == 320 and info.frame_count == 50
    assert info.has_audio is None  # unknown without FFmpeg — not guessed


def test_corrupt_raises(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"junk" * 1000)
    with pytest.raises(media_probe.ProbeError):
        media_probe.probe_media(p)
