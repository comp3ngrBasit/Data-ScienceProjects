from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _synthetic_frames(n: int, size: tuple[int, int]):
    w, h = size
    for i in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        # textured "octagon" floor + moving blocks so frames have real detail
        cv2.putText(img, f"F{i:04d}", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        x = int((i * 7) % max(1, w - 60))
        cv2.rectangle(img, (x, 20), (x + 50, 90), (0, 0, 220), -1)
        cv2.rectangle(img, (w - x - 60, h - 90), (w - x - 10, h - 20), (220, 220, 0), -1)
        yield img


def make_opencv_video(path: Path, seconds: float = 2.0, fps: int = 25, size=(320, 240), fourcc="mp4v") -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    assert writer.isOpened(), f"VideoWriter could not open {path}"
    for frame in _synthetic_frames(int(seconds * fps), size):
        writer.write(frame)
    writer.release()
    return path


def ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_ffmpeg_video(path: Path, seconds: int = 3, audio: bool = True, vcodec: str = "libx264") -> Path:
    ff = ffmpeg_bin()
    if not ff:
        pytest.skip("ffmpeg not available")
    cmd = [ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", vcodec, "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture()
def make_client(data_dir: Path, monkeypatch) -> Callable[[], TestClient]:
    """Factory so tests can 'restart' the app against the same data directory."""
    monkeypatch.setenv("MMA_DATA_DIR", str(data_dir))
    from app.core import config
    from app.services import ffmpeg_tools

    config.get_settings.cache_clear()
    ffmpeg_tools.get_ffmpeg_tools.cache_clear()

    def factory() -> TestClient:
        config.get_settings.cache_clear()
        from app.main import create_app

        return TestClient(create_app())

    return factory


@pytest.fixture()
def client(make_client) -> Iterator[TestClient]:
    with make_client() as c:
        yield c


@pytest.fixture()
def videos(tmp_path: Path) -> dict[str, Path]:
    src = tmp_path / "src"
    src.mkdir()
    out = {
        "mp4": make_opencv_video(src / "fight_round1.mp4", seconds=3),
        "avi": make_opencv_video(src / "sparring.avi", seconds=2, fourcc="MJPG"),
    }
    corrupt = src / "corrupt.mp4"
    corrupt.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00\x01garbage" * 2000)
    out["corrupt"] = corrupt
    txt = src / "notes.txt"
    txt.write_text("not a video")
    out["txt"] = txt
    fake_avi = src / "fake.avi"
    fake_avi.write_bytes(b"this is not an avi file at all" * 10)
    out["fake_avi"] = fake_avi
    return out


def upload(client: TestClient, *paths: Path, mime: str = "video/mp4"):
    files = [("files", (p.name, p.read_bytes(), mime)) for p in paths]
    return client.post("/api/videos/upload", files=files)


def wait_for(client: TestClient, video_id: str, statuses=("completed", "failed"), timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/videos/{video_id}/status")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in statuses:
            return body
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {video_id} -> {statuses}; last={body}")
