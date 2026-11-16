from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProcessingSettings(BaseModel):
    """Runtime-adjustable settings, persisted in the database."""

    sampling_mode: Literal["interval", "uniform"] = Field(
        "interval",
        description="interval: one frame every N seconds; uniform: N evenly spaced frames across the video",
    )
    sample_interval_seconds: float = Field(1.0, ge=0.04, le=600)
    uniform_frame_count: int = Field(60, ge=1, le=5000)
    max_sampled_frames: int = Field(300, ge=1, le=5000, description="Hard cap on stored frames per video")
    save_frame_images: bool = Field(True, description="Store sampled frames as JPEGs (otherwise only stats)")
    frame_max_width: int = Field(640, ge=64, le=3840)
    frame_jpeg_quality: int = Field(85, ge=30, le=100)
    thumbnail_width: int = Field(480, ge=64, le=1920)
    generate_preview_proxy: bool = Field(
        True, description="Transcode non-MP4/non-H.264 uploads into a browser-playable MP4 preview"
    )
    auto_process_on_upload: bool = True
    max_upload_size_mb: int = Field(4096, ge=1, le=20480)


class SystemInfo(BaseModel):
    app_name: str
    version: str
    ffmpeg_path: str | None
    ffprobe_path: str | None
    ffmpeg_source: str | None
    opencv_version: str
    data_dir: str
    database_url: str
    allowed_extensions: list[str]
    worker_threads: int
    storage_bytes: dict[str, int]
