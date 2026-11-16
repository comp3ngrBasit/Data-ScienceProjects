"""Pipeline primitives.

A processing job runs an ordered list of *stages* against a video. Each stage
reads/writes a shared ``ProcessingContext``. Future phases (detection, pose,
strike classification...) plug in by adding stages or by registering a new
job type with its own stage list — the upload/library/job system does not
need to change.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.schemas.settings import ProcessingSettings
from app.services.media_probe import MediaInfo
from app.services.storage import Storage

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised inside a stage when the job was cancelled (e.g. video deleted)."""


class StageError(RuntimeError):
    """A stage failed in a way that should fail the job, with a user-facing message."""


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_seconds: float
    image_path: Path | None
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float


@dataclass
class ProcessingContext:
    video_id: str
    source_path: Path
    file_ext: str
    settings: ProcessingSettings
    storage: Storage
    progress_cb: Callable[[str, float], None] = lambda stage, frac: None
    is_cancelled: Callable[[], bool] = lambda: False

    # Outputs
    media: MediaInfo | None = None
    thumbnail_path: Path | None = None
    frames: list[FrameRecord] = field(default_factory=list)
    frame_read_failures: int = 0
    quality: dict[str, Any] | None = None
    preview_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise JobCancelled()

    def report(self, stage: str, fraction: float) -> None:
        self.progress_cb(stage, max(0.0, min(1.0, fraction)))


class Stage(Protocol):
    name: str
    label: str
    weight: float  # relative share of the progress bar
    required: bool  # if False, failures become warnings instead of failing the job

    def run(self, ctx: ProcessingContext) -> None: ...
