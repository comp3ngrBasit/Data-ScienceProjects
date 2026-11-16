"""Runs an ordered list of stages and records a per-stage log."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from app.pipeline.base import JobCancelled, ProcessingContext, Stage, StageError
from app.pipeline.stages.frame_sampling import FrameSamplingStage
from app.pipeline.stages.preview import PreviewProxyStage
from app.pipeline.stages.probe import ProbeStage
from app.pipeline.stages.quality import QualityStage
from app.pipeline.stages.thumbnail import ThumbnailStage

log = logging.getLogger(__name__)

# Registry of job types -> stage factories. Phase 2+ can register new job
# types here (e.g. "phase2_detection") without touching the job system.
JOB_TYPES: dict[str, Callable[[], list[Stage]]] = {
    "phase1_ingest": lambda: [
        ProbeStage(),
        ThumbnailStage(),
        FrameSamplingStage(),
        QualityStage(),
        PreviewProxyStage(),
    ],
}


class PipelineRunner:
    def __init__(self, stages: Sequence[Stage]):
        self.stages = list(stages)
        self.total_weight = sum(s.weight for s in self.stages) or 1.0
        self.stage_log: list[dict[str, Any]] = []

    def run(self, ctx: ProcessingContext, on_progress: Callable[[str, float, list[dict]], None]) -> None:
        done_weight = 0.0
        for stage in self.stages:
            entry: dict[str, Any] = {"stage": stage.name, "label": stage.label, "status": "running"}
            self.stage_log.append(entry)
            start = time.monotonic()

            def cb(_name: str, frac: float, _w=done_weight, _s=stage) -> None:
                on_progress(_s.name, (_w + _s.weight * frac) / self.total_weight, self.stage_log)

            ctx.progress_cb = cb
            cb(stage.name, 0.0)
            try:
                ctx.check_cancelled()
                stage.run(ctx)
                entry["status"] = "completed"
            except JobCancelled:
                entry["status"] = "cancelled"
                raise
            except StageError as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                if stage.required:
                    raise
                ctx.warnings.append(f"{stage.label}: {exc}")
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                log.exception("Stage %s failed for video %s", stage.name, ctx.video_id)
                if stage.required:
                    raise StageError(f"{stage.label} failed: {exc}") from exc
                ctx.warnings.append(f"{stage.label} failed: {exc}")
            finally:
                entry["seconds"] = round(time.monotonic() - start, 2)
            done_weight += stage.weight
            cb(stage.name, 1.0)
