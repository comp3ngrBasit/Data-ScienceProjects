"""Configurable frame sampling.

Only a bounded subset of frames is decoded and (optionally) stored, controlled
by ``ProcessingSettings``: sampling mode, interval / count, hard cap, output
width and JPEG quality.
"""
from __future__ import annotations

import logging

import cv2

from app.pipeline.base import FrameRecord, ProcessingContext, StageError
from app.pipeline.stages.imaging import frame_stats, resize_to_width, write_jpeg
from app.schemas.settings import ProcessingSettings

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 8


def compute_sample_indices(frame_count: int, fps: float, settings: ProcessingSettings) -> list[int]:
    """Return sorted, unique frame indices to sample."""
    if frame_count <= 0:
        return []
    if settings.sampling_mode == "uniform":
        n = min(settings.uniform_frame_count, frame_count)
        indices = [int((i + 0.5) * frame_count / n) for i in range(n)]
    else:
        step = max(1, int(round(settings.sample_interval_seconds * fps))) if fps else 1
        indices = list(range(0, frame_count, step))

    cap = settings.max_sampled_frames
    if len(indices) > cap:
        # Spread the capped budget evenly over the selected indices.
        indices = [indices[int(i * len(indices) / cap)] for i in range(cap)]
    return sorted(set(min(i, frame_count - 1) for i in indices))


class FrameSamplingStage:
    name = "frame_sampling"
    label = "Sampling frames"
    weight = 0.45
    required = True

    def run(self, ctx: ProcessingContext) -> None:
        media = ctx.media
        assert media is not None
        s = ctx.settings
        fps = media.fps or 30.0
        frame_count = media.frame_count or 0
        indices = compute_sample_indices(frame_count, fps, s)
        if not indices:
            raise StageError("Video has no frames to sample")

        out_dir = ctx.storage.frames_dir(ctx.video_id)
        ctx.storage.remove_tree(out_dir)  # reprocessing replaces previous samples
        if s.save_frame_images:
            out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(ctx.source_path))
        if not cap.isOpened():
            raise StageError("Could not open video for frame sampling")

        seek_threshold = max(int(fps * 2), 48)  # beyond this gap, seeking beats grabbing
        pos = 0  # index of the next frame cap.read() will return
        consecutive_failures = 0
        try:
            for n, target in enumerate(indices):
                ctx.check_cancelled()
                gap = target - pos
                if gap < 0 or gap > seek_threshold:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                else:
                    for _ in range(gap):
                        if not cap.grab():
                            break
                ok, frame = cap.read()
                pos = target + 1
                if not ok or frame is None:
                    ctx.frame_read_failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        log.info("Stopping sampling for %s after %d consecutive read failures (likely end of stream)",
                                 ctx.video_id, consecutive_failures)
                        break
                    continue
                consecutive_failures = 0

                brightness, contrast, sharpness = frame_stats(frame)
                img_path = None
                out = resize_to_width(frame, s.frame_max_width)
                if s.save_frame_images:
                    img_path = out_dir / f"frame_{target:07d}.jpg"
                    write_jpeg(img_path, out, s.frame_jpeg_quality)
                ctx.frames.append(
                    FrameRecord(
                        frame_index=target,
                        timestamp_seconds=round(target / fps, 4),
                        image_path=img_path,
                        width=out.shape[1],
                        height=out.shape[0],
                        brightness=brightness,
                        contrast=contrast,
                        sharpness=sharpness,
                    )
                )
                if n % 5 == 0:
                    ctx.report(self.name, (n + 1) / len(indices))
        finally:
            cap.release()

        if not ctx.frames:
            raise StageError("No frames could be decoded — the file may be corrupted")
        if ctx.frame_read_failures:
            ctx.warnings.append(
                f"{ctx.frame_read_failures} of {len(indices)} sampled frames could not be decoded"
            )
        ctx.report(self.name, 1.0)
