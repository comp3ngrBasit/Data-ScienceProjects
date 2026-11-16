from __future__ import annotations

import cv2

from app.pipeline.base import ProcessingContext, StageError
from app.pipeline.stages.imaging import frame_stats, resize_to_width, write_jpeg

CANDIDATE_POSITIONS = (0.1, 0.25, 0.4, 0.55)


class ThumbnailStage:
    """Pick the sharpest, reasonably exposed frame among a few candidates."""

    name = "thumbnail"
    label = "Generating thumbnail"
    weight = 0.1
    required = True

    def run(self, ctx: ProcessingContext) -> None:
        media = ctx.media
        assert media is not None
        cap = cv2.VideoCapture(str(ctx.source_path))
        if not cap.isOpened():
            raise StageError("Could not open video for thumbnail generation")
        best = None
        best_score = -1.0
        try:
            total = media.frame_count or 0
            positions = [int(total * p) for p in CANDIDATE_POSITIONS] if total > 10 else [0]
            for i, pos in enumerate(positions):
                ctx.check_cancelled()
                if pos > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                brightness, _contrast, sharpness = frame_stats(frame)
                exposure_ok = 25 <= brightness <= 230
                score = sharpness * (1.0 if exposure_ok else 0.1)
                if score > best_score:
                    best, best_score = frame, score
                ctx.report(self.name, (i + 1) / len(positions))
            if best is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, best = cap.read()
                if not ok or best is None:
                    raise StageError("Could not decode a frame for the thumbnail")
        finally:
            cap.release()

        out = ctx.storage.thumbnail_path(ctx.video_id)
        write_jpeg(out, resize_to_width(best, ctx.settings.thumbnail_width), 85)
        ctx.thumbnail_path = out
        ctx.report(self.name, 1.0)
