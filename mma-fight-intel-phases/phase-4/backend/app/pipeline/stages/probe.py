from __future__ import annotations

from app.pipeline.base import ProcessingContext, StageError
from app.services.media_probe import ProbeError, probe_media


class ProbeStage:
    name = "metadata"
    label = "Extracting metadata"
    weight = 0.1
    required = True

    def run(self, ctx: ProcessingContext) -> None:
        ctx.report(self.name, 0.0)
        try:
            ctx.media = probe_media(ctx.source_path)
        except ProbeError as exc:
            raise StageError(str(exc)) from exc
        ctx.warnings.extend(ctx.media.warnings)
        if not ctx.media.width or not ctx.media.height:
            raise StageError("Could not determine video resolution")
        ctx.report(self.name, 1.0)
