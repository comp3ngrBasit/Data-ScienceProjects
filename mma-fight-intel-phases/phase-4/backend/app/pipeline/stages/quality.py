"""Basic, transparent video quality heuristics computed from sampled frames.

These are simple signal measures (resolution, frame rate, bitrate, exposure,
Laplacian-variance sharpness) — not a learned quality model.
"""
from __future__ import annotations

import statistics

from app.pipeline.base import ProcessingContext

BLUR_THRESHOLD = 60.0  # Laplacian variance at 640px width below which a frame is considered blurry
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 220.0


def resolution_class(w: int | None, h: int | None) -> str:
    if not w or not h:
        return "unknown"
    short = min(w, h)
    if short >= 2160:
        return "4K"
    if short >= 1440:
        return "QHD"
    if short >= 1080:
        return "Full HD"
    if short >= 720:
        return "HD"
    if short >= 480:
        return "SD"
    return "Low"


class QualityStage:
    name = "quality"
    label = "Assessing quality"
    weight = 0.05
    required = False

    def run(self, ctx: ProcessingContext) -> None:
        media = ctx.media
        assert media is not None
        frames = ctx.frames
        b = [f.brightness for f in frames]
        c = [f.contrast for f in frames]
        sh = [f.sharpness for f in frames]
        n = len(frames)

        blurry_pct = round(100 * sum(1 for v in sh if v < BLUR_THRESHOLD) / n, 1) if n else None
        dark_pct = round(100 * sum(1 for v in b if v < DARK_THRESHOLD) / n, 1) if n else None
        bright_pct = round(100 * sum(1 for v in b if v > BRIGHT_THRESHOLD) / n, 1) if n else None

        score = 100.0
        issues: list[str] = []
        res = resolution_class(media.width, media.height)
        if res in ("Low", "SD"):
            score -= 25 if res == "Low" else 12
            issues.append(f"Low resolution ({media.width}x{media.height})")
        if media.fps and media.fps < 24:
            score -= 15
            issues.append(f"Low frame rate ({media.fps:.1f} fps) — fast strikes may be missed")
        if blurry_pct is not None and blurry_pct > 30:
            score -= min(30, blurry_pct / 2)
            issues.append(f"{blurry_pct}% of sampled frames look blurry")
        if dark_pct is not None and dark_pct > 20:
            score -= min(20, dark_pct / 3)
            issues.append(f"{dark_pct}% of sampled frames are very dark")
        if bright_pct is not None and bright_pct > 20:
            score -= min(15, bright_pct / 4)
            issues.append(f"{bright_pct}% of sampled frames are overexposed")
        if media.bitrate_kbps and media.width and media.height:
            bpp = media.bitrate_kbps * 1000 / (media.width * media.height * (media.fps or 30))
            if bpp < 0.03:
                score -= 10
                issues.append("Very low bitrate for the resolution — compression artifacts likely")
        if ctx.frame_read_failures:
            score -= 10
            issues.append(f"{ctx.frame_read_failures} sampled frames failed to decode")

        score = max(0.0, round(score, 1))
        rating = "Good" if score >= 75 else "Fair" if score >= 50 else "Poor"

        ctx.quality = {
            "method": "heuristic-v1",
            "score": score,
            "rating": rating,
            "issues": issues,
            "resolution_class": res,
            "frames_analyzed": n,
            "mean_brightness": round(statistics.fmean(b), 2) if b else None,
            "mean_contrast": round(statistics.fmean(c), 2) if c else None,
            "mean_sharpness": round(statistics.fmean(sh), 2) if sh else None,
            "median_sharpness": round(statistics.median(sh), 2) if sh else None,
            "blurry_frame_pct": blurry_pct,
            "dark_frame_pct": dark_pct,
            "overexposed_frame_pct": bright_pct,
            "decode_failures": ctx.frame_read_failures,
            "thresholds": {"blur": BLUR_THRESHOLD, "dark": DARK_THRESHOLD, "bright": BRIGHT_THRESHOLD},
        }
        ctx.report(self.name, 1.0)
