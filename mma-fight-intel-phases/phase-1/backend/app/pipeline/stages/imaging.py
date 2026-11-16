"""Small OpenCV helpers shared by stages."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ANALYSIS_WIDTH = 640  # frames are normalised to this width before computing stats


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    new_h = max(1, int(round(h * width / w)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def frame_stats(img: np.ndarray) -> tuple[float, float, float]:
    """Return (brightness 0-255, contrast = std of luma, sharpness = variance of Laplacian)."""
    small = resize_to_width(img, ANALYSIS_WIDTH)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(brightness, 2), round(contrast, 2), round(sharpness, 2)


def write_jpeg(path: Path, img: np.ndarray, quality: int) -> None:
    """Encode in memory and write with Python I/O (cv2.imwrite fails on non-ASCII Windows paths)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
