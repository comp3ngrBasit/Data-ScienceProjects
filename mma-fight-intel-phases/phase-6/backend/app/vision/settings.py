"""Runtime-adjustable computer-vision settings (persisted in ``app_settings`` under key ``vision``)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.db.models import AppSetting
from app.vision.constants import AVAILABLE_MODELS

VISION_KEY = "vision"


class VisionSettings(BaseModel):
    model: str = Field("yolo26n-pose", description="Pretrained Ultralytics pose model (n=fastest ... x=most accurate)")
    device: str = Field("auto", description="auto | cpu | cuda | cuda:N | mps")
    half_precision: bool = Field(True, description="FP16 inference on CUDA (ignored on CPU)")
    image_size: int = Field(640, ge=320, le=1920, description="Inference resolution (multiple of 32)")
    analysis_fps: float = Field(10.0, ge=0, le=240, description="Frames analysed per second of video; 0 = every frame")
    detection_confidence: float = Field(0.35, ge=0.05, le=0.95)
    iou_threshold: float = Field(0.6, ge=0.1, le=0.95)
    keypoint_confidence: float = Field(0.35, ge=0.05, le=0.95, description="Landmarks below this are treated as missing")
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"
    track_buffer_seconds: float = Field(2.0, ge=0.1, le=30, description="How long a lost track is kept alive (occlusion)")
    min_box_height: float = Field(0.08, ge=0.0, le=0.9, description="Ignore people shorter than this fraction of frame height (crowd)")
    max_people_per_frame: int = Field(6, ge=1, le=50, description="Keep only the N largest people per frame")
    generate_annotated_video: bool = True
    annotated_max_width: int = Field(1280, ge=320, le=3840)
    smoothing_window: int = Field(5, ge=1, le=31, description="Centered rolling window (analysed frames)")
    max_gap_interpolation: int = Field(3, ge=0, le=30, description="Fill landmark gaps up to N analysed frames")
    stationary_threshold: float = Field(0.15, ge=0.0, le=5.0, description="Torso-lengths/s below which a body counts as stationary")

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        v = v.removesuffix(".pt")
        if v not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{v}'. Choose one of: {', '.join(AVAILABLE_MODELS)}")
        return v

    @field_validator("device")
    @classmethod
    def _device(cls, v: str) -> str:
        v = v.strip().lower()
        if v in ("auto", "cpu", "cuda", "mps") or (v.startswith("cuda:") and v[5:].isdigit()):
            return v
        raise ValueError("device must be auto, cpu, cuda, cuda:N or mps")

    @field_validator("image_size")
    @classmethod
    def _imgsz(cls, v: int) -> int:
        return int(round(v / 32) * 32)


def get_vision_settings(db: Session) -> VisionSettings:
    row = db.get(AppSetting, VISION_KEY)
    base = VisionSettings().model_dump()
    if row and row.value:
        base.update({k: v for k, v in row.value.items() if k in VisionSettings.model_fields})
    try:
        return VisionSettings.model_validate(base)
    except Exception:
        return VisionSettings()


def save_vision_settings(db: Session, s: VisionSettings) -> VisionSettings:
    row = db.get(AppSetting, VISION_KEY)
    if row is None:
        db.add(AppSetting(key=VISION_KEY, value=s.model_dump()))
    else:
        row.value = s.model_dump()
    db.commit()
    return s
