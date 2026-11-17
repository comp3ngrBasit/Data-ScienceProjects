"""Runtime settings for Phase 3 (persisted in app_settings under key 'actions')."""
from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.actions.models import ALGORITHMS
from app.db.models import AppSetting

KEY = "actions"


class ActionSettings(BaseModel):
    # inference
    confidence_threshold: float = Field(0.6, ge=0.05, le=0.99, description="Minimum model probability to report a labelled action")
    motion_gate: float = Field(1.5, ge=0, le=50, description="Skip windows whose fastest limb moves slower than this (torso lengths/s)")
    unknown_motion: float = Field(6.0, ge=0, le=100, description="Report 'Unknown' for fast movement the model can't classify confidently")
    hop_fraction: float = Field(0.25, ge=0.05, le=1.0, description="Sliding-window step as a fraction of the window length")
    window_seconds: float | None = Field(None, ge=0.2, le=5.0, description="Override the model's window length")
    nms_iou: float = Field(0.3, ge=0.0, le=1.0, description="Overlapping detections above this IoU are de-duplicated")
    min_visible_fraction: float = Field(0.5, ge=0.0, le=1.0, description="Skip windows where fewer core landmarks are visible")
    sequence_gap: float = Field(2.0, ge=0.2, le=30, description="A new sequence starts after this many seconds without an action")
    fighters_only: bool = Field(True, description="Only analyse tracks marked as fighters (or suggested fighters if none are marked)")
    # training
    algorithms: list[str] = Field(default_factory=lambda: ["random_forest", "xgboost", "svm", "lstm", "tcn"])
    min_examples: int = Field(4, ge=1, le=1000, description="A class needs at least this many annotated examples")
    min_groups: int = Field(2, ge=1, le=100, description="... from at least this many different videos/fights")
    augment: int = Field(4, ge=0, le=20, description="Temporal-jitter copies per training example")
    negatives_per_positive: float = Field(1.5, ge=0, le=10)
    seq_len: int = Field(16, ge=6, le=64)
    context: float = Field(0.15, ge=0, le=1.0, description="Seconds of context around annotated actions")
    include_draft_annotations: bool = False
    epochs: int = Field(60, ge=5, le=500)
    device: str = Field("auto", description="auto | cpu | cuda | mps (temporal models)")

    def model_post_init(self, __context) -> None:
        self.algorithms = [a for a in self.algorithms if a in ALGORITHMS] or ["random_forest"]


def get_action_settings(db: Session) -> ActionSettings:
    row = db.get(AppSetting, KEY)
    data = ActionSettings().model_dump()
    if row and row.value:
        data.update({k: v for k, v in row.value.items() if k in ActionSettings.model_fields})
    try:
        return ActionSettings.model_validate(data)
    except Exception:
        return ActionSettings()


def save_action_settings(db: Session, s: ActionSettings) -> ActionSettings:
    row = db.get(AppSetting, KEY)
    if row is None:
        db.add(AppSetting(key=KEY, value=s.model_dump()))
    else:
        row.value = s.model_dump()
    db.commit()
    return s
