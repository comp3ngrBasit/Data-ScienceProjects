"""Runtime settings for Phase 4 (persisted in app_settings under key 'patterns')."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import AppSetting
from app.patterns.engine import PatternConfig

KEY = "patterns"


class PatternSettings(BaseModel):
    source: Literal["best", "detections", "annotations"] = Field(
        "best", description="best = verified annotations for fully-annotated videos, model detections elsewhere")
    min_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Model detections below this probability are excluded (reviewed ones are kept)")
    include_low_confidence: bool = Field(False, description="Include unreviewed detections Phase 3 flagged as low-confidence")
    sequence_gap: float = Field(2.0, ge=0.2, le=30, description="A new sequence starts after this many seconds without an action")
    window: float = Field(1.5, ge=0.1, le=10, description="Observation window for 'next action within N seconds'")
    overlap_iou: float = Field(0.5, ge=0.05, le=1.0, description="Same-fighter detections overlapping more than this are duplicates")
    min_occurrences: int = Field(2, ge=1, le=1000, description="Minimum occurrences for a combination / pattern to be listed")
    min_fights: int = Field(1, ge=1, le=100, description="Minimum distinct fights for a combination / pattern to be listed")
    min_observations: int = Field(5, ge=1, le=10000, description="Probabilities based on fewer observations are labelled insufficient")
    max_n: int = Field(5, ge=2, le=8, description="Longest combination length")
    smoothing_alpha: float = Field(0.5, ge=0.0, le=5.0, description="Additive smoothing for Markov matrices (0 = none)")
    max_order: int = Field(3, ge=1, le=3)
    min_context_count: int = Field(5, ge=1, le=10000, description="Higher-order Markov contexts need this many observations")
    significance: float = Field(0.05, gt=0.0, lt=0.5)
    opponent_reaction_window: float = Field(1.0, ge=0.1, le=5.0, description="A response must start within this many seconds after an opponent attack ends")
    min_actions_per_opponent: int = Field(10, ge=1, le=10000, description="Opponent-specific findings need this many of the fighter's actions against that opponent")

    def engine_config(self) -> PatternConfig:
        return PatternConfig(
            sequence_gap=self.sequence_gap, window=self.window, overlap_iou=self.overlap_iou,
            min_occurrences=self.min_occurrences, min_fights=self.min_fights, min_observations=self.min_observations,
            max_n=self.max_n, smoothing_alpha=self.smoothing_alpha, max_order=self.max_order,
            min_context_count=self.min_context_count, significance=self.significance,
            opponent_reaction_window=self.opponent_reaction_window)


def get_pattern_settings(db: Session) -> PatternSettings:
    row = db.get(AppSetting, KEY)
    data = PatternSettings().model_dump()
    if row and row.value:
        data.update({k: v for k, v in row.value.items() if k in PatternSettings.model_fields})
    try:
        return PatternSettings.model_validate(data)
    except Exception:
        return PatternSettings()


def save_pattern_settings(db: Session, s: PatternSettings) -> PatternSettings:
    row = db.get(AppSetting, KEY)
    if row is None:
        db.add(AppSetting(key=KEY, value=s.model_dump()))
    else:
        row.value = s.model_dump()
    db.commit()
    return s
