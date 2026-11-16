from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.models import Stance


class FighterBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    nickname: str | None = Field(None, max_length=120)
    stance: Stance = Stance.unknown
    height_cm: float | None = Field(None, gt=100, lt=250, description="Height in centimetres")
    reach_cm: float | None = Field(None, gt=100, lt=260, description="Reach in centimetres")
    weight_class: str | None = Field(None, max_length=60)
    notes: str | None = Field(None, max_length=5000)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be blank")
        return v

    @field_validator("nickname", "weight_class", "notes")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class FighterCreate(FighterBase):
    pass


class FighterUpdate(FighterBase):
    """PUT payload — full replacement of editable fields."""


class FighterVideoRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    status: str
    duration_seconds: float | None
    uploaded_at: datetime
    has_thumbnail: bool = False
    is_target: bool = False


class FighterOut(FighterBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime
    video_count: int = 0
    target_video_count: int = 0


class FighterDetail(FighterOut):
    videos: list[FighterVideoRef] = []
