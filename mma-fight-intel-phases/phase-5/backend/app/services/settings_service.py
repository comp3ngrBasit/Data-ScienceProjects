"""Persisted runtime settings (stored as one JSON row in ``app_settings``)."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import AppSetting
from app.schemas.settings import ProcessingSettings

PROCESSING_KEY = "processing"


def get_processing_settings(db: Session) -> ProcessingSettings:
    row = db.get(AppSetting, PROCESSING_KEY)
    if row is None:
        return ProcessingSettings()
    # Unknown/old keys are ignored, missing keys take defaults.
    merged = {**ProcessingSettings().model_dump(), **(row.value or {})}
    return ProcessingSettings.model_validate(
        {k: v for k, v in merged.items() if k in ProcessingSettings.model_fields}
    )


def save_processing_settings(db: Session, settings: ProcessingSettings) -> ProcessingSettings:
    row = db.get(AppSetting, PROCESSING_KEY)
    if row is None:
        row = AppSetting(key=PROCESSING_KEY, value=settings.model_dump())
        db.add(row)
    else:
        row.value = settings.model_dump()
    db.commit()
    return settings
