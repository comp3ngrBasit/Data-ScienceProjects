"""Pydantic response schemas for the REST API (stable contract for a future React frontend)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    results_available: bool
    stages: dict[str, str] = Field(default_factory=dict)


class TableResponse(BaseModel):
    module: str
    name: str
    total_rows: int
    offset: int
    limit: int
    columns: list[str]
    rows: list[dict[str, Any]]


class ArtifactList(BaseModel):
    module: str
    tables: list[str]
    documents: list[str]


class KPIResponse(BaseModel):
    kpis: dict[str, Any]


class ForecastResponse(BaseModel):
    key: str
    summary: dict[str, Any]
    metrics: list[dict[str, Any]]
