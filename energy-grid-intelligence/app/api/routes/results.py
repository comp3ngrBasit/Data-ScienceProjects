"""Read-only routes over the analytical result store."""
from __future__ import annotations

import json
import math
import re

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.api.schemas.models import ArtifactList, ForecastResponse, Health, KPIResponse, TableResponse
from app.config import REPORTS_DIR, RESULTS_DIR
from app.utils.helpers import load_json, load_table

router = APIRouter()
SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _check(*parts: str) -> None:
    for p in parts:
        if not SAFE.match(p):
            raise HTTPException(400, f"invalid name: {p}")


def _records(df: pd.DataFrame) -> list[dict]:
    d = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    out = json.loads(d.to_json(orient="records", date_format="iso"))
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in out]


@router.get("/health", response_model=Health)
def health() -> Health:
    man = RESULTS_DIR / "_manifest.json"
    stages = {k: v.get("status", "?") for k, v in json.loads(man.read_text()).items()} if man.exists() else {}
    return Health(status="ok", results_available=bool(stages), stages=stages)


@router.get("/kpis", response_model=KPIResponse)
def kpis() -> KPIResponse:
    k = load_json("overview", "kpis")
    if k is None:
        raise HTTPException(404, "KPIs not computed yet - run `python -m app.pipeline run`")
    return KPIResponse(kpis=k)


@router.get("/modules", response_model=list[ArtifactList])
def modules() -> list[ArtifactList]:
    out = []
    for d in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        out.append(ArtifactList(module=d.name, tables=sorted(p.stem for p in d.glob("*.parquet")),
                                documents=sorted(p.stem for p in d.glob("*.json"))))
    return out


@router.get("/results/{module}/{name}/json")
def document(module: str, name: str) -> dict:
    _check(module, name)
    doc = load_json(module, name)
    if doc is None:
        raise HTTPException(404, f"{module}/{name} not found")
    return doc


@router.get("/results/{module}/{name}", response_model=TableResponse)
def table(module: str, name: str, offset: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=100_000),
          columns: str | None = Query(None, description="comma-separated column subset")) -> TableResponse:
    _check(module, name)
    df = load_table(module, name)
    if df is None:
        raise HTTPException(404, f"{module}/{name} not found")
    if columns:
        cols = [c for c in columns.split(",") if c in df.columns]
        df = df[cols]
    page = df.iloc[offset: offset + limit]
    return TableResponse(module=module, name=name, total_rows=len(df), offset=offset, limit=limit,
                         columns=[str(c) for c in page.columns], rows=_records(page))


@router.get("/forecasts/{key}", response_model=ForecastResponse)
def forecast(key: str) -> ForecastResponse:
    _check(key)
    s, m = load_json("forecasting", f"{key}_summary"), load_table("forecasting", f"{key}_metrics")
    if s is None or m is None:
        raise HTTPException(404, f"forecast {key} not found")
    return ForecastResponse(key=key, summary=s, metrics=_records(m))


@router.get("/anomalies")
def anomalies(severity: str | None = None, limit: int = Query(500, le=10_000)) -> list[dict]:
    df = load_table("anomaly", "consensus")
    if df is None:
        raise HTTPException(404, "anomalies not computed")
    if severity:
        df = df[df["severity"] == severity]
    return _records(df.sort_values("severity_score", ascending=False).head(limit))


@router.get("/reports/pdf")
def report_pdf() -> FileResponse:
    p = REPORTS_DIR / "energy_grid_intelligence_report.pdf"
    if not p.exists():
        raise HTTPException(404, "report not generated - run the 'report' stage")
    return FileResponse(p, media_type="application/pdf", filename=p.name)


@router.get("/reports/csv")
def report_csv() -> FileResponse:
    p = REPORTS_DIR / "energy_grid_intelligence_tables.zip"
    if not p.exists():
        raise HTTPException(404, "CSV export not generated - run the 'report' stage")
    return FileResponse(p, media_type="application/zip", filename=p.name)
