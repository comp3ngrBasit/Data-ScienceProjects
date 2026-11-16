from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scip.api.deps import records
from scip.copilot import rag
from scip.core.db import read_sql

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=15)


@router.post("/ask")
def ask(q: Question):
    try:
        return rag.ask(q.question, k=q.top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.post("/reindex")
def reindex():
    return {"documents": rag.build_index()}


@router.get("/history")
def history(limit: int = 50):
    return records(read_sql("SELECT * FROM copilot_messages ORDER BY id DESC LIMIT :l", {"l": limit}))
