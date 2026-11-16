from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db
from app.db.models import Fighter, Video
from app.schemas.fighter import FighterCreate, FighterDetail, FighterOut, FighterUpdate, FighterVideoRef

router = APIRouter(prefix="/api/fighters", tags=["fighters"])


def _counts(db: Session, fighter: Fighter) -> tuple[int, int]:
    assigned = {v.id for v in fighter.videos}
    target_ids = set(db.scalars(select(Video.id).where(Video.target_fighter_id == fighter.id)).all())
    return len(assigned | target_ids), len(target_ids)


def _to_out(db: Session, fighter: Fighter) -> FighterOut:
    out = FighterOut.model_validate(fighter, from_attributes=True)
    out.video_count, out.target_video_count = _counts(db, fighter)
    return out


def _to_detail(db: Session, fighter: Fighter) -> FighterDetail:
    base = _to_out(db, fighter)
    videos = {v.id: v for v in fighter.videos}
    for v in db.scalars(select(Video).where(Video.target_fighter_id == fighter.id)).all():
        videos.setdefault(v.id, v)
    refs = [
        FighterVideoRef(
            id=v.id,
            title=v.title,
            status=v.status.value,
            duration_seconds=v.duration_seconds,
            uploaded_at=v.uploaded_at,
            has_thumbnail=bool(v.thumbnail_path),
            is_target=v.target_fighter_id == fighter.id,
        )
        for v in sorted(videos.values(), key=lambda v: v.uploaded_at, reverse=True)
    ]
    return FighterDetail(**base.model_dump(), videos=refs)


def _get(db: Session, fighter_id: str) -> Fighter:
    f = db.scalar(select(Fighter).where(Fighter.id == fighter_id).options(selectinload(Fighter.videos)))
    if f is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fighter not found")
    return f


def _check_unique_name(db: Session, name: str, exclude_id: str | None = None) -> None:
    q = select(Fighter.id).where(func.lower(Fighter.name) == name.lower())
    if exclude_id:
        q = q.where(Fighter.id != exclude_id)
    if db.scalar(q):
        raise HTTPException(status.HTTP_409_CONFLICT, f"A fighter named '{name}' already exists")


@router.get("", response_model=list[FighterOut], summary="List fighters")
def list_fighters(search: str | None = Query(None, max_length=120), db: Session = Depends(get_db)):
    q = select(Fighter).options(selectinload(Fighter.videos)).order_by(func.lower(Fighter.name))
    if search:
        like = f"%{search.strip()}%"
        q = q.where(Fighter.name.ilike(like) | Fighter.nickname.ilike(like))
    return [_to_out(db, f) for f in db.scalars(q).all()]


@router.post("", response_model=FighterDetail, status_code=status.HTTP_201_CREATED, summary="Create fighter profile")
def create_fighter(payload: FighterCreate, db: Session = Depends(get_db)):
    _check_unique_name(db, payload.name)
    f = Fighter(**payload.model_dump())
    db.add(f)
    db.commit()
    return _to_detail(db, _get(db, f.id))


@router.get("/{fighter_id}", response_model=FighterDetail, summary="Fighter profile with associated videos")
def get_fighter(fighter_id: str, db: Session = Depends(get_db)):
    return _to_detail(db, _get(db, fighter_id))


@router.put("/{fighter_id}", response_model=FighterDetail, summary="Update fighter profile")
def update_fighter(fighter_id: str, payload: FighterUpdate, db: Session = Depends(get_db)):
    f = _get(db, fighter_id)
    _check_unique_name(db, payload.name, exclude_id=fighter_id)
    for k, v in payload.model_dump().items():
        setattr(f, k, v)
    db.commit()
    return _to_detail(db, _get(db, fighter_id))


@router.delete("/{fighter_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete fighter (videos are kept, just unassigned)")
def delete_fighter(fighter_id: str, db: Session = Depends(get_db)):
    f = _get(db, fighter_id)
    for v in db.scalars(select(Video).where(Video.target_fighter_id == fighter_id)).all():
        v.target_fighter_id = None
    f.videos = []
    db.delete(f)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
