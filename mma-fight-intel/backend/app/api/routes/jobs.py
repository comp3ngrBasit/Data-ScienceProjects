from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db, get_job_manager
from app.db.models import JobStatus, ProcessingJob, Video
from app.schemas.video import JobOut
from app.services import video_service as vs
from app.services.job_manager import JobConflictError, JobManager

router = APIRouter(prefix="/api/jobs", tags=["processing jobs"])


@router.get("", response_model=dict, summary="List processing jobs")
def list_jobs(
    status_filter: JobStatus | None = Query(None, alias="status"),
    video_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = select(ProcessingJob)
    if status_filter:
        q = q.where(ProcessingJob.status == status_filter)
    if video_id:
        q = q.where(ProcessingJob.video_id == video_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    counts = dict(db.execute(select(ProcessingJob.status, func.count()).group_by(ProcessingJob.status)).all())
    rows = db.scalars(
        q.options(selectinload(ProcessingJob.video)).order_by(ProcessingJob.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return {
        "items": [vs.job_to_out(j).model_dump(mode="json") for j in rows],
        "total": total,
        "counts": {s.value: counts.get(s, 0) for s in JobStatus},
        "limit": limit,
        "offset": offset,
    }


@router.get("/{job_id}", response_model=JobOut, summary="Get a processing job")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return vs.job_to_out(job)


@router.post("/{job_id}/retry", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Retry a failed or cancelled job (creates a new attempt)")
def retry_job(job_id: str, db: Session = Depends(get_db), jobs: JobManager = Depends(get_job_manager)):
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status not in (JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Only failed or cancelled jobs can be retried (status: {job.status.value})")
    video = db.scalar(select(Video).where(Video.id == job.video_id).options(selectinload(Video.jobs)))
    try:
        new_job = jobs.create_job(db, video, job.job_type)
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return vs.job_to_out(new_job)
