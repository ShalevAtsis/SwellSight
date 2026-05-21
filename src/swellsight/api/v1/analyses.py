from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from swellsight.api.validation import validate_image_upload
from swellsight.api.v1.deps import get_current_user
from swellsight.db.models import Analysis, User
from swellsight.db.session import get_db
from swellsight.platform.dependencies import get_idempotency_store, get_job_queue
from swellsight.platform.settings import get_settings
from swellsight.storage import get_storage

router = APIRouter(prefix="/analyses", tags=["analyses"])

MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class AnalysisResponse(BaseModel):
    id: str
    status: str
    surf_score: Optional[float] = None
    score_breakdown: Optional[dict] = None
    result_json: Optional[dict] = None
    model_version: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


def _to_response(row: Analysis) -> AnalysisResponse:
    return AnalysisResponse(
        id=row.id,
        status=row.status,
        surf_score=row.surf_score,
        score_breakdown=row.score_breakdown,
        result_json=row.result_json,
        model_version=row.model_version,
        error_message=row.error_message,
        created_at=row.created_at.isoformat() if row.created_at else None,
        completed_at=row.completed_at.isoformat() if row.completed_at else None,
    )


def _start_of_utc_day() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@router.post("", response_model=AnalysisResponse, status_code=202)
async def create_analysis(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    settings = get_settings()

    if idempotency_key:
        store = get_idempotency_store()
        existing_id = store.get_analysis_id(user.id, idempotency_key)
        if existing_id:
            row = (
                db.query(Analysis)
                .filter(Analysis.id == existing_id, Analysis.user_id == user.id)
                .first()
            )
            if row:
                return _to_response(row)

    start_of_day = _start_of_utc_day()
    daily_count = (
        db.query(Analysis)
        .filter(Analysis.user_id == user.id, Analysis.created_at >= start_of_day)
        .count()
    )
    if daily_count >= settings.analyses_per_day_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily analysis limit reached ({settings.analyses_per_day_limit}/day)",
        )

    content = await file.read()
    detected_mime, _, _ = validate_image_upload(
        content,
        file.content_type,
        settings.max_upload_bytes,
        settings.max_image_dimension,
    )

    analysis_id = str(uuid.uuid4())
    ext = MIME_EXT.get(detected_mime, ".jpg")
    object_key = f"{user.id}/{analysis_id}{ext}"

    storage = get_storage()
    storage.put(object_key, content, content_type=detected_mime)

    row = Analysis(
        id=analysis_id,
        user_id=user.id,
        status="pending",
        storage_key=object_key,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    if idempotency_key:
        get_idempotency_store().set_analysis_id(user.id, idempotency_key, analysis_id)

    try:
        get_job_queue().enqueue(analysis_id, user.id, object_key)
    except Exception as exc:
        row.status = "failed"
        row.error_message = f"Queue unavailable: {exc}"
        db.commit()
        db.refresh(row)

    return _to_response(row)


@router.get("/{analysis_id}/image")
def get_analysis_image(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return uploaded image bytes for history thumbnails (P4-T09)."""
    row = (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id, Analysis.user_id == user.id)
        .first()
    )
    if not row or not row.storage_key:
        raise HTTPException(status_code=404, detail="Analysis not found")

    storage = get_storage()
    try:
        data = storage.get(row.storage_key)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    ext = row.storage_key.rsplit(".", 1)[-1].lower()
    media = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return Response(content=data, media_type=media)


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id, Analysis.user_id == user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _to_response(row)


@router.get("", response_model=List[AnalysisResponse])
def list_analyses(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
):
    rows = (
        db.query(Analysis)
        .filter(Analysis.user_id == user.id)
        .order_by(Analysis.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    return [_to_response(r) for r in rows]
