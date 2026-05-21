from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from swellsight.api.v1.deps import get_current_user
from swellsight.db.models import Analysis, User
from swellsight.db.session import get_db
from swellsight.jobs.queue import JobQueue

router = APIRouter(prefix="/analyses", tags=["analyses"])
UPLOAD_ROOT = Path(__file__).resolve().parents[5] / "data" / "uploads"


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


@router.post("", response_model=AnalysisResponse, status_code=202)
async def create_analysis(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    analysis_id = str(uuid.uuid4())
    user_dir = UPLOAD_ROOT / user.id
    user_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    storage_key = str(user_dir / f"{analysis_id}{ext}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    Path(storage_key).write_bytes(content)

    row = Analysis(
        id=analysis_id,
        user_id=user.id,
        status="pending",
        storage_key=storage_key,
    )
    db.add(row)
    db.commit()

    try:
        JobQueue().enqueue(analysis_id, user.id, storage_key)
    except Exception as exc:
        row.status = "failed"
        row.error_message = f"Queue unavailable: {exc}"
        db.commit()

    return _to_response(row)


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
