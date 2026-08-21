from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from datetime import date
from app.schemas.assessment import FirstAssessment
from app.db.repository import save_assessment, get_assessment_by_id, list_assessments
from app.api.deps import get_db

router = APIRouter()

@router.post("/assessments", status_code=201)
async def create_assessment(
    assessment: FirstAssessment,
    transcript: Optional[str] = None,
    db=Depends(get_db)
):
    """Save a parsed assessment to MongoDB."""
    id = await save_assessment(assessment, transcript)
    return {"id": id, "assessment": assessment.model_dump()}

@router.get("/assessments/{id}")
async def retrieve_assessment(id: str, db=Depends(get_db)):
    """Retrieve a saved assessment by ID."""
    doc = await get_assessment_by_id(id)
    if not doc:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return {
        "id": doc.get("_id"),
        "assessment": doc.get("assessment"),
        "transcript": doc.get("transcript", ""),
        "created_at": doc.get("created_at")
    }

@router.get("/assessments")
async def list_all_assessments(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db=Depends(get_db)
):
    """List all assessments, optionally filtered by date range."""
    docs = await list_assessments(start_date, end_date)
    return [
        {
            "id": d.get("_id"),
            "assessment": d.get("assessment"),
            "created_at": d.get("created_at")
        }
        for d in docs
    ]