"""Save and retrieve endpoints for stored assessments."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas import FirstAssessment
from app.services.storage import (
    get_assessment_by_id,
    list_assessments,
    save_assessment_to_db,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["assessments"])


@router.post("/assessments", status_code=status.HTTP_201_CREATED)
async def save_assessment(assessment: FirstAssessment):
    """Persist a parsed FirstAssessment payload to MongoDB."""
    try:
        saved_record = await save_assessment_to_db(assessment.model_dump())
        return {
            "message": "Assessment successfully saved to database",
            "data": saved_record
        }
    except Exception as e:
        logger.error(f"Error saving assessment to database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database save error: {str(e)}"
        )


@router.get("/assessments/{id}")
async def get_assessment(id: str):
    """Retrieve a saved assessment report by its unique ID."""
    record = await get_assessment_by_id(id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment with ID '{id}' not found."
        )
    return record


@router.get("/assessments")
async def get_all_assessments(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)")
):
    """List saved assessment records, optionally filtered by date."""
    records = await list_assessments(filter_date=date)
    return {
        "count": len(records),
        "filter_date": date,
        "assessments": records
    }
