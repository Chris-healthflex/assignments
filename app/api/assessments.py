from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.db.mongodb import (
    get_assessment,
    list_assessments,
    save_assessment,
)
from app.graph.clinical_graph import clinical_graph
from app.models.assessment import FirstAssessment
from app.services.transcription import transcribe_audio


router = APIRouter(
    prefix="/assessments",
    tags=["assessments"],
)


@router.post("/parse", response_model=FirstAssessment)
async def parse_assessment(file: UploadFile = File(...)):
    """
    Upload a WAV file, transcribe it with Whisper, extract a structured
    FirstAssessment using LangGraph, and return the assessment.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is required.",
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only WAV audio files are supported.",
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    import tempfile
    from pathlib import Path

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        transcript = transcribe_audio(temp_path)

        result = clinical_graph.invoke(
            {
                "transcript": transcript,
            }
        )

        if result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result["error"],
            )

        assessment = result.get("assessment")

        if assessment is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Clinical assessment could not be extracted."
                },
            )

        return assessment

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to process the audio file.",
                "error": str(exc),
            },
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@router.post("", response_model=dict[str, str])
async def create_assessment(assessment: FirstAssessment):
    """
    Save a structured assessment to MongoDB.
    """

    try:
        assessment_id = save_assessment(
            assessment.model_dump()
        )

        return {
            "id": assessment_id,
            "message": "Assessment saved successfully.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Unable to save assessment to MongoDB.",
                "error": str(exc),
            },
        )


@router.get("/{assessment_id}")
async def retrieve_assessment(assessment_id: str):
    """
    Retrieve a saved assessment by MongoDB ID.
    """

    try:
        assessment = get_assessment(assessment_id)

        if assessment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found.",
            )

        return assessment

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Unable to retrieve assessment.",
                "error": str(exc),
            },
        )


@router.get("")
async def retrieve_assessments(
    start_date: datetime | None = Query(
        default=None,
        description="Return assessments created on or after this UTC datetime.",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Return assessments created before this UTC datetime.",
    ),
) -> list[dict[str, Any]]:
    """
    List saved assessments, optionally filtered by creation date.
    """

    if start_date and end_date and start_date >= end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be earlier than end_date.",
        )

    try:
        return list_assessments(
            start_date=start_date,
            end_date=end_date,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Unable to list assessments.",
                "error": str(exc),
            },
        )