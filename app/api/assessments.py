from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.transcription import transcribe_audio
from app.services.extraction import (
    AssessmentExtractionError,
    extract_assessment,
)
from app.services.database import assessments_collection


router = APIRouter(
    prefix="/assessments",
    tags=["assessments"],
)


@router.post("/parse")
async def parse_assessment(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required",
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported",
        )

    temp_path = Path("clinical_assessment_upload.wav")

    try:
        contents = await file.read()
        temp_path.write_bytes(contents)

        transcript = transcribe_audio(temp_path)

        try:
            assessment = extract_assessment(transcript)
        except AssessmentExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Assessment could not be extracted "
                        "with sufficient confidence."
                    ),
                    "confidence": exc.confidence,
                    "missing_fields": exc.missing_fields,
                },
            )

        assessment_data = assessment.model_dump()

        result = assessments_collection.insert_one(assessment_data)

        assessment_data["_id"] = str(result.inserted_id)

        return assessment_data

    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.get("/{assessment_id}")
def get_assessment(assessment_id: str):
    if not ObjectId.is_valid(assessment_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment ID",
        )

    assessment = assessments_collection.find_one(
        {"_id": ObjectId(assessment_id)}
    )

    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    assessment["_id"] = str(assessment["_id"])

    return assessment


@router.get("/")
def list_assessments(
    date_from: datetime | None = None,
    date_to: datetime | None = None,
):
    query = {}

    if date_from is not None or date_to is not None:
        id_query = {}

        if date_from is not None:
            if date_from.tzinfo is None:
                date_from = date_from.replace(tzinfo=timezone.utc)

            id_query["$gte"] = ObjectId.from_datetime(date_from)

        if date_to is not None:
            if date_to.tzinfo is None:
                date_to = date_to.replace(tzinfo=timezone.utc)

            id_query["$lte"] = ObjectId.from_datetime(date_to)

        query["_id"] = id_query

    assessments = list(
        assessments_collection.find(query).sort("_id", -1)
    )

    for assessment in assessments:
        assessment["_id"] = str(assessment["_id"])

    return assessments