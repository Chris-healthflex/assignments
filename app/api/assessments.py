import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.services.transcription import transcribe_audio
from app.services.extraction import extract_clinical_data
from app.services.mapper import map_to_first_assessment
from app.core.exceptions import LowConfidenceExtractionError
from app.db.repository import save_assessment, get_assessment_by_id, list_assessments

router = APIRouter(prefix="/assessments", tags=["assessments"])

TEMP_DIR = "data/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/parse")
async def parse_assessment(file: UploadFile = File(...)):
    """
    EP1: Accept a WAV file, run the full pipeline (transcribe -> extract -> map),
    and return the FirstAssessment JSON. Returns 422 if extraction confidence is low.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.wav")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        transcript = transcribe_audio(temp_path)
        raw_extraction = extract_clinical_data(transcript)

        try:
            assessment = map_to_first_assessment(raw_extraction)
        except LowConfidenceExtractionError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Extraction confidence below threshold.",
                    "missing_fields": e.missing_fields,
                },
            )

        return assessment.model_dump()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("")
async def create_assessment(assessment: dict):
    """
    EP2: Save an already-parsed FirstAssessment JSON to MongoDB.
    """
    from app.schemas.first_assessment import FirstAssessment

    try:
        validated = FirstAssessment(**assessment)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FirstAssessment payload: {e}")

    inserted_id = save_assessment(validated)
    return {"id": inserted_id}


@router.get("/{assessment_id}")
async def get_assessment(assessment_id: str):
    """
    EP3: Retrieve a saved assessment by ID.
    """
    doc = get_assessment_by_id(assessment_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return doc


@router.get("")
async def list_all_assessments(date: str | None = Query(None, description="Filter by date YYYY-MM-DD")):
    """
    EP4: List all assessments, optionally filtered by date.
    """
    return list_assessments(date_filter=date)