from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.parse_service import process_audio_file
from app.core.exceptions import ConfidenceError
from app.schemas.assessment import FirstAssessment

router = APIRouter()

@router.post("/assessments/parse", response_model=FirstAssessment, status_code=200)
async def parse_assessment(file: UploadFile = File(...)):
    """Transcribe WAV and extract structured FirstAssessment JSON."""
    try:
        return await process_audio_file(file)
    except ConfidenceError as e:
        raise HTTPException(status_code=422, detail=e.errors)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")