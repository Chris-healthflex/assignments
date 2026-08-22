import os
import tempfile
import logging
import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.assessment import FirstAssessment
from app.schemas.assessment import ParseResponse, ValidationErrorDetail
from app.services.transcription import transcribe_audio
from app.graph.assessment_graph import app_graph
from app.database.mongodb import (
    create_assessment,
    get_assessment_by_id,
    list_assessments,
    check_db_health
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assessments", tags=["Assessments"])

def db_doc_to_parse_response(doc: dict) -> ParseResponse:
    """
    Cleans database metadata out of a retrieved document and wraps it in ParseResponse.
    """
    db_id = str(doc["_id"])
    cleaned = {k: v for k, v in doc.items() if k not in ("_id", "created_at", "updated_at")}
    try:
        assessment = FirstAssessment.model_validate(cleaned)
        return ParseResponse(id=db_id, assessment=assessment)
    except Exception as e:
        logger.error(f"Failed to validate database document {db_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored assessment data format is invalid"
        )

@router.post(
    "/parse",
    response_model=ParseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Transcribe WAV audio and parse clinical assessment data"
)
async def parse_assessment(file: UploadFile = File(...)):
    """
    Parses a WAV audio file, runs transcription and extraction, and persists the assessment.
    """
    # 1. Validate extension
    filename = file.filename or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only WAV files are allowed"
        )

    # 2. Write upload to a safe temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(temp_fd, "wb") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded audio file is empty"
                )
            tmp.write(content)
            
        # 3. Transcribe audio
        try:
            transcript = transcribe_audio(temp_path)
        except ValueError as e:
            # Format/validation error
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.error(f"Transcription failure: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to transcribe audio file"
            )
            
        # 4. Invoke LangGraph workflow
        logger.info("Invoking LangGraph assessment pipeline...")
        try:
            state = app_graph.invoke({"transcript": transcript})
        except Exception as e:
            logger.error(f"LangGraph pipeline execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Extraction pipeline failed"
            )
            
        # 5. Check confidence/validation errors
        validation_errors = state.get("validation_errors") or []
        if validation_errors:
            logger.warning(f"Rejecting parse due to low confidence fields: {validation_errors}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": validation_errors}
            )
            
        # 6. Persist to MongoDB
        assessment_dict = state.get("first_assessment")
        if not assessment_dict:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Pipeline did not produce an assessment"
            )
            
        try:
            inserted_id = create_assessment(assessment_dict)
        except ConnectionError as e:
            # 503 if Mongo is down
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database persist failed. MongoDB Atlas is unreachable."
            )
        except Exception as e:
            logger.error(f"Failed to insert assessment: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist assessment data"
            )
            
        return ParseResponse(
            id=inserted_id,
            assessment=FirstAssessment.model_validate(assessment_dict)
        )
        
    finally:
        # Cleanup temporary audio file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

@router.get(
    "/{id}",
    response_model=ParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Get clinical assessment by DB ID"
)
def get_assessment(id: str):
    """
    Looks up a clinical assessment by its MongoDB ObjectID string.
    """
    try:
        doc = get_assessment_by_id(id)
    except ValueError as e:
        # Invalid ObjectId format
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid database ID format"
        )
    except ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable"
        )
        
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clinical assessment not found"
        )
        
    return db_doc_to_parse_response(doc)

@router.get(
    "",
    response_model=List[ParseResponse],
    status_code=status.HTTP_200_OK,
    summary="List clinical assessments with filters"
)
def get_assessments(
    start_date: Optional[str] = Query(None, description="Start date filter in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date filter in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Lists assessments sorted by created_at descending. Filters by start_date and end_date.
    """
    dt_start = None
    dt_end = None
    
    # Parse dates if provided
    if start_date:
        try:
            dt_start = datetime.datetime.fromisoformat(start_date)
            # Make timezone aware if not specified
            if dt_start.tzinfo is None:
                dt_start = dt_start.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date must be in ISO format"
            )
            
    if end_date:
        try:
            dt_end = datetime.datetime.fromisoformat(end_date)
            if dt_end.tzinfo is None:
                dt_end = dt_end.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="end_date must be in ISO format"
            )

    try:
        docs = list_assessments(limit=limit, offset=offset, start_date=dt_start, end_date=dt_end)
    except ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable"
        )

    results = []
    for doc in docs:
        try:
            results.append(db_doc_to_parse_response(doc))
        except Exception:
            # Skip invalid docs in listing to preserve reliability
            continue
            
    return results
