import os
import tempfile
from datetime import date, datetime, time, timezone
from typing import Optional

from bson import ObjectId
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field

from app.database.mongodb import MongoDB
from app.graph.assessment_graph import build_assessment_graph
from app.models.assessment import FirstAssessment
from app.services.extraction import CONFIDENCE_THRESHOLD
from app.services.transcription import WhisperTranscriber


router = APIRouter(
    prefix="/assessments",
    tags=["Assessments"],
)


# ============================================================
# Response models
# ============================================================


class AssessmentCreateResponse(BaseModel):
    id: str
    message: str


class ConfidenceIssueResponse(BaseModel):
    field_path: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reason: str


class LowConfidenceDetail(BaseModel):
    error: str
    threshold: float
    issues: list[ConfidenceIssueResponse]


class LowConfidenceResponse(BaseModel):
    detail: LowConfidenceDetail


class ErrorResponse(BaseModel):
    detail: str


class StoredAssessment(FirstAssessment):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    assessment_id: str = Field(
        default="",
        alias="_id",
    )

    createdAt: Optional[datetime] = None


# ============================================================
# Confidence handling
# ============================================================


def build_low_confidence_error(confidence) -> dict:
    issues = [
        issue
        for issue in confidence.issues
        if issue.confidence < CONFIDENCE_THRESHOLD
    ]

    if not issues:
        issues = [
            type(
                "FallbackConfidenceIssue",
                (),
                {
                    "field_path": "assessment",
                    "confidence": confidence.overall_confidence,
                    "reason": (
                        "Overall extraction confidence is below "
                        "the configured threshold."
                    ),
                },
            )()
        ]

    return {
        "error": "Low-confidence clinical extraction.",
        "threshold": CONFIDENCE_THRESHOLD,
        "issues": [
            {
                "field_path": issue.field_path,
                "confidence": issue.confidence,
                "reason": issue.reason,
            }
            for issue in issues
        ],
    }


def validate_confidence_or_raise(confidence) -> None:
    overall_failure = (
        confidence.overall_confidence
        < CONFIDENCE_THRESHOLD
    )

    field_failure = any(
        issue.confidence < CONFIDENCE_THRESHOLD
        for issue in confidence.issues
    )

    if overall_failure or field_failure:
        raise HTTPException(
            status_code=422,
            detail=build_low_confidence_error(
                confidence
            ),
        )


# ============================================================
# POST /assessments/parse
# ============================================================


@router.post(
    "/parse",
    response_model=FirstAssessment,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid WAV upload.",
        },
        422: {
            "model": LowConfidenceResponse,
            "description": (
                "Clinical extraction failed the "
                "configured confidence threshold."
            ),
        },
        500: {
            "model": ErrorResponse,
            "description": "Unexpected processing failure.",
        },
    },
)
async def parse_assessment(
    file: UploadFile = File(
        ...,
        description="Clinical assessment WAV file.",
    ),
):
    """
    Transcribe a WAV file and extract a validated clinical
    assessment.

    This endpoint does not save the assessment.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported.",
        )

    temp_path: Optional[str] = None

    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=".wav",
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        transcriber = WhisperTranscriber(
            model_name="base",
        )

        transcript = transcriber.transcribe(
            temp_path
        )

        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="No usable transcript could be produced.",
            )

        graph = build_assessment_graph()

        result = graph.invoke(
            {
                "transcript": transcript,
            }
        )

        assessment = result.get("assessment")
        confidence = result.get("confidence")

        if assessment is None:
            raise HTTPException(
                status_code=500,
                detail="Assessment extraction produced no result.",
            )

        if confidence is None:
            raise HTTPException(
                status_code=500,
                detail="Confidence verification produced no result.",
            )

        validate_confidence_or_raise(
            confidence
        )

        return assessment

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Assessment parsing failed: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Assessment parsing failed.",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# POST /assessments
# ============================================================


@router.post(
    "",
    response_model=AssessmentCreateResponse,
    status_code=201,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Assessment could not be saved.",
        },
    },
)
async def create_assessment(
    assessment: FirstAssessment,
):
    try:
        database = MongoDB()

        assessment_id = database.save_assessment(
            assessment
        )

        return AssessmentCreateResponse(
            id=assessment_id,
            message="Assessment saved successfully",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Assessment save failed.",
        ) from exc


# ============================================================
# GET /assessments
# ============================================================


@router.get(
    "",
    response_model=list[StoredAssessment],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid date range.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Assessment retrieval failed.",
        },
    },
)
async def list_assessments(
    date_from: Optional[date] = Query(
        default=None
    ),
    date_to: Optional[date] = Query(
        default=None
    ),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=400,
            detail="date_from cannot be later than date_to.",
        )

    try:
        database = MongoDB()

        query = {}

        if date_from or date_to:
            created_at_query = {}

            if date_from:
                created_at_query["$gte"] = datetime.combine(
                    date_from,
                    time.min,
                    tzinfo=timezone.utc,
                )

            if date_to:
                created_at_query["$lte"] = datetime.combine(
                    date_to,
                    time.max,
                    tzinfo=timezone.utc,
                )

            query["createdAt"] = created_at_query

        assessments = list(
            database.collection.find(
                query
            ).sort(
                "createdAt",
                -1,
            )
        )

        for assessment in assessments:
            if "_id" in assessment:
                assessment["_id"] = str(
                    assessment["_id"]
                )

        return assessments

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve assessments.",
        ) from exc


# ============================================================
# GET /assessments/{assessment_id}
# ============================================================


@router.get(
    "/{assessment_id}",
    response_model=StoredAssessment,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid MongoDB assessment ID.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Assessment not found.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Assessment retrieval failed.",
        },
    },
)
async def get_assessment(
    assessment_id: str,
):
    if not ObjectId.is_valid(
        assessment_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment ID.",
        )

    try:
        database = MongoDB()

        assessment = database.get_assessment(
            assessment_id
        )

        if assessment is None:
            raise HTTPException(
                status_code=404,
                detail="Assessment not found.",
            )

        return assessment

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve assessment.",
        ) from exc