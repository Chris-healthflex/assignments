import os
import tempfile
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException
from bson import ObjectId

from app.schemas import FirstAssessment
from app.extractor import (
    extract_clinical_assessment,
    ExtractionConfidenceError
)
from app.whisper_service import transcribe_audio
from app.database import assessments_collection


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(
    title="Clinical Assessment API",
    version="1.0.0"
)


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Clinical Assessment API is running"
    }


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# EP1
# POST /assessments/parse
# WAV → WHISPER → GROQ → FirstAssessment JSON
# =========================================================

@app.post(
    "/assessments/parse",
    response_model=FirstAssessment
)
async def parse_assessment(
    file: UploadFile = File(...)
):

    temp_audio_path = None

    try:

        # -------------------------------------------------
        # 1. Validate uploaded file
        # -------------------------------------------------

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected."
            )

        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(
                status_code=400,
                detail="Only WAV audio files are supported."
            )

        # -------------------------------------------------
        # 2. Read uploaded WAV
        # -------------------------------------------------

        audio_data = await file.read()

        if not audio_data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded WAV file is empty."
            )

        # -------------------------------------------------
        # 3. Save temporary WAV
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(audio_data)
            temp_audio_path = temp_file.name

        # -------------------------------------------------
        # 4. Whisper transcription
        # -------------------------------------------------

        transcription = transcribe_audio(
            temp_audio_path
        )

        if not transcription or not transcription.strip():
            raise HTTPException(
                status_code=400,
                detail="Whisper could not generate a transcription."
            )

        # -------------------------------------------------
        # 5. Groq extraction
        # -------------------------------------------------

        try:

            assessment = extract_clinical_assessment(
                transcription
            )

        except ExtractionConfidenceError as e:

            # Required assignment behaviour:
            # HTTP 422 with field-level confidence errors

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Extraction confidence below threshold.",
                    "fields": e.fields
                }
            )

        # -------------------------------------------------
        # 6. Return parsed assessment
        # -------------------------------------------------

        return assessment

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # -------------------------------------------------
        # 7. Delete temporary WAV
        # -------------------------------------------------

        if temp_audio_path and os.path.exists(temp_audio_path):

            try:
                os.remove(temp_audio_path)

            except Exception:
                pass


# =========================================================
# EP2
# POST /assessments
# Save parsed FirstAssessment → MongoDB
# =========================================================

@app.post("/assessments")
def save_assessment(
    assessment: FirstAssessment
):

    try:

        assessment_data = assessment.model_dump()

        document = {
            "assessment": assessment_data,
            "createdAt": datetime.now(timezone.utc)
        }

        result = assessments_collection.insert_one(
            document
        )

        return {
            "message": "Assessment saved successfully",
            "assessment_id": str(result.inserted_id)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# EP3
# GET /assessments/{assessment_id}
# Retrieve saved assessment by ID
# =========================================================

@app.get("/assessments/{assessment_id}")
def get_assessment(
    assessment_id: str
):

    try:

        if not ObjectId.is_valid(assessment_id):

            raise HTTPException(
                status_code=400,
                detail="Invalid assessment ID."
            )

        document = assessments_collection.find_one(
            {
                "_id": ObjectId(assessment_id)
            }
        )

        if not document:

            raise HTTPException(
                status_code=404,
                detail="Assessment not found."
            )

        document["_id"] = str(
            document["_id"]
        )

        return document

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# EP4
# GET /assessments
#
# Optional:
# /assessments?date=2026-09-01
# =========================================================

@app.get("/assessments")
def get_assessments(
    date: str | None = None
):

    try:

        query = {}

        # -------------------------------------------------
        # Optional date filter
        # -------------------------------------------------

        if date:

            try:

                selected_date = datetime.strptime(
                    date,
                    "%Y-%m-%d"
                )

            except ValueError:

                raise HTTPException(
                    status_code=400,
                    detail="Date must be in YYYY-MM-DD format."
                )

            start_date = selected_date.replace(
                tzinfo=timezone.utc
            )

            end_date = start_date + timedelta(
                days=1
            )

            query = {
                "createdAt": {
                    "$gte": start_date,
                    "$lt": end_date
                }
            }

        # -------------------------------------------------
        # Fetch assessments
        # -------------------------------------------------

        documents = list(
            assessments_collection.find(
                query
            ).sort(
                "createdAt",
                -1
            )
        )

        # -------------------------------------------------
        # Convert ObjectId → string
        # -------------------------------------------------

        for document in documents:

            document["_id"] = str(
                document["_id"]
            )

        return documents

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )