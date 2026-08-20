from fastapi import UploadFile
from app.core.logging import logger
from app.schemas.first_assessment import FirstAssessment
from app.models.assessment import AssessmentDocument, SaveAssessmentResponse, AssessmentListResponse, AssessmentSummaryItem
from app.repositories.assessment_repository import AssessmentRepository, get_assessment_repository
from app.services.audio_validator import AudioValidator, get_audio_validator
from app.services.transcription import BaseWhisperTranscriber, get_transcriber
from app.services.extraction import ClinicalExtractionService, get_extraction_service


class AssessmentService:
    """Orchestrates audio validation, transcription, extraction, and database persistence."""

    def __init__(
        self,
        audio_validator: AudioValidator | None = None,
        transcriber: BaseWhisperTranscriber | None = None,
        extraction_service: ClinicalExtractionService | None = None,
        repository: AssessmentRepository | None = None,
    ) -> None:
        self.audio_validator = audio_validator or get_audio_validator()
        self.transcriber = transcriber or get_transcriber()
        self.extraction_service = extraction_service or get_extraction_service()
        self.repository = repository or get_assessment_repository()

    async def parse_audio(self, file: UploadFile) -> FirstAssessment:
        """
        Executes complete pipeline:
        WAV Upload -> Validation -> Whisper Transcription -> LangGraph Extraction -> FirstAssessment JSON
        """
        logger.info("Starting audio assessment parse for upload: '%s'", file.filename)

        # Stage 1: Audio Validation
        audio_bytes = await self.audio_validator.validate_upload(file)

        # Stage 2: Whisper Transcription
        transcript = await self.transcriber.transcribe(audio_bytes, filename=file.filename or "audio.wav")

        # Stage 3 & 4: LangGraph Clinical Extraction & Pydantic Validation
        first_assessment = await self.extraction_service.extract_assessment(transcript)

        logger.info("Successfully completed audio parsing into FirstAssessment model.")
        return first_assessment

    async def parse_dual_audio(self, doctor_file: UploadFile, patient_file: UploadFile) -> FirstAssessment:
        """
        Executes dual-stream sequential pipeline:
        Transcribes Doctor's audio and Patient's response audio separately,
        combines them into an attributed consultation transcript,
        and extracts structured clinical entities into FirstAssessment JSON.
        """
        logger.info(
            "Starting dual audio assessment parse: doctor='%s', patient='%s'",
            doctor_file.filename, patient_file.filename
        )

        # Stage 1: Validate both WAV files
        doctor_bytes = await self.audio_validator.validate_upload(doctor_file)
        patient_bytes = await self.audio_validator.validate_upload(patient_file)

        # Stage 2: Transcribe both files via Whisper
        doctor_transcript = await self.transcriber.transcribe(doctor_bytes, filename=doctor_file.filename or "doctor.wav")
        patient_transcript = await self.transcriber.transcribe(patient_bytes, filename=patient_file.filename or "patient.wav")

        # Stage 3: Combine sequentially with speaker attribution
        combined_transcript = (
            f"Doctor:\n{doctor_transcript.strip()}\n\n"
            f"Patient:\n{patient_transcript.strip()}"
        )
        logger.info("Combined sequential doctor-patient transcript (%d characters)", len(combined_transcript))

        # Stage 4: LangGraph Clinical Extraction & Pydantic Validation
        first_assessment = await self.extraction_service.extract_assessment(combined_transcript)
        logger.info("Successfully completed dual audio parsing into FirstAssessment model.")
        return first_assessment


    async def save_assessment(self, assessment: FirstAssessment) -> SaveAssessmentResponse:
        """Persists a FirstAssessment instance to MongoDB."""
        doc: AssessmentDocument = self.repository.save_assessment(assessment)
        return SaveAssessmentResponse(
            id=doc.id,
            message="Assessment saved successfully",
            created_at=doc.created_at.isoformat()
        )

    async def get_assessment(self, assessment_id: str) -> AssessmentDocument:
        """Retrieves a single assessment by ID."""
        return self.repository.get_assessment(assessment_id)

    async def list_assessments(self, date_filter: str | None = None) -> AssessmentListResponse:
        """Lists stored assessments with optional date filter."""
        docs = self.repository.list_assessments(date_filter=date_filter)
        items = [
            AssessmentSummaryItem(
                id=doc.id,
                created_at=doc.created_at.isoformat(),
                chiefComplaint=doc.assessment.clinicalDetails.chiefComplaint,
                assessment=doc.assessment
            )
            for doc in docs
        ]
        return AssessmentListResponse(total=len(items), assessments=items)


def get_assessment_service() -> AssessmentService:
    """Dependency provider for AssessmentService."""
    return AssessmentService()
