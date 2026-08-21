import tempfile
import asyncio
from pathlib import Path
from fastapi import UploadFile, HTTPException

from app.core.config import settings
from app.core.exceptions import AudioValidationError, AudioProcessingError, ConfidenceError
from app.core.logging import logger
from app.audio.validation import validate_wav_bytes, get_wav_duration_seconds
from app.audio.transcription import transcribe_wav
from app.agent.graph import build_extraction_graph
from app.agent.state import AgentState
from app.schemas.extraction import ExtractionResult
from app.guardrails.confidence_gate import apply_source_verification, collect_field_errors
from app.schemas.assessment import FirstAssessment
from app.services.assembly import assemble_first_assessment  # we'll define this below

async def process_audio_file(file: UploadFile) -> FirstAssessment:
    """Run full pipeline: validate -> transcribe -> extract -> verify -> assemble -> gate."""
    # 1. Read and validate
    data = await file.read()
    try:
        validate_wav_bytes(data, file.filename or "upload.wav")
    except AudioValidationError as e:
        raise e

    # 2. Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # 3. Check duration
        duration = await asyncio.to_thread(get_wav_duration_seconds, tmp_path)
        if duration > settings.MAX_AUDIO_DURATION_SEC:
            raise HTTPException(status_code=422, detail="Audio duration exceeds limit")
        if duration <= 0:
            raise HTTPException(status_code=422, detail="Invalid audio duration")

        # 4. Transcribe
        transcript = await transcribe_wav(tmp_path)
        if not transcript.strip():
            raise HTTPException(status_code=422, detail="No speech detected in audio")

        # 5. Run LangGraph extraction agent
        initial_state: AgentState = {
            "transcript": transcript,
            "result": ExtractionResult(),
            "retry_count": 0,
            "section_errors": [],
            "retry_needed": False
        }
        graph = build_extraction_graph()
        final_state = await graph.ainvoke(initial_state)
        extraction_result = final_state["result"]

        # 6. Apply source verification (fuzzy matching)
        extraction_result = apply_source_verification(extraction_result, transcript)

        # 7. Assemble public FirstAssessment
        first_assessment = assemble_first_assessment(extraction_result, transcript)

        # 8. Confidence gate
        errors = collect_field_errors(extraction_result)
        if errors:
            raise ConfidenceError(errors)

        logger.info(f"Pipeline successful for file: {file.filename}")
        return first_assessment

    finally:
        Path(tmp_path).unlink(missing_ok=True)