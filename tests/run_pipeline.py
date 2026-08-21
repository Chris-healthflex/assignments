import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.pipeline.transcription import WhisperTranscriber, TranscriptionError
from app.pipeline.extraction import ClinicalExtractionGraph
from app.pipeline.mapping import map_and_validate, low_confidence_fields

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))

WAV_PATH = Path(__file__).parent.parent / "clinical_assessment.wav"


def main() -> None:
    print(f"Transcribing {WAV_PATH.name} with Whisper model '{WHISPER_MODEL}'...")
    transcriber = WhisperTranscriber(WHISPER_MODEL)
    try:
        transcript = transcriber.transcribe(WAV_PATH)
    except TranscriptionError as exc:
        print(f"Transcription failed: {exc}")
        return

    print(f"Transcript ({len(transcript)} chars):\n{transcript}\n")

    print(f"Running extraction with '{EXTRACTION_MODEL}'...")
    graph = ClinicalExtractionGraph(EXTRACTION_MODEL, GROQ_API_KEY)
    state = graph.extract(transcript)

    confidence = state.get("confidence", {})
    print("Field confidence:")
    print(json.dumps(confidence, indent=2))
    print()

    assessment, grounding_issues = map_and_validate(state["assessment"], transcript)
    for issue in grounding_issues:
        confidence[issue["field"]] = 0.0
    issues = low_confidence_fields(confidence, CONFIDENCE_THRESHOLD)
    if issues:
        print("Low-confidence fields detected:")
        print(json.dumps(issues, indent=2))
        return

    print("FirstAssessment JSON:")
    print(json.dumps(assessment.model_dump(), indent=2))


if __name__ == "__main__":
    main()