import json

from app.database.mongodb import MongoDB
from app.graph.assessment_graph import (
    build_assessment_graph,
)
from app.services.extraction import CONFIDENCE_THRESHOLD
from app.services.transcription import WhisperTranscriber


AUDIO_PATH = "data/clinical_assessment.wav"


def main():
    # ============================================================
    # 1. TRANSCRIPTION
    # ============================================================

    print("Loading Whisper model...")

    transcriber = WhisperTranscriber(
        model_name="base",
    )

    print("Transcribing audio...")

    transcript = transcriber.transcribe(
        AUDIO_PATH,
    )

    if not transcript or not transcript.strip():
        raise RuntimeError(
            "Whisper returned an empty transcript."
        )

    print("\n--- TRANSCRIPT ---\n")
    print(transcript)

    # ============================================================
    # 2. LANGGRAPH
    # ============================================================

    print("\n--- RUNNING LANGGRAPH ---\n")

    graph = build_assessment_graph()

    result = graph.invoke(
        {
            "transcript": transcript,
        }
    )

    assessment = result.get("assessment")
    confidence = result.get("confidence")

    if assessment is None:
        raise RuntimeError(
            "LangGraph did not return an assessment."
        )

    if confidence is None:
        raise RuntimeError(
            "LangGraph did not return confidence information."
        )

    # ============================================================
    # 3. CONFIDENCE
    # ============================================================

    print("\n--- CONFIDENCE ---\n")

    print(
        confidence.model_dump_json(
            indent=2,
        )
    )
    print("\n--- FIRST ASSESSMENT BEFORE CONFIDENCE REJECTION ---\n")
    print(
       assessment.model_dump_json(
          indent=2,
    )
)

    low_confidence_issues = [
        issue
        for issue in confidence.issues
        if issue.confidence < CONFIDENCE_THRESHOLD
    ]

    # ============================================================
    # 4. REJECTION PATH
    # ============================================================

    if (
        confidence.overall_confidence
        < CONFIDENCE_THRESHOLD
        or low_confidence_issues
    ):
        print(
            "\nAssessment rejected because one or more "
            "fields are below the confidence threshold."
        )

        for issue in low_confidence_issues:
            print(
                f"- {issue.field_path}: "
                f"{issue.confidence:.2f} "
                f"({issue.reason})"
            )

        print(
            "\nNo MongoDB save was performed."
        )

        return

    # ============================================================
    # 5. VALID ASSESSMENT
    # ============================================================

    print("\n--- FIRST ASSESSMENT ---\n")

    print(
        assessment.model_dump_json(
            indent=2,
        )
    )

    # ============================================================
    # 6. SAVE ONLY AFTER CONFIDENCE PASSES
    # ============================================================

    print("\n--- SAVING TO MONGODB ---\n")

    database = MongoDB()

    assessment_id = database.save_assessment(
        assessment,
    )

    print(
        "Assessment saved successfully."
    )

    print(
        "Assessment ID:",
        assessment_id,
    )

    # ============================================================
    # 7. ROUND-TRIP VERIFICATION
    # ============================================================

    print("\n--- VERIFYING MONGODB ROUND TRIP ---\n")

    saved = database.get_assessment(
        assessment_id,
    )

    if saved is None:
        raise RuntimeError(
            "Assessment was saved but could not be retrieved."
        )

    print(
        json.dumps(
            saved,
            indent=2,
            default=str,
        )
    )

    print(
        "\nEnd-to-end pipeline verification passed."
    )


if __name__ == "__main__":
    main()