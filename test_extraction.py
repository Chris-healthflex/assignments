import sys
from pathlib import Path

from app.graph.clinical_graph import clinical_graph
from app.services.transcription import transcribe_audio


def main():
    audio_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "clinical_assessment.wav"
    )

    print(f"Transcribing: {audio_path}")

    transcript = transcribe_audio(audio_path)

    result = clinical_graph.invoke({
        "transcript": transcript
    })

    if result.get("error"):
        raise RuntimeError(result["error"])

    assessment = result["assessment"]

    print("\n===== TRANSCRIPT =====\n")
    print(transcript)

    print("\n===== FIRST ASSESSMENT =====\n")
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()