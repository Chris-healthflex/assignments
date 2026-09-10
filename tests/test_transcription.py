from pathlib import Path

from app.services.transcription import transcribe_audio


AUDIO_FILE = Path("clinical_assessment.wav")


def test_transcription():
    assert AUDIO_FILE.exists(), "clinical_assessment.wav is missing"

    transcript = transcribe_audio(AUDIO_FILE)

    assert isinstance(transcript, str)
    assert transcript.strip()

    print("\n--- TRANSCRIPT ---")
    print(transcript)
    print("--- END TRANSCRIPT ---")