from app.pipeline.mapping import low_confidence_fields
from app.pipeline.transcription import TranscriptionError, validate_wav


def test_low_confidence_fields_are_reported() -> None:
    issues = low_confidence_fields({"clinicalDetails.duration": 0.4, "chiefComplaint": 0.9}, 0.75)
    assert issues[0]["field"] == "clinicalDetails.duration"


def test_non_wav_input_is_rejected(tmp_path) -> None:
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"audio")
    try:
        validate_wav(path)
    except TranscriptionError as exc:
        assert "WAV" in str(exc)
    else:
        raise AssertionError("Non-WAV input must be rejected")
