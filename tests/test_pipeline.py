from pathlib import Path

import pytest

from app.config import Settings
from app.errors import ExtractionConfidenceError
from app.pipeline.pipeline import parse_audio


def test_clinical_assessment_audio_fixture_is_optional():
    path = Path("data/clinical_assessment.wav")
    if not path.exists():
        pytest.skip("data/clinical_assessment.wav is not present")

    assert path.stat().st_size > 0


@pytest.mark.asyncio
async def test_parse_audio_empty_transcript(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")

    def fake_transcribe(path, settings):
        return ""

    monkeypatch.setattr("app.pipeline.pipeline.transcribe_wav", fake_transcribe)

    with pytest.raises(ExtractionConfidenceError) as exc:
        await parse_audio(audio_path, Settings(), extractor=lambda _: {})

    assert exc.value.low_confidence_fields == ["transcript"]
