"""Tests for the transcription layer.

Deliberately no audio and no model download: what needs proving here is the
confidence conversion and the cache contract, both of which are pure logic.
The one test that does touch the real recording skips itself when the cache is
absent, so a fresh clone still runs green.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app import transcription
from app.schemas import TranscriptionResult, TranscriptSegment, TranscriptWord


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        text="ankle dorsiflexion of four point five degrees",
        language="en",
        durationSec=6.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=6.0,
                text="ankle dorsiflexion of four point five degrees",
                confidence=0.86,
                noSpeechProbability=0.01,
                words=[
                    TranscriptWord(start=0.0, end=0.5, word="ankle", confidence=0.97),
                    TranscriptWord(start=0.5, end=1.2, word="dorsiflexion", confidence=0.48),
                    TranscriptWord(start=1.2, end=1.4, word="of", confidence=0.99),
                ],
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Confidence conversion
# --------------------------------------------------------------------------- #
def test_logprob_becomes_a_real_probability():
    """A raw -0.15 in a field named "confidence" would be actively misleading."""
    assert transcription._logprob_to_probability(-0.15) == pytest.approx(math.exp(-0.15))
    assert transcription._logprob_to_probability(0.0) == 1.0


def test_logprob_conversion_is_clamped_and_null_safe():
    assert transcription._logprob_to_probability(None) == 0.0
    assert transcription._logprob_to_probability(-50.0) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= transcription._logprob_to_probability(0.7) <= 1.0


def test_a_bad_patch_scores_far_below_clean_speech():
    clean = transcription._logprob_to_probability(-0.15)
    bad = transcription._logprob_to_probability(-1.2)
    assert clean > 0.85 and bad < 0.35


# --------------------------------------------------------------------------- #
# Cache contract
# --------------------------------------------------------------------------- #
def test_cache_round_trip_preserves_word_confidences(tmp_path, monkeypatch):
    """The cache is worthless if it drops the probabilities we cached it for."""
    monkeypatch.setattr(transcription, "CACHE_DIR", tmp_path)
    target = tmp_path / "abc123.json"

    transcription._write_cache(target, _result())
    restored = transcription._read_cache(target)

    assert restored == _result()
    assert restored.segments[0].words[1].confidence == pytest.approx(0.48)


def test_cache_writes_a_readable_text_sibling(tmp_path):
    """A human has to be able to read what the model heard, without jq."""
    target = tmp_path / "abc123.json"
    transcription._write_cache(target, _result())

    sibling = target.with_suffix(".txt")
    assert sibling.read_text(encoding="utf-8") == _result().text
    assert json.loads(target.read_text(encoding="utf-8"))["segments"][0]["words"]


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")
    assert transcription._read_cache(target) is None


def test_missing_cache_returns_none(tmp_path):
    assert transcription._read_cache(tmp_path / "nope.json") is None


def test_cache_key_follows_the_bytes_not_the_filename(tmp_path):
    """A re-recorded file at the same path must miss, not serve a stale result."""
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    first.write_bytes(b"RIFF....one")
    second.write_bytes(b"RIFF....two")
    assert transcription._cache_key(first) != transcription._cache_key(second)

    second.write_bytes(b"RIFF....one")
    assert transcription._cache_key(first) == transcription._cache_key(second)


def test_missing_audio_file_is_a_clear_error(tmp_path):
    with pytest.raises(transcription.TranscriptionError, match="not found"):
        transcription.transcribe(tmp_path / "absent.wav")


# --------------------------------------------------------------------------- #
# Against the real recording (skipped when the cache is not present)
# --------------------------------------------------------------------------- #
AUDIO = Path("clinical_assessment.wav")
_cached = (
    transcription.cache_path_for(AUDIO) if AUDIO.is_file() else Path("nonexistent")
)

pytestmark_real = pytest.mark.skipif(
    not _cached.is_file(),
    reason="run `python -m app.transcription clinical_assessment.wav` first",
)


@pytestmark_real
def test_real_transcript_exposes_per_word_confidence():
    result = transcription.transcribe(AUDIO)
    words = [w for s in result.segments for w in s.words]
    assert len(words) > 200
    # The point of the whole module: the spread must be real. A transcript where
    # every word scores the same tells us nothing about which values to trust.
    assert min(w.confidence for w in words) < 0.5
    assert max(w.confidence for w in words) > 0.95


@pytestmark_real
def test_a_measurement_can_be_scored_by_its_quoted_span():
    """The Phase 3 handshake: quote a span, get the audio confidence for it."""
    result = transcription.transcribe(AUDIO)
    assert result.confidence_for("left knee flexion of 124 degrees") is not None
    assert result.confidence_for("patient reported severe nausea") is None
