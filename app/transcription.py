"""Whisper transcription: WAV in, text plus per-word confidence out.

Two things make this more than a thin wrapper around faster-whisper.

**We keep the probabilities.** Whisper reports how sure it was about every word
it emitted, and a clinical pipeline throws that away at its peril: "forty
degrees" and "fourteen degrees" sound alike, and the extraction agent will
happily pull the wrong number out of a wrong transcript with total confidence.
Those probabilities are what later lets us say *"this measurement came from a
span the model was only 41% sure of"*. See `FieldEvidence` in `app.schemas`.

**We cache to disk.** Transcription is the slow, deterministic, expensive stage;
prompt-tuning the extraction agent is the fast, iterative one. Re-running
Whisper on every prompt tweak would make the inner loop minutes long instead of
seconds. The cache key covers the audio bytes *and* the decode settings, so
changing the model invalidates it rather than silently serving a stale result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.schemas import TranscriptionResult, TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache") / "transcripts"


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def _logprob_to_probability(avg_logprob: float | None) -> float:
    """Convert Whisper's average log-probability into a plain 0-1 probability.

    faster-whisper reports segment quality as a mean log-probability -- a
    negative number, typically around -0.15 for clean speech and below -1.0 for
    a bad patch. That is not comparable to anything else in the pipeline until
    it is exponentiated, and a raw -0.4 sitting in a field called "confidence"
    would be actively misleading.
    """
    if avg_logprob is None:
        return 0.0
    return max(0.0, min(1.0, math.exp(avg_logprob)))


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_key(path: Path) -> str:
    """Fingerprint the audio *and* the settings that decode it.

    Hashing the file contents rather than its name means a re-recorded file at
    the same path is a cache miss, which is the safe direction to be wrong in.
    """
    settings = get_settings()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(
        f"|{settings.whisper_backend}|{settings.whisper_model}|{settings.whisper_language}".encode()
    )
    return digest.hexdigest()[:16]


def cache_path_for(audio_path: str | Path) -> Path:
    return CACHE_DIR / f"{_cache_key(Path(audio_path))}.json"


def _read_cache(path: Path) -> TranscriptionResult | None:
    if not path.is_file():
        return None
    try:
        return TranscriptionResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- a stale cache must never break a run
        logger.warning("Ignoring unreadable transcript cache at %s", path)
        return None


def _write_cache(path: Path, result: TranscriptionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    # A plain-text sibling so a human can actually read what the model heard.
    # Verifying that the agent is not inventing data starts with reading this.
    path.with_suffix(".txt").write_text(result.text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@lru_cache
def _load_model():
    """Load the Whisper model once per process (it is several hundred MB)."""
    from faster_whisper import WhisperModel  # imported lazily: heavy dependency

    settings = get_settings()
    logger.info(
        "Loading Whisper model=%s device=%s",
        settings.whisper_model,
        settings.whisper_device,
    )
    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type="int8" if settings.whisper_device == "cpu" else "float16",
    )


def _to_segment(raw) -> TranscriptSegment:
    words = [
        TranscriptWord(
            start=w.start,
            end=w.end,
            word=w.word,
            confidence=max(0.0, min(1.0, w.probability)),
        )
        for w in (raw.words or [])
    ]
    return TranscriptSegment(
        start=raw.start,
        end=raw.end,
        text=raw.text.strip(),
        confidence=_logprob_to_probability(getattr(raw, "avg_logprob", None)),
        noSpeechProbability=max(0.0, min(1.0, getattr(raw, "no_speech_prob", 0.0) or 0.0)),
        words=words,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def transcribe(audio_path: str | Path, *, use_cache: bool = True) -> TranscriptionResult:
    """Transcribe an audio file into text, timed segments and word confidences."""
    path = Path(audio_path)
    if not path.is_file():
        raise TranscriptionError(f"Audio file not found: {path}")

    settings = get_settings()
    if settings.whisper_backend == "api":
        # Deliberately not wired: this is patient audio, and shipping PHI to a
        # third party is a decision for a compliance review, not a default.
        raise NotImplementedError(
            "Hosted Whisper backend is not enabled; set WHISPER_BACKEND=local."
        )

    cache_file = cache_path_for(path)
    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            logger.info("Transcript cache hit for %s (%s)", path.name, cache_file.name)
            return cached

    model = _load_model()
    segments, info = model.transcribe(
        str(path),
        language=settings.whisper_language,
        # The whole point of this module: without word timestamps there are no
        # per-word probabilities, and no way to score a quoted span later.
        word_timestamps=True,
        vad_filter=True,
    )

    collected = [_to_segment(s) for s in segments]  # generator: consumed once
    text = " ".join(s.text for s in collected).strip()
    if not text:
        raise TranscriptionError(f"Whisper produced no text for {path.name}")

    result = TranscriptionResult(
        text=text,
        language=getattr(info, "language", ""),
        durationSec=getattr(info, "duration", 0.0) or 0.0,
        segments=collected,
    )
    _write_cache(cache_file, result)
    return result


def _main() -> int:
    """Standalone runner: `python -m app.transcription clinical_assessment.wav`."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Transcribe a WAV file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--no-cache", action="store_true", help="Force a fresh run")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(message)s")
    started = time.perf_counter()
    result = transcribe(args.audio, use_cache=not args.no_cache)
    elapsed = time.perf_counter() - started

    # The cache is keyed by content hash, which is right for correctness and
    # useless for a human trying to find the file. Drop a copy at a predictable
    # path so the transcript can just be opened and read.
    latest = Path("transcript.txt")
    latest.write_text(result.text, encoding="utf-8")

    words = [w for s in result.segments for w in s.words]
    weak = sorted(words, key=lambda w: w.confidence)[:10]
    print(f"\n{result.text}\n")
    print("-" * 70)
    print(
        f"{len(result.text.split())} words, {len(result.segments)} segments, "
        f"{result.durationSec:.1f}s audio, lang={result.language}, in {elapsed:.1f}s"
    )
    print(f"cache:      {cache_path_for(args.audio)}")
    print(f"readable:   {latest}")
    if weak:
        print("\nLeast confident words (the ones worth double-checking):")
        for w in weak:
            print(f"  {w.confidence:5.0%}  {w.word.strip()!r}  @{w.start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
