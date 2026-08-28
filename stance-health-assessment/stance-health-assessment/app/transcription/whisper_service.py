"""
WAV -> text transcription.

Primary engine: OpenAI Whisper, run locally via `faster-whisper` (CTranslate2) or
the reference `openai-whisper` package. Either downloads model weights on first
use (from Hugging Face or openaipublic.azureedge.net respectively) — this
requires outbound network access to those hosts.

Fallback engine: `pocketsphinx`, whose small acoustic model ships *inside* the
PyPI wheel, so it needs zero network access at runtime. It is meaningfully
less accurate than Whisper (no punctuation, weaker on medical/clinical
vocabulary, more substitution errors) — it exists purely so this module is
runnable end-to-end in network-locked environments (e.g. CI sandboxes,
offline dev boxes). Production deployments should always run with
TRANSCRIPTION_ENGINE=whisper.

Engine is selected via the TRANSCRIPTION_ENGINE env var: "whisper" (default)
or "pocketsphinx".
"""

from __future__ import annotations

import os
import wave
import audioop
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    text: str
    engine: str
    language: str = "en"
    low_confidence: bool = False


def _resample_to_16k_mono(src_path: str, dst_path: str) -> None:
    with wave.open(src_path, "rb") as src:
        frames = src.readframes(src.getnframes())
        converted, _ = audioop.ratecv(
            frames,
            src.getsampwidth(),
            src.getnchannels(),
            src.getframerate(),
            16000,
            None,
        )
        # downmix to mono if needed
        if src.getnchannels() == 2:
            converted = audioop.tomono(converted, src.getsampwidth(), 0.5, 0.5)

    with wave.open(dst_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(converted)


def _transcribe_whisper(wav_path: str, model_size: str = "small") -> TranscriptionResult:
    """Real Whisper transcription. Requires network access to fetch model weights
    on first run (cached locally after that)."""
    from faster_whisper import WhisperModel  # lazy import: heavy dependency

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path, beam_size=5, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    return TranscriptionResult(text=text.strip(), engine=f"whisper-{model_size}", language=info.language)


def _transcribe_pocketsphinx(wav_path: str) -> TranscriptionResult:
    """Offline fallback with zero network dependency. Lower accuracy than Whisper —
    callers should treat the result as low_confidence."""
    from pocketsphinx import Decoder

    tmp_path = wav_path + ".16k.wav"
    _resample_to_16k_mono(wav_path, tmp_path)

    decoder = Decoder()
    try:
        with open(tmp_path, "rb") as f:
            f.read(44)  # skip WAV header
            decoder.start_utt()
            while True:
                buf = f.read(4096)
                if not buf:
                    break
                decoder.process_raw(buf, False, False)
            decoder.end_utt()
        hyp = decoder.hyp()
        text = hyp.hypstr if hyp else ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return TranscriptionResult(text=text.strip(), engine="pocketsphinx-en-us", low_confidence=True)


def transcribe(wav_path: str, engine: str | None = None) -> TranscriptionResult:
    """Entry point used by the API layer and the LangGraph agent's first node."""
    engine = engine or os.getenv("TRANSCRIPTION_ENGINE", "whisper")

    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    if engine == "whisper":
        try:
            return _transcribe_whisper(wav_path, model_size=os.getenv("WHISPER_MODEL_SIZE", "small"))
        except Exception as e:  # noqa: BLE001 - surfaced to caller, not swallowed silently
            raise RuntimeError(
                "Whisper transcription failed (likely missing model weights / no "
                "network access to download them). Set TRANSCRIPTION_ENGINE="
                "pocketsphinx to use the offline fallback, or ensure network "
                "access to Hugging Face / openaipublic.azureedge.net."
            ) from e

    if engine == "pocketsphinx":
        return _transcribe_pocketsphinx(wav_path)

    raise ValueError(f"Unknown TRANSCRIPTION_ENGINE: {engine}")
