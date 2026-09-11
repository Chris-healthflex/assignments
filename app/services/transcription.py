from pathlib import Path
import tempfile

import whisper


WHISPER_MODEL_NAME = "base"

_model = None


def get_whisper_model():
    global _model

    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_NAME)

    return _model


def transcribe_audio(audio_path: str | Path) -> str:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError("Only WAV audio files are supported.")

    model = get_whisper_model()

    result = model.transcribe(
        str(path),
        language="en",
        fp16=False,
    )

    transcript = result.get("text", "").strip()

    if not transcript:
        raise ValueError("Whisper returned an empty transcript.")

    return transcript