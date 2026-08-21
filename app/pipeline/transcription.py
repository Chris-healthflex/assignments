from importlib import import_module
from pathlib import Path


class TranscriptionError(RuntimeError):
    pass


def validate_wav(path: Path) -> None:
    if path.suffix.lower() != ".wav":
        raise TranscriptionError("Only WAV audio files are accepted")
    if not path.is_file() or path.stat().st_size == 0:
        raise TranscriptionError("The WAV file is missing or empty")


class WhisperTranscriber:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: Path) -> str:
        validate_wav(audio_path)
        try:
            whisper = import_module("whisper")
        except ImportError as exc:
            raise TranscriptionError(
                "Whisper is not installed; install project dependencies before parsing audio"
            ) from exc

        if self._model is None:
            self._model = whisper.load_model(self.model_name)
        result = self._model.transcribe(str(audio_path))
        text = str(result.get("text", "")).strip()
        if not text:
            raise TranscriptionError("Whisper returned an empty transcript")
        return text
