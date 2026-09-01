import whisper

_model = None


def get_model(model_size: str = "base"):
    """
    Loads the Whisper model once and reuses it (avoids reloading on every request).
    """
    global _model
    if _model is None:
        _model = whisper.load_model(model_size)
    return _model


def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Transcribes a WAV file to plain text using local Whisper.
    Raises an exception if transcription fails or produces empty output.
    """
    model = get_model(model_size)
    result = model.transcribe(file_path, fp16=False)

    text = result.get("text", "").strip()
    if not text:
        raise ValueError("Transcription produced empty text — check audio quality or file path.")

    return text