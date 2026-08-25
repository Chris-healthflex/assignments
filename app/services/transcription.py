import whisper


class WhisperTranscriber:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(
            audio_path,
            fp16=False
        )

        return result["text"].strip()