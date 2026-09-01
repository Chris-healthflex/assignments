import whisper


def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")

    print("Transcribing audio...")
    result = model.transcribe(audio_path)

    return result["text"]