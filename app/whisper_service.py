import whisper


# ---------------------------------------------------------
# LOAD WHISPER MODEL
# ---------------------------------------------------------

model = whisper.load_model("base")


# ---------------------------------------------------------
# TRANSCRIBE AUDIO
# ---------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using Whisper.
    """

    result = model.transcribe(
        audio_path,
        fp16=False
    )

    transcription = result.get(
        "text",
        ""
    ).strip()

    return transcription