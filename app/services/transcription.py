from pathlib import Path

from faster_whisper import WhisperModel


# Use the CPU-friendly configuration so the assignment can run
# without requiring a GPU.
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)


def transcribe_audio(audio_path: str | Path) -> str:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
    )

    transcript = " ".join(segment.text.strip() for segment in segments)

    return transcript.strip()