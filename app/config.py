"""Single place where environment configuration and tuning constants are read.

Non-secret values are read once at import time. The Groq API key is read on each
call instead (`get_groq_api_key`), so a missing key surfaces as a handled error
at request time rather than as an import failure at startup.
"""

import os
from dotenv import load_dotenv

# Automatically load environment variables from .env file if present
load_dotenv()


# --- Model selection (Groq) ---
WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")
LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b")

# The upload is the slow part of transcription. The provided 8.88 MB file needs
# ~150s on a modest uplink, which exceeds the Groq SDK default and fails with
# "Request timed out".
TRANSCRIPTION_TIMEOUT_SECONDS = 300.0

# Confidence below which POST /assessments/parse returns HTTP 422.
CONFIDENCE_THRESHOLD = 0.70

# --- MongoDB ---
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "stance_health")
COLLECTION_NAME = "assessments"

# --- Upload validation ---
ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def get_groq_api_key() -> str:
    """Return the Groq API key, stripped. Empty string if it is not set.

    Callers decide how to handle absence: transcription raises, the extraction
    agent reports it through `field_errors` so the pipeline fails closed.
    """
    return (os.environ.get("GROQ_API_KEY") or "").strip()
