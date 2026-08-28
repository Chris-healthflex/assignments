import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stance_health")

TRANSCRIPTION_ENGINE = os.getenv("TRANSCRIPTION_ENGINE", "whisper")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

EXTRACTION_CONFIDENCE_THRESHOLD = float(os.getenv("EXTRACTION_CONFIDENCE_THRESHOLD", "0.4"))
