from __future__ import annotations

import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv() 

class Settings(BaseModel):
    # Whisper — runs locally (openai-whisper package), no API key needed.
    # Model size options: tiny, base, small, medium, large (bigger = more
    # accurate but slower / more memory).
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")

    # Gemini — used for the LangGraph clinical-extraction agent (text -> structured data)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # MongoDB
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "clinical_assessments")

    # Confidence gate: fields scoring below this are treated as "insufficiently
    # supported" and trigger a 422 rather than being silently invented/blanked.
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
