from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mongo
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "stance_health"
    mongo_collection: str = "assessments"

    # Whisper
    whisper_model: str = "base"  # tiny/base/small/medium/large-v3
    whisper_device: str = "cpu"  # "cuda" if available

    # LLM used by the LangGraph extraction agent — local via Ollama, no API key,
    # no cost, and clinical transcript text never leaves the machine.
    ollama_base_url: str = "http://localhost:11434"
    # Must be a model that supports tool-calling in Ollama's API.
    # Good options: "llama3.1:8b", "qwen2.5:7b", "mistral-nemo".
    extraction_model: str = "qwen2.5:7b"

    # Confidence gating (brief: HTTP 422 below threshold)
    min_field_confidence: float = 0.55
    min_overall_confidence: float = 0.6

    upload_dir: str = "./data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
