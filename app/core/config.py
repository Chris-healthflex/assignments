import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    mongodb_uri: str = Field(default="", validation_alias="MONGODB_URI")
    mongodb_database: str = Field(default="clinical_assessment", validation_alias="MONGODB_DATABASE")

    llm_provider: str = Field(default="groq", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="LLM_MODEL")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", validation_alias="GROQ_BASE_URL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")

    whisper_provider: str = Field(default="groq", validation_alias="WHISPER_PROVIDER")
    whisper_model: str = Field(default="whisper-large-v3-turbo", validation_alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", validation_alias="WHISPER_COMPUTE_TYPE")

    confidence_threshold: float = Field(default=0.70, validation_alias="CONFIDENCE_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
