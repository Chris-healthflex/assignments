from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    whisper_backend: str = "local"
    whisper_model: str = "base"

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "stance_assessments"

    confidence_threshold: float = 0.5


settings = Settings()
