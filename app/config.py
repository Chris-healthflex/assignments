from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "stance_health"
    confidence_flag_threshold: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
