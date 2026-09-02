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
    # Domain vocabulary primed into Whisper's decoder. Generic musculoskeletal /
    # physiotherapy terminology -- NOT content specific to any one recording.
    whisper_initial_prompt: str = (
        "Physiotherapy assessment. Orthopaedic and musculoskeletal terminology: "
        "patellar mobility, tibial condyle fracture, tibial plateau, avulsion ACL tear, "
        "open reduction and internal fixation, non-weight bearing, range of motion, "
        "knee flexion, knee extension, hip internal rotation, hip external rotation, "
        "ankle dorsiflexion, plantarflexion, quadriceps, hamstrings, posterior chain, "
        "goniometer, bilaterally, overpressure, irritability, single leg stability. "
        "Ranges of motion are read in degrees and may be negative, "
        "for example negative 5 degrees of extension."
    )

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "stance_assessments"

    confidence_threshold: float = 0.5


settings = Settings()
