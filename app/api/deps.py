"""FastAPI dependency injection providers."""

from typing import Generator
from app.db.mongo import MongoDBManager, db_manager
from app.repositories.assessment_repo import AssessmentRepository
from app.services.langgraph_agent import ClinicalExtractionAgent
from app.services.transcriber import WhisperTranscriber


def get_db_manager() -> MongoDBManager:
    """Provide the global MongoDBManager singleton."""
    return db_manager


def get_assessment_repo() -> AssessmentRepository:
    """Provide an AssessmentRepository instance reusing the global db manager."""
    return AssessmentRepository(manager=db_manager)


def get_transcriber() -> WhisperTranscriber:
    """Provide a WhisperTranscriber instance configured with settings."""
    return WhisperTranscriber()


def get_extraction_agent() -> ClinicalExtractionAgent:
    """Provide a ClinicalExtractionAgent instance configured with settings."""
    return ClinicalExtractionAgent()
