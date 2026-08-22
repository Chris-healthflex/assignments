"""Structured Clinical Assessment Form Filler.

Pipeline: WAV -> Whisper transcript -> LangGraph extraction -> FirstAssessment
JSON -> MongoDB, exposed over a FastAPI service.
"""

__version__ = "1.0.0"
