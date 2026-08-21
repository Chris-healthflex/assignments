from typing import List, Dict, Any
from fastapi import HTTPException

class ConfidenceError(Exception):
    """Custom exception for low-confidence extraction."""
    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        super().__init__("Extraction confidence below threshold")

class AudioValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)

class AudioProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)