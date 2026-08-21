from typing import TypedDict, List, Dict, Any
from app.schemas.extraction import ExtractionResult

class AgentState(TypedDict):
    transcript: str
    result: ExtractionResult
    retry_count: int               # global retry counter
    section_errors: List[Dict[str, Any]]
    retry_needed: bool