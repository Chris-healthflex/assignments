from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Any

class FieldError(BaseModel):
    model_config = ConfigDict(extra='forbid')
    field: str
    confidence: Optional[float] = None
    reason: str = ""

class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    detail: List[FieldError] = Field(default_factory=list)