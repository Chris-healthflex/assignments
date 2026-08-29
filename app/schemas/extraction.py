from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FieldExtraction(BaseModel):
    """A single extracted value plus how confident we are it's real."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(description="Extracted value, or '' if not present in transcript")
    confidence: float = Field(ge=0.0, le=1.0, description="0 = not mentioned, 1 = explicit and unambiguous")
    evidence: str = Field(default="", description="Short quote/paraphrase from transcript supporting this value")


class LowConfidenceField(BaseModel):
    """Reported back to the client in the 422 response body."""

    field: str
    confidence: float
    reason: str
