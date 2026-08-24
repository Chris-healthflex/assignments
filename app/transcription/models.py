"""Plain data structures for transcription output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    language: str = "en"
    duration_seconds: float = 0.0
    segments: List[Segment] = field(default_factory=list)
    model: str = ""
    backend: str = ""

    def as_meta(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "durationSeconds": round(self.duration_seconds, 2),
            "segments": len(self.segments),
            "model": self.model,
            "backend": self.backend,
        }
