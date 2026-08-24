"""End-to-end audio -> structured assessment pipeline.

Stages (mirrors the architecture diagram):
    WAV -> validate -> Whisper -> transcript
        -> LangGraph section extraction
        -> normalizer (-> FirstAssessment)
        -> grounding verification (blank hallucinated values)
        -> confidence scoring + flagging
        -> PipelineResult envelope
"""
from __future__ import annotations

import time
from typing import Optional

from app.api.schemas import (
    ConfidenceReport,
    FlaggedField,
    PipelineResult,
    TranscriptMeta,
)
from app.config import settings
from app.extraction import confidence as confidence_mod
from app.extraction import grounding as grounding_mod
from app.extraction.graph import run_extraction
from app.extraction.normalizer import build_assessment
from app.transcription.models import Transcript
from app.transcription.whisper_service import transcribe


def run_from_transcript(
    transcript_text: str,
    transcript_meta: Optional[Transcript] = None,
) -> PipelineResult:
    """Run extraction onward from an already-produced transcript."""
    timings: dict = {}

    t0 = time.perf_counter()
    state = run_extraction(transcript_text)
    timings.update(state.get("timings", {}))

    t1 = time.perf_counter()
    assessment = build_assessment(state)
    timings["assemble"] = round(time.perf_counter() - t1, 2)

    t2 = time.perf_counter()
    grounding = grounding_mod.verify(assessment, transcript_text)
    timings["grounding"] = round(time.perf_counter() - t2, 2)

    t3 = time.perf_counter()
    conf = confidence_mod.score(assessment, grounding, settings.confidence_threshold)
    timings["confidence"] = round(time.perf_counter() - t3, 2)

    timings["total"] = round(time.perf_counter() - t0, 2)

    if transcript_meta is not None:
        meta = TranscriptMeta(**transcript_meta.as_meta())
    else:
        meta = TranscriptMeta(text=transcript_text, language="en", backend="provided")

    return PipelineResult(
        assessment=assessment,
        transcript=meta,
        confidence=ConfidenceReport(
            overall=conf.overall,
            threshold=conf.threshold,
            meetsThreshold=conf.meets_threshold,
            sectionScores=conf.section_scores,
            rejectedCount=conf.rejected_count,
        ),
        flaggedFields=[FlaggedField(**f) for f in conf.flagged],
        timings=timings,
    )


def run_from_wav(wav_path: str) -> PipelineResult:
    """Full pipeline from a WAV file on disk."""
    t0 = time.perf_counter()
    tr = transcribe(wav_path)
    transcribe_secs = round(time.perf_counter() - t0, 2)

    result = run_from_transcript(tr.text, transcript_meta=tr)
    result.timings["transcribe"] = transcribe_secs
    return result
