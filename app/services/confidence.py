from typing import List, Tuple
from app.models.internal import FieldEvidence, ConfidenceReport
from app.services.grounding import is_numeric_field, is_date_or_duration_field


def fuse_confidence(
    llm_confidence: float,
    grounded: bool,
    is_numeric_or_date: bool,
) -> float:
    """
    Combines LLM self-reported confidence with deterministic transcript grounding.
    If a numeric or date field is not grounded in the transcript, hard-caps confidence at 0.35.
    """
    clamped_llm = max(0.0, min(1.0, float(llm_confidence)))
    if is_numeric_or_date and not grounded:
        return min(clamped_llm, 0.35)
    return clamped_llm


def evaluate_confidence(
    field_evidence_list: List[FieldEvidence],
    threshold: float = 0.70,
) -> ConfidenceReport:
    """
    Evaluates field-level evidence and produces the overall ConfidenceReport.
    Overall confidence is the minimum fused score across populated fields (not an average),
    guaranteeing that a single hallucinated number fails or flags the report.
    """
    if not field_evidence_list:
        return ConfidenceReport(
            overall=1.0,
            threshold=threshold,
            flags=[],
            passed=True,
        )

    flags: List[FieldEvidence] = []
    fused_scores: List[float] = []

    for ev in field_evidence_list:
        fused_scores.append(ev.confidence)
        is_num_or_date = is_numeric_field(ev.field) or is_date_or_duration_field(ev.field)
        if ev.confidence < threshold or (is_num_or_date and not ev.grounded):
            flags.append(ev)

    overall = min(fused_scores) if fused_scores else 1.0
    passed = overall >= threshold

    return ConfidenceReport(
        overall=round(overall, 3),
        threshold=threshold,
        flags=flags,
        passed=passed,
    )
