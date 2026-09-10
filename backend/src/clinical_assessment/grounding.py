"""Deterministic verification of an extraction against its transcript.

This is the anti-hallucination core. Nothing here asks the model how sure it
is: a field counts as grounded only if its evidence can be found in the
transcript and its digits can be found in its evidence. Pure functions only —
no I/O, no model call, no clock.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from .extraction_models import ExtractedValue, RawExtraction

# Punctuation becomes a space rather than being deleted, so "range-of-motion"
# and "range of motion" normalise alike instead of to "rangeofmotion".
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"\d+")


class GroundingFailure(Enum):
    """Why a field failed verification."""

    EVIDENCE_NOT_FOUND = "evidence does not appear verbatim in the transcript"
    MISSING_EVIDENCE = "a value was given with no supporting evidence"
    DIGITS_NOT_IN_EVIDENCE = "value contains digits that its evidence does not"


@dataclass(frozen=True)
class FieldVerdict:
    """The verification outcome for a single extracted field.

    Attributes:
        field_path: Dotted path to the field, list levels indexed, e.g.
            ``objectiveAssessment.tests[0].value``.
        value: The value the model proposed.
        evidence: The transcript span the model quoted for it.
        failure: Why the field failed, or ``None`` when it is grounded.
    """

    field_path: str
    value: str
    evidence: str
    failure: GroundingFailure | None

    @property
    def grounded(self) -> bool:
        """Whether the field survived verification."""
        return self.failure is None


@dataclass(frozen=True)
class GroundingReport:
    """Per-field verdicts and the confidence ratio derived from them."""

    verdicts: tuple[FieldVerdict, ...]
    confidence: float

    @property
    def ungrounded(self) -> tuple[FieldVerdict, ...]:
        """The verdicts that failed, in extraction order."""
        return tuple(verdict for verdict in self.verdicts if not verdict.grounded)

    def rejected_quotes(self) -> dict[str, str]:
        """Map each failed field path to the quote that was rejected."""
        return {verdict.field_path: verdict.evidence for verdict in self.ungrounded}


def normalise(text: str) -> str:
    """Reduce text to a form where quoting differences do not matter.

    Lowercases, replaces punctuation with spaces, and collapses whitespace, so
    that a difference in casing, a trailing comma, or a line break between the
    quote and the transcript does not read as a fabrication.

    Args:
        text: Raw text.

    Returns:
        The normalised form.
    """
    without_punctuation = _PUNCTUATION.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def is_grounded(value: str, evidence: str, transcript: str) -> bool:
    """Whether a value is supported by its evidence and the transcript.

    Args:
        value: The proposed clinical datum.
        evidence: The transcript span quoted in support of it.
        transcript: The full transcript.

    Returns:
        ``True`` when the field is grounded.
    """
    return _failure_for(value, evidence, transcript) is None


def score(extraction: RawExtraction, transcript: str) -> GroundingReport:
    """Verify every field of an extraction against the transcript.

    Args:
        extraction: The evidence-carrying extraction to verify.
        transcript: The full transcript it was drawn from.

    Returns:
        A report holding one verdict per field and the grounded ratio.
    """
    normalised_transcript = normalise(transcript)
    verdicts = tuple(
        FieldVerdict(
            field_path=field_path,
            value=extracted.value,
            evidence=extracted.evidence,
            failure=_failure_for(
                extracted.value, extracted.evidence, normalised_transcript
            ),
        )
        for field_path, extracted in _walk_leaves(extraction)
    )
    return GroundingReport(verdicts=verdicts, confidence=_confidence(verdicts))


def _failure_for(value: str, evidence: str, transcript: str) -> GroundingFailure | None:
    """Return the reason a field fails verification, or None if it holds.

    ``transcript`` may be passed raw or already normalised; normalising twice is
    a no-op, and the caller in :func:`score` normalises once for all fields.
    """
    # A field asserting nothing cannot be a fabrication. Per the confidence
    # rule, "not stated in the transcript" is a correct answer, not a failure.
    if not value:
        return None
    if not evidence:
        return GroundingFailure.MISSING_EVIDENCE
    if normalise(evidence) not in normalise(transcript):
        return GroundingFailure.EVIDENCE_NOT_FOUND
    if not _digits_supported(value, evidence):
        return GroundingFailure.DIGITS_NOT_IN_EVIDENCE
    return None


def _digits_supported(value: str, evidence: str) -> bool:
    """Whether every digit run in the value also appears in its evidence.

    Whole runs are compared rather than substrings, so a value of "45" is not
    accepted on evidence reading "145 degrees". This is the specific defence
    against the worst failure mode: a real quote paired with an invented number.
    """
    evidence_runs = set(_DIGIT_RUN.findall(evidence))
    return all(run in evidence_runs for run in _DIGIT_RUN.findall(value))


def _confidence(verdicts: tuple[FieldVerdict, ...]) -> float:
    """Grounded fields divided by total fields."""
    # The current schema always yields leaves, so this guard is unreachable in
    # practice; it exists so the ratio is total rather than a ZeroDivisionError.
    if not verdicts:
        return 0.0
    grounded_count = sum(1 for verdict in verdicts if verdict.grounded)
    return grounded_count / len(verdicts)


def _walk_leaves(
    node: object, prefix: str = ""
) -> Iterator[tuple[str, ExtractedValue]]:
    """Yield every ``ExtractedValue`` in the tree with its dotted path."""
    if isinstance(node, ExtractedValue):
        yield prefix, node
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_leaves(item, f"{prefix}[{index}]")
        return
    for field_name in type(node).model_fields:
        child = getattr(node, field_name)
        yield from _walk_leaves(
            child, f"{prefix}.{field_name}" if prefix else field_name
        )
