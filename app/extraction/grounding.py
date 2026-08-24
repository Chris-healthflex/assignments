"""Grounding verification — the anti-hallucination gate.

For every populated field we check that its content is actually supported by the
transcript. Numeric values (measurements, angles) must appear verbatim; free-text
fields must overlap the transcript strongly enough to not be invented. Anything that
fails is BLANKED (set to "") and recorded as an `ungrounded` flag, so a hallucinated
clinical value never reaches the frontend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from app.schemas.assessment import FirstAssessment


@dataclass
class GroundingReport:
    ungrounded: List[dict] = field(default_factory=list)  # {path, reason, detail}
    # coverage[section] in [0,1]: how completely the section captured what the
    # transcript actually states. Drives the confidence score so it can't report
    # 1.0 on a table that is missing or duplicating measurements.
    coverage: dict = field(default_factory=dict)

    def flag(self, path: str, detail: str = "", reason: str = "ungrounded") -> None:
        self.ungrounded.append({"path": path, "reason": reason, "detail": detail})


_WORD = re.compile(r"[a-z0-9.]+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _numbers(s: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", s)


def _text_supported(value: str, transcript_norm: str, threshold: float = 0.6) -> bool:
    """A text value is grounded if enough of its tokens appear in the transcript."""
    v = _norm(value)
    if not v:
        return True  # empty is trivially fine
    tokens = _WORD.findall(v)
    if not tokens:
        return True
    hits = sum(1 for tok in tokens if tok in transcript_norm)
    return (hits / len(tokens)) >= threshold


def _value_supported(value: str, transcript: str, transcript_norm: str) -> bool:
    """Numeric values must appear exactly; text values must overlap."""
    if not value.strip():
        return True
    nums = _numbers(value)
    if nums:
        return all(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", transcript) for n in nums)
    return _text_supported(value, transcript_norm)


# --------------------------------------------------------------------------- #
# Objective-measurement coverage (catches missing / duplicate / mis-counted rows)
# --------------------------------------------------------------------------- #
# A measurement clause in this domain reads "<name> of <N> degrees ...".
_MEASURE_CLAUSE = re.compile(r"\bof\s+(\d+(?:\.\d+)?)\s*degrees?", re.IGNORECASE)
_ANY_DEGREE = re.compile(r"(\d+(?:\.\d+)?)\s*degrees?", re.IGNORECASE)


def _objective_coverage(assessment: FirstAssessment, transcript: str, report: GroundingReport) -> float:
    """Compare stated measurements against extracted rows.

    recall  = transcript measurement values that made it into some row
    penalty = agreement between #stated clauses and #extracted rows
              (extra rows => splits/duplicates; too few => dropped measurements)
    score   = recall * penalty  -> only a complete, correctly-counted table hits 1.0.
    Emits `missing_measurement`, `duplicate_row`, and `count_mismatch` flags.
    """
    tests = assessment.objectiveAssessment.tests
    expected = len(_MEASURE_CLAUSE.findall(transcript))
    stated_values = {v for v in _ANY_DEGREE.findall(transcript)}

    if not stated_values:
        return 1.0 if not tests else 1.0  # no measurements stated; nothing to cover

    # values present anywhere in the extracted rows
    output_values: set[str] = set()
    for t in tests:
        for side in (t.left, t.right, t.value):
            for n in _numbers(side):
                output_values.add(n)

    missing = sorted(stated_values - output_values, key=float)
    for v in missing:
        report.flag(
            "objectiveAssessment.tests",
            detail=f"transcript states a measurement of {v} degrees not present in any row",
            reason="missing_measurement",
        )

    # duplicate / split detection: same testName appearing more than once
    seen: dict[str, int] = {}
    for i, t in enumerate(tests):
        key = _norm(t.testName)
        if key and key in seen:
            report.flag(
                f"objectiveAssessment.tests[{i}]",
                detail=f"duplicate of tests[{seen[key]}] ('{t.testName}')",
                reason="duplicate_row",
            )
        elif key:
            seen[key] = i

    # count agreement between stated clauses and extracted rows
    penalty = 1.0
    if expected and tests:
        penalty = min(expected, len(tests)) / max(expected, len(tests))
        if len(tests) != expected:
            report.flag(
                "objectiveAssessment.tests",
                detail=f"transcript states {expected} measurements but {len(tests)} rows were extracted",
                reason="count_mismatch",
            )

    recall = len(stated_values & output_values) / len(stated_values)
    return round(recall * penalty, 2)


def verify(assessment: FirstAssessment, transcript: str) -> GroundingReport:
    """Mutates `assessment` in place, blanking ungrounded values. Returns the report."""
    report = GroundingReport()
    tnorm = _norm(transcript)

    cd = assessment.clinicalDetails
    for name in ("clinicalHistory", "chiefComplaint", "duration"):
        val = getattr(cd, name)
        if val and not _text_supported(val, tnorm):
            setattr(cd, name, "")
            report.flag(f"clinicalDetails.{name}", "value not found in transcript")

    for i, s in enumerate(assessment.subjectiveAssessments):
        if s.testName and not _text_supported(s.testName, tnorm):
            report.flag(f"subjectiveAssessments[{i}].testName", "not found in transcript")
            s.testName = ""

    for i, test in enumerate(assessment.objectiveAssessment.tests):
        for side in ("value", "left", "right"):
            v = getattr(test, side)
            if v and not _value_supported(v, transcript, tnorm):
                setattr(test, side, "")
                report.flag(
                    f"objectiveAssessment.tests[{i}].{side}",
                    f"'{v}' not present in transcript",
                )

    # completeness of the objective table vs. what the transcript actually states
    report.coverage["objectiveAssessment"] = _objective_coverage(assessment, transcript, report)

    for i, g in enumerate(assessment.objectiveGoals):
        if g.goalName and not _text_supported(g.goalName, tnorm):
            report.flag(f"objectiveGoals[{i}].goalName", "not found in transcript")
            g.goalName = ""
        if g.value and not _value_supported(g.value, transcript, tnorm):
            report.flag(f"objectiveGoals[{i}].value", "not found in transcript")
            g.value = ""

    for i, r in enumerate(assessment.recommendation):
        if r.sessionFrequency and not _text_supported(r.sessionFrequency, tnorm):
            report.flag(f"recommendation[{i}].sessionFrequency", "not found in transcript")
            r.sessionFrequency = ""

    return report
