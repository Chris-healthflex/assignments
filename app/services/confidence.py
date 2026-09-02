"""Grounding verification and anti-hallucination validation module.

Validates that extracted clinical entities and measurements are strictly grounded
in the source transcript, flagging ungrounded or contradictory fields internally.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.schemas.assessment import FirstAssessment, ObjectiveTest


@dataclass
class GroundingCheckResult:
    """Internal validation and grounding result."""

    is_grounded: bool = True
    evidence: Dict[str, str] = field(default_factory=dict)
    uncertain_fields: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)


def _contains_number(text: str, number_str: str) -> bool:
    """Check if a standalone numeric value (e.g. '124', '4.5', '-5', '20', '0') appears in the transcript text."""
    if not number_str or not number_str.strip():
        return True

    clean_num = number_str.strip().lower()
    # Match as a standalone number with negative sign support, ensuring no adjacent digits/decimals
    pattern = r"(?<![0-9.])" + re.escape(clean_num) + r"(?![0-9.])"
    if re.search(pattern, text, re.IGNORECASE):
        return True

    return False


def validate_grounding(
    transcript: str,
    assessment: FirstAssessment,
) -> GroundingCheckResult:
    """Verify that all extracted clinical values, measurements, and recommendations are grounded in the transcript.

    Args:
        transcript: Source clinical transcript text.
        assessment: Extracted FirstAssessment instance.

    Returns:
        GroundingCheckResult with evidence mapping, uncertain fields, and validation errors.
    """
    result = GroundingCheckResult()
    normalized_transcript = transcript.lower()

    # 1. Validate Objective Measurements Grounding
    for idx, test in enumerate(assessment.objectiveAssessment.tests):
        field_prefix = f"objectiveAssessment.tests[{idx}] ({test.testName})"

        # Check numeric measurement values (value, left, right)
        for val_attr, val_label in [("value", "value"), ("left", "left"), ("right", "right")]:
            val = getattr(test, val_attr, "")
            if val and val.strip():
                if not _contains_number(normalized_transcript, val):
                    result.is_grounded = False
                    result.uncertain_fields.append({
                        "field": f"{field_prefix}.{val_label}",
                        "value": val,
                        "reason": f"Measurement value '{val}' is not found or supported in the transcript text.",
                    })
                else:
                    result.evidence[f"{field_prefix}.{val_label}"] = f"Matched '{val}' in transcript"

        # Check test name relevance
        if test.testName and test.testName.strip():
            test_keywords = [w for w in re.split(r"\W+", test.testName.lower()) if len(w) > 3]
            if test_keywords and not any(kw in normalized_transcript for kw in test_keywords):
                result.uncertain_fields.append({
                    "field": f"{field_prefix}.testName",
                    "value": test.testName,
                    "reason": f"Test name keywords {test_keywords} not found in transcript.",
                })

    # 2. Validate Clinical Details Grounding
    if assessment.clinicalDetails.chiefComplaint:
        cc = assessment.clinicalDetails.chiefComplaint
        cc_words = [w for w in re.split(r"\W+", cc.lower()) if len(w) > 3]
        if cc_words and any(w in normalized_transcript for w in cc_words):
            result.evidence["clinicalDetails.chiefComplaint"] = cc
        elif cc_words:
            result.uncertain_fields.append({
                "field": "clinicalDetails.chiefComplaint",
                "value": cc,
                "reason": "Chief complaint does not align with transcript keywords.",
            })

    # 3. Validate Subjective Assessments Grounding
    for idx, sub in enumerate(assessment.subjectiveAssessments):
        field_name = f"subjectiveAssessments[{idx}] ({sub.testName})"
        if sub.conclusion:
            matched = any(c.lower() in normalized_transcript for c in sub.conclusion)
            if matched:
                result.evidence[field_name] = "; ".join(sub.conclusion)

    # 4. Validate Goals and Dates Grounding
    for idx, goal in enumerate(assessment.subjectiveGoals):
        if goal.goalDetails and goal.goalDetails.strip():
            goal_words = [w for w in re.split(r"\W+", goal.goalDetails.lower()) if len(w) > 3]
            if goal_words and not any(w in normalized_transcript for w in goal_words):
                result.uncertain_fields.append({
                    "field": f"subjectiveGoals[{idx}].goalDetails",
                    "value": goal.goalDetails,
                    "reason": "Subjective goal details do not match any transcript statements.",
                })
        if goal.targetDate and goal.targetDate.strip():
            if not _contains_number(normalized_transcript, goal.targetDate):
                result.uncertain_fields.append({
                    "field": f"subjectiveGoals[{idx}].targetDate",
                    "value": goal.targetDate,
                    "reason": f"Target date '{goal.targetDate}' was not explicitly mentioned in transcript.",
                })

    for idx, o_goal in enumerate(assessment.objectiveGoals):
        if o_goal.value and o_goal.value.strip():
            if not _contains_number(normalized_transcript, o_goal.value):
                result.is_grounded = False
                result.uncertain_fields.append({
                    "field": f"objectiveGoals[{idx}].value",
                    "value": o_goal.value,
                    "reason": f"Objective goal target value '{o_goal.value}' is not supported or spoken in the transcript.",
                })
        if o_goal.targetDate and o_goal.targetDate.strip():
            if not _contains_number(normalized_transcript, o_goal.targetDate):
                result.uncertain_fields.append({
                    "field": f"objectiveGoals[{idx}].targetDate",
                    "value": o_goal.targetDate,
                    "reason": f"Target date '{o_goal.targetDate}' was not explicitly mentioned in transcript.",
                })

    # 5. Validate Patient Advice Grounding
    if assessment.patientAdvice and assessment.patientAdvice.adviceDetails:
        adv = assessment.patientAdvice.adviceDetails
        adv_words = [w for w in re.split(r"\W+", adv.lower()) if len(w) > 3]
        if adv_words and not any(w in normalized_transcript for w in adv_words):
            result.uncertain_fields.append({
                "field": "patientAdvice.adviceDetails",
                "value": adv,
                "reason": "Patient advice details are not supported by transcript.",
            })

    return result
