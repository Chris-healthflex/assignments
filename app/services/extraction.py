import os
import re
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ConfigDict, Field

from app.models.assessment import FirstAssessment


load_dotenv()


CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# Confidence models
# ============================================================


class ConfidenceIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_path: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reason: str = ""


class ExtractionConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    issues: list[ConfidenceIssue] = Field(
        default_factory=list
    )


# ============================================================
# Gemini
# ============================================================


def create_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY was not found. Check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=api_key,
    )


# ============================================================
# Normalization
# ============================================================


NULL_LIKE_VALUES = {
    "",
    "null",
    "none",
    "n/a",
    "na",
}


QUALITATIVE_VALUES = {
    "restricted",
    "painful",
    "pain",
    "good",
    "poor",
    "present",
    "absent",
    "normal",
    "abnormal",
    "full",
    "limited",
    "healed",
}


AMBIGUOUS_MARKERS = {
    "negic",
    "unclear",
    "inaudible",
    "unintelligible",
}


def clean_string(value: Any) -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value.lower() in NULL_LIKE_VALUES:
        return ""

    return value


def normalize_test_name(value: str) -> str:
    value = clean_string(value).lower()

    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def append_comment(
    existing: str,
    new_comment: str,
) -> str:
    existing = clean_string(existing)
    new_comment = clean_string(new_comment)

    if not existing:
        return new_comment

    if not new_comment:
        return existing

    if new_comment.lower() in existing.lower():
        return existing

    return f"{existing}; {new_comment}"


# ============================================================
# Normalization pipeline
# ============================================================


def normalize_subjective_assessments(
    assessment: FirstAssessment,
) -> FirstAssessment:

    result = []
    seen = set()

    for item in assessment.subjectiveAssessments:
        item = item.model_copy(deep=True)

        item.testName = clean_string(item.testName)
        item.conclusion = clean_string(item.conclusion)

        key = (
            normalize_test_name(item.testName),
            item.conclusion.lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    assessment.subjectiveAssessments = result

    return assessment


def normalize_objective_assessment(
    assessment: FirstAssessment,
) -> FirstAssessment:

    items = []

    for item in assessment.objectiveAssessment.tests:
        item = item.model_copy(deep=True)

        item.testName = clean_string(item.testName)
        item.unitName = clean_string(item.unitName)
        item.value = clean_string(item.value)
        item.left = clean_string(item.left)
        item.right = clean_string(item.right)
        item.comments = clean_string(item.comments)

        # Qualitative values belong in comments.
        for side in ("left", "right"):
            side_value = getattr(item, side)

            if side_value.lower() in QUALITATIVE_VALUES:
                item.comments = append_comment(
                    item.comments,
                    side_value,
                )
                setattr(item, side, "")

        if item.value.lower() in QUALITATIVE_VALUES:
            item.comments = append_comment(
                item.comments,
                item.value,
            )
            item.value = ""

        # Bilateral value.
        if (
            item.value
            and not item.left
            and not item.right
            and "bilateral" in item.comments.lower()
        ):
            value = item.value

            item.left = value
            item.right = value
            item.value = ""

            item.comments = re.sub(
                r"\bbilateral(?:ly)?\b",
                "",
                item.comments,
                flags=re.IGNORECASE,
            ).strip(" ;,.")

        items.append(item)

    merged = {}

    for item in items:
        key = normalize_test_name(item.testName)

        if not key:
            key = f"unnamed-{len(merged)}"

        if key not in merged:
            merged[key] = item
            continue

        existing = merged[key]

        if not existing.unitName and item.unitName:
            existing.unitName = item.unitName

        if not existing.value and item.value:
            existing.value = item.value

        if not existing.left and item.left:
            existing.left = item.left

        if not existing.right and item.right:
            existing.right = item.right

        existing.comments = append_comment(
            existing.comments,
            item.comments,
        )

    assessment.objectiveAssessment.tests = list(
        merged.values()
    )

    return assessment


def normalize_all_strings(
    assessment: FirstAssessment,
) -> FirstAssessment:

    assessment.clinicalDetails.clinicalHistory = clean_string(
        assessment.clinicalDetails.clinicalHistory
    )

    assessment.clinicalDetails.chiefComplaint = clean_string(
        assessment.clinicalDetails.chiefComplaint
    )

    assessment.clinicalDetails.duration = clean_string(
        assessment.clinicalDetails.duration
    )

    for item in assessment.subjectiveAssessments:
        item.testName = clean_string(item.testName)
        item.conclusion = clean_string(item.conclusion)

    for item in assessment.objectiveAssessment.tests:
        item.testName = clean_string(item.testName)
        item.unitName = clean_string(item.unitName)
        item.value = clean_string(item.value)
        item.left = clean_string(item.left)
        item.right = clean_string(item.right)
        item.comments = clean_string(item.comments)

    for item in assessment.subjectiveGoals:
        item.goalDetails = clean_string(item.goalDetails)
        item.targetDate = clean_string(item.targetDate)

    for item in assessment.objectiveGoals:
        item.goalName = clean_string(item.goalName)
        item.goalCategory = clean_string(item.goalCategory)
        item.unitName = clean_string(item.unitName)
        item.value = clean_string(item.value)
        item.targetDate = clean_string(item.targetDate)

    for item in assessment.recommendation:
        item.sessionType = clean_string(item.sessionType)
        item.sessionFrequency = clean_string(item.sessionFrequency)

    assessment.patientAdvice.adviceDetails = clean_string(
        assessment.patientAdvice.adviceDetails
    )

    return assessment


def normalize_assessment(
    assessment: FirstAssessment,
) -> FirstAssessment:

    assessment = normalize_all_strings(assessment)
    assessment = normalize_subjective_assessments(assessment)
    assessment = normalize_objective_assessment(assessment)
    assessment = normalize_all_strings(assessment)

    return FirstAssessment.model_validate(
        assessment
    )


# ============================================================
# Extraction
# ============================================================


def extract_assessment(
    transcript: str,
) -> FirstAssessment:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = create_llm()

    structured_llm = llm.with_structured_output(
        FirstAssessment
    )

    prompt = f"""
You are a strict clinical information extraction system.

Convert the transcript into the FirstAssessment schema.

Use ONLY information supported by the transcript.

Never invent clinical facts.

Never invent:
- diagnoses
- measurements
- scores
- dates
- treatment recommendations
- goals
- numerical targets
- patient advice

Do not use outside medical knowledge to fill gaps.

If information is absent, use "" for strings.

If an array has no supported entries, use [].

Never use null for a string field.

Do not add fields.

Do not rename fields.

============================================================
CLINICAL DETAILS
============================================================

Extract only explicit clinical history, chief complaint and duration.

============================================================
SUBJECTIVE ASSESSMENTS
============================================================

Include patient-reported symptoms, complaints and functional
limitations.

Group closely related information.

Do not put examination findings here.

============================================================
OBJECTIVE ASSESSMENT
============================================================

The exact structure is:

objectiveAssessment.tests[]

Create one test record per distinct examination test/finding.

Fields:

testName
unitName
value
left
right
comments

Qualitative findings such as "restricted", "painful", "good",
"full", "limited", "swelling", "healed" belong in comments
when explicitly stated.

Numerical measurements belong in value/left/right.

============================================================
LEFT / RIGHT
============================================================

Only put a value in left when the transcript explicitly states
a left-side value.

Only put a value in right when the transcript explicitly states
a right-side value.

Do not infer side values.

============================================================
BILATERAL MEASUREMENTS
============================================================

If the transcript explicitly says:

"45 degrees bilaterally"

then create/use the appropriate test and return:

unitName = "degrees"
value = ""
left = "45"
right = "45"
comments = ""

Do not leave the 45-degree value only in comments.

============================================================
AMBIGUOUS WHISPER TRANSCRIPTION
============================================================

If a source expression is unclear, preserve it exactly enough
to show the ambiguity.

Example:

"negic 5"

Do NOT convert that to:
- 5
- -5
- any guessed clinical value

============================================================
GOALS
============================================================

Only include explicit goals.

Never invent numerical targets or target dates.

Only populate goalCategory when supported.

============================================================
RECOMMENDATION
============================================================

Only include explicitly stated recommendations.

============================================================
PATIENT ADVICE
============================================================

Only include advice explicitly directed to the patient.

If absent:

adviceDetails = ""

============================================================
FINAL CHECK
============================================================

Before returning:

- no hallucinations
- no null strings
- no unsupported numbers
- no unsupported dates
- no unsupported goals
- no unsupported recommendations
- no renamed fields
- no extra fields

SOURCE TRANSCRIPT:

{transcript}
"""

    result = structured_llm.invoke(prompt)

    assessment = FirstAssessment.model_validate(result)

    return normalize_assessment(assessment)


# ============================================================
# Assessment fields
# ============================================================


def collect_assessment_strings(
    assessment: FirstAssessment,
) -> Iterator[tuple[str, str]]:

    yield (
        "clinicalDetails.clinicalHistory",
        assessment.clinicalDetails.clinicalHistory,
    )

    yield (
        "clinicalDetails.chiefComplaint",
        assessment.clinicalDetails.chiefComplaint,
    )

    yield (
        "clinicalDetails.duration",
        assessment.clinicalDetails.duration,
    )

    for index, item in enumerate(
        assessment.subjectiveAssessments
    ):
        yield (
            f"subjectiveAssessments[{index}].testName",
            item.testName,
        )

        yield (
            f"subjectiveAssessments[{index}].conclusion",
            item.conclusion,
        )

    for index, item in enumerate(
        assessment.objectiveAssessment.tests
    ):
        yield (
            f"objectiveAssessment.tests[{index}].testName",
            item.testName,
        )

        yield (
            f"objectiveAssessment.tests[{index}].unitName",
            item.unitName,
        )

        yield (
            f"objectiveAssessment.tests[{index}].value",
            item.value,
        )

        yield (
            f"objectiveAssessment.tests[{index}].left",
            item.left,
        )

        yield (
            f"objectiveAssessment.tests[{index}].right",
            item.right,
        )

        yield (
            f"objectiveAssessment.tests[{index}].comments",
            item.comments,
        )

    for index, item in enumerate(
        assessment.subjectiveGoals
    ):
        yield (
            f"subjectiveGoals[{index}].goalDetails",
            item.goalDetails,
        )

        yield (
            f"subjectiveGoals[{index}].targetDate",
            item.targetDate,
        )

    for index, item in enumerate(
        assessment.objectiveGoals
    ):
        yield (
            f"objectiveGoals[{index}].goalName",
            item.goalName,
        )

        yield (
            f"objectiveGoals[{index}].goalCategory",
            item.goalCategory,
        )

        yield (
            f"objectiveGoals[{index}].unitName",
            item.unitName,
        )

        yield (
            f"objectiveGoals[{index}].value",
            item.value,
        )

        yield (
            f"objectiveGoals[{index}].targetDate",
            item.targetDate,
        )

    for index, item in enumerate(
        assessment.recommendation
    ):
        yield (
            f"recommendation[{index}].sessionType",
            item.sessionType,
        )

        yield (
            f"recommendation[{index}].sessionFrequency",
            item.sessionFrequency,
        )

    yield (
        "patientAdvice.adviceDetails",
        assessment.patientAdvice.adviceDetails,
    )


# ============================================================
# Deterministic confidence
# ============================================================


def deterministic_confidence_issues(
    transcript: str,
    assessment: FirstAssessment,
) -> list[ConfidenceIssue]:

    issues: list[ConfidenceIssue] = []

    transcript_lower = transcript.lower()

    # Null-like values.
    for field_path, value in collect_assessment_strings(
        assessment
    ):
        if value.strip().lower() in {
            "null",
            "none",
            "n/a",
            "na",
        }:
            issues.append(
                ConfidenceIssue(
                    field_path=field_path,
                    confidence=0.0,
                    reason="Null-like value found.",
                )
            )

    # Ambiguous transcription markers.
    for field_path, value in collect_assessment_strings(
        assessment
    ):
        value_lower = value.lower()

        for marker in AMBIGUOUS_MARKERS:
            if marker in value_lower:
                issues.append(
                    ConfidenceIssue(
                        field_path=field_path,
                        confidence=0.55,
                        reason=(
                            f"Ambiguous transcription marker "
                            f"'{marker}' detected."
                        ),
                    )
                )
                break

    # Unsupported numeric values.
    numeric_pattern = re.compile(
        r"^-?\d+(?:\.\d+)?$"
    )

    for index, item in enumerate(
        assessment.objectiveAssessment.tests
    ):

        for field_name, value in (
            ("value", item.value),
            ("left", item.left),
            ("right", item.right),
        ):

            if not value:
                continue

            if not numeric_pattern.fullmatch(value):
                continue

            if value.lower() not in transcript_lower:
                issues.append(
                    ConfidenceIssue(
                        field_path=(
                            "objectiveAssessment."
                            f"tests[{index}].{field_name}"
                        ),
                        confidence=0.40,
                        reason=(
                            f"Numeric value '{value}' is not "
                            "present in the transcript."
                        ),
                    )
                )

    # Bilateral measurement check.
    bilateral_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s+degrees\s+bilaterally",
        re.IGNORECASE,
    )

    bilateral_values = bilateral_pattern.findall(
        transcript
    )

    for value in bilateral_values:

        matched = any(
            item.unitName.lower() == "degrees"
            and item.left == value
            and item.right == value
            for item in assessment.objectiveAssessment.tests
        )

        if not matched:
            issues.append(
                ConfidenceIssue(
                    field_path="objectiveAssessment.tests",
                    confidence=0.60,
                    reason=(
                        f"The transcript explicitly contains "
                        f"'{value} degrees bilaterally', but the "
                        "structured output did not represent the "
                        "same value on both sides."
                    ),
                )
            )

    # Completely empty extraction.
    has_extracted_content = any(
        value.strip()
        for _, value in collect_assessment_strings(
            assessment
        )
    )

    if transcript.strip() and not has_extracted_content:
        issues.append(
            ConfidenceIssue(
                field_path="assessment",
                confidence=0.0,
                reason=(
                    "The transcript contains content but the "
                    "assessment is completely empty."
                ),
            )
        )

    unique = {}

    for issue in issues:
        unique[
            (issue.field_path, issue.reason)
        ] = issue

    return list(unique.values())


# ============================================================
# LLM grounding
# ============================================================


def llm_grounding_check(
    transcript: str,
    assessment: FirstAssessment,
) -> ExtractionConfidence:

    llm = create_llm()

    verifier = llm.with_structured_output(
        ExtractionConfidence
    )

    assessment_json = assessment.model_dump_json(
        indent=2
    )

    prompt = f"""
You are a strict clinical extraction auditor.

Compare the structured assessment to the transcript.

Only flag information that is genuinely:
- unsupported
- fabricated
- contradicted
- clinically ambiguous

Do NOT flag merely because:
- a value is empty
- a qualitative finding is attached to a broader test
- a side has no numerical value
- the wording is a harmless paraphrase
- a clinically equivalent test label is used
- the assessment preserves an ambiguous source phrase

============================================================
IMPORTANT
============================================================

If the transcript explicitly states a qualitative finding such as:

"swelling"

then that finding is supported.

Do not require the transcript to separately say "swelling is present"
when the clinical statement itself clearly reports swelling.

If a test says:

"left hip extension restricted"

and the assessment places "restricted" in a related hip range-of-motion
test's comments while leaving the numerical side field empty, DO NOT
flag that as low confidence.

If a field is intentionally empty because the transcript gives no
numerical measurement, DO NOT flag it.

If the transcript explicitly states:

"45 degrees bilaterally"

then the correct representation is:

left = "45"
right = "45"

If the structured output fails to represent that explicit bilateral
measurement, flag the objective test.

============================================================
AMBIGUOUS VALUES
============================================================

If the transcript contains an expression such as:

"negic 5"

and the assessment preserves "negic 5" without guessing its meaning,
flag that field as ambiguous.

============================================================
CONFIDENCE
============================================================

1.00 = explicit and clear
0.90 = clear paraphrase
0.80 = controlled normalization
0.70 = supported but indirect
0.60 = ambiguous
0.40 = weakly supported
0.20 = mostly unsupported
0.00 = fabricated or contradicted

============================================================
FIELD PATHS
============================================================

Use the current schema:

clinicalDetails.duration

subjectiveAssessments[0].conclusion

objectiveAssessment.tests[0].right

objectiveGoals[1].goalName

recommendation[0].sessionFrequency

patientAdvice.adviceDetails

============================================================
SOURCE TRANSCRIPT
============================================================

{transcript}

============================================================
STRUCTURED ASSESSMENT
============================================================

{assessment_json}
"""

    result = verifier.invoke(prompt)

    return ExtractionConfidence.model_validate(
        result
    )


# ============================================================
# Public grounding interface
# ============================================================


def verify_assessment_grounding(
    transcript: str,
    assessment: FirstAssessment,
) -> ExtractionConfidence:

    deterministic = deterministic_confidence_issues(
        transcript=transcript,
        assessment=assessment,
    )

    try:
        llm_report = llm_grounding_check(
            transcript=transcript,
            assessment=assessment,
        )

    except Exception:
        return ExtractionConfidence(
            overall_confidence=0.0,
            issues=[
                *deterministic,
                ConfidenceIssue(
                    field_path="assessment",
                    confidence=0.0,
                    reason=(
                        "Grounding verification failed; "
                        "the extraction cannot be trusted."
                    ),
                ),
            ],
        )

    merged: dict[str, ConfidenceIssue] = {}

    for issue in llm_report.issues:
        merged[issue.field_path] = issue

    for issue in deterministic:
        existing = merged.get(issue.field_path)

        if existing is None:
            merged[issue.field_path] = issue

        elif issue.confidence < existing.confidence:
            merged[issue.field_path] = issue

    issues = list(merged.values())

    overall = llm_report.overall_confidence

    if issues:
        overall = min(
            overall,
            min(
                issue.confidence
                for issue in issues
            ),
        )

    if (
        overall < CONFIDENCE_THRESHOLD
        and not issues
    ):
        issues.append(
            ConfidenceIssue(
                field_path="assessment",
                confidence=overall,
                reason=(
                    "Overall extraction confidence is below "
                    "the configured threshold."
                ),
            )
        )

    return ExtractionConfidence(
        overall_confidence=max(
            0.0,
            min(1.0, overall),
        ),
        issues=issues,
    )