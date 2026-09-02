from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.models.assessment import FirstAssessment


# ============================================================
# EXTRACTION MODELS
# ============================================================


class FieldConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: FirstAssessment
    confidence: list[FieldConfidence] = Field(
        default_factory=list
    )


class ExtractionState(TypedDict, total=False):
    transcript: str
    raw_result: dict[str, Any]
    assessment: FirstAssessment
    confidence: list[FieldConfidence]


# ============================================================
# LLM
# ============================================================


def _build_llm() -> ChatOllama:
    settings = get_settings()

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        format="json",
    )


# ============================================================
# BASIC CLEANING
# ============================================================


def _clean_string(value: Any) -> str:
    """
    Convert any value to a clean string.

    Production schema requires strings rather than null.
    """
    if value is None:
        return ""

    return str(value).strip()


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    return {}


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value

    return []


# ============================================================
# SECTION CLEANERS
# ============================================================


def _clean_subjective_assessments(
    items: Any,
) -> list[dict[str, str]]:
    """
    Subjective assessments are allowed to contain ONLY:

        testName
        conclusion

    This prevents an LLM from accidentally putting objective
    fields such as left/right/unitName/comments here.
    """

    result: list[dict[str, str]] = []

    for item in _ensure_list(items):

        if not isinstance(item, dict):
            continue

        test_name = _clean_string(
            item.get("testName")
        )

        conclusion = _clean_string(
            item.get("conclusion")
        )

        # Ignore completely empty records.
        if not test_name and not conclusion:
            continue

        result.append(
            {
                "testName": test_name,
                "conclusion": conclusion,
            }
        )

    return result


def _clean_objective_tests(
    items: Any,
) -> list[dict[str, str]]:
    """
    Objective tests must contain exactly the fields required
    by the production schema.
    """

    result: list[dict[str, str]] = []

    allowed_fields = (
        "testName",
        "unitName",
        "value",
        "left",
        "right",
        "comments",
    )

    for item in _ensure_list(items):

        if not isinstance(item, dict):
            continue

        cleaned = {
            field: _clean_string(
                item.get(field)
            )
            for field in allowed_fields
        }

        # Ignore completely empty objective records.
        if not any(cleaned.values()):
            continue

        result.append(cleaned)

    return result


def _clean_subjective_goals(
    items: Any,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    for item in _ensure_list(items):

        if not isinstance(item, dict):
            continue

        cleaned = {
            "goalDetails": _clean_string(
                item.get("goalDetails")
            ),
            "targetDate": _clean_string(
                item.get("targetDate")
            ),
        }

        if not any(cleaned.values()):
            continue

        result.append(cleaned)

    return result


def _clean_objective_goals(
    items: Any,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    allowed_fields = (
        "goalName",
        "goalCategory",
        "unitName",
        "value",
        "targetDate",
    )

    for item in _ensure_list(items):

        if not isinstance(item, dict):
            continue

        cleaned = {
            field: _clean_string(
                item.get(field)
            )
            for field in allowed_fields
        }

        if not any(cleaned.values()):
            continue

        result.append(cleaned)

    return result


def _clean_recommendations(
    items: Any,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    for item in _ensure_list(items):

        if not isinstance(item, dict):
            continue

        cleaned = {
            "sessionType": _clean_string(
                item.get("sessionType")
            ),
            "sessionFrequency": _clean_string(
                item.get("sessionFrequency")
            ),
        }

        if not any(cleaned.values()):
            continue

        result.append(cleaned)

    return result


# ============================================================
# RAW LLM -> PRODUCTION FIRSTASSESSMENT
# ============================================================


def _extract_assessment(
    raw: dict[str, Any],
) -> FirstAssessment:
    """
    Convert potentially imperfect LLM JSON into the exact
    FirstAssessment production schema.

    IMPORTANT:
    This function never creates clinical information.
    It only maps/cleans information already returned by the LLM.
    """

    assessment = _ensure_dict(
        raw.get("assessment")
    )

    clinical_details = _ensure_dict(
        assessment.get("clinicalDetails")
    )

    objective_assessment = _ensure_dict(
        assessment.get("objectiveAssessment")
    )

    patient_advice = _ensure_dict(
        assessment.get("patientAdvice")
    )

    data = {
        "clinicalDetails": {
            "clinicalHistory": _clean_string(
                clinical_details.get(
                    "clinicalHistory"
                )
            ),
            "chiefComplaint": _clean_string(
                clinical_details.get(
                    "chiefComplaint"
                )
            ),
            "duration": _clean_string(
                clinical_details.get(
                    "duration"
                )
            ),
        },

        "subjectiveAssessments": (
            _clean_subjective_assessments(
                assessment.get(
                    "subjectiveAssessments"
                )
            )
        ),

        "objectiveAssessment": {
            "tests": _clean_objective_tests(
                objective_assessment.get(
                    "tests"
                )
            )
        },

        "subjectiveGoals": (
            _clean_subjective_goals(
                assessment.get(
                    "subjectiveGoals"
                )
            )
        ),

        "objectiveGoals": (
            _clean_objective_goals(
                assessment.get(
                    "objectiveGoals"
                )
            )
        ),

        "recommendation": (
            _clean_recommendations(
                assessment.get(
                    "recommendation"
                )
            )
        ),

        "patientAdvice": {
            "adviceDetails": _clean_string(
                patient_advice.get(
                    "adviceDetails"
                )
            )
        },
    }

    return FirstAssessment.model_validate(
        data
    )


# ============================================================
# CONFIDENCE
# ============================================================


def _extract_confidence(
    raw: dict[str, Any],
) -> list[FieldConfidence]:

    result: list[FieldConfidence] = []

    items = raw.get("confidence")

    for item in _ensure_list(items):

        if not isinstance(item, dict):
            continue

        field = _clean_string(
            item.get("field")
        )

        if not field:
            continue

        try:
            score = float(
                item.get(
                    "confidence",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        result.append(
            FieldConfidence(
                field=field,
                confidence=score,
                reason=_clean_string(
                    item.get("reason")
                ),
            )
        )

    return result


# ============================================================
# PROMPT
# ============================================================

def _build_prompt(
    transcript: str,
) -> str:

    return f"""
You are a clinical documentation extraction system.

Extract structured clinical information from the transcript below.

CRITICAL RULE:
Use ONLY information explicitly present in the transcript.

NEVER invent clinical information.

If information is not explicitly stated:
- string field = ""
- array field = []

============================================================
1. CLINICAL DETAILS
============================================================

clinicalHistory:
Extract the complete relevant clinical history explicitly stated.

Include, when present:
- mechanism of injury
- accident/event
- diagnosis
- fracture
- ligament injury
- surgery
- procedure
- postoperative status
- non-weight-bearing period
- progressive loading
- previous treatment

IMPORTANT:
Do NOT leave clinicalHistory empty if the transcript contains
these facts.

chiefComplaint:
Extract the patient's presenting complaints.

duration:
Extract only an explicitly stated duration.
Never calculate a duration.

============================================================
2. SUBJECTIVE ASSESSMENTS
============================================================

Each item MUST contain ONLY:

testName
conclusion

NEVER put these fields here:

unitName
value
left
right
comments

Only create a subjective assessment when there is an explicitly
stated subjective assessment, named test, or subjective finding.

Example:

"moderate pain"

may become:

{{
  "testName": "pain",
  "conclusion": "moderate"
}}

============================================================
3. OBJECTIVE ASSESSMENT
============================================================

Each test MUST contain EXACTLY:

testName
unitName
value
left
right
comments

Use objectiveAssessment for:

- range of motion
- strength
- pain measurements
- swelling
- mobility
- scars
- physical examination findings
- left/right measurements
- objective observations

============================================================
4. MEASUREMENT RULES
============================================================

For bilateral measurements:

value MUST be ""

left = left measurement

right = right measurement

Example:

"left knee flexion of 124 degrees compared with 130 degrees
on the right"

MUST produce:

{{
  "testName": "knee flexion",
  "unitName": "degrees",
  "value": "",
  "left": "124",
  "right": "130",
  "comments": ""
}}

DO NOT put "124" into value.

============================================================

Negative measurements MUST preserve the negative sign.

Example:

"left knee extension of 20 degrees compared with negative
5 degrees on the right"

MUST produce:

{{
  "testName": "knee extension",
  "unitName": "degrees",
  "value": "",
  "left": "20",
  "right": "-5",
  "comments": ""
}}

"negative 5", "minus 5", "negative five", and "minus five"
must become "-5".

NEVER convert a negative measurement into a positive value.

============================================================

For bilateral measurements:

"hip internal rotation of 45 degrees bilaterally"

becomes:

{{
  "testName": "hip internal rotation",
  "unitName": "degrees",
  "value": "",
  "left": "45",
  "right": "45",
  "comments": ""
}}

============================================================

For a single measurement where left/right are not specified:

"pain score was 5"

becomes:

{{
  "testName": "pain score",
  "unitName": "",
  "value": "5",
  "left": "",
  "right": "",
  "comments": ""
}}

============================================================
5. OBJECTIVE FINDINGS WITHOUT NUMBERS
============================================================

If an objective finding has no numerical measurement, put the
finding in comments.

Example:

"healed surgical scar was noted on the medial aspect of the knee"

becomes:

{{
  "testName": "surgical scar",
  "unitName": "",
  "value": "",
  "left": "",
  "right": "",
  "comments": "Healed surgical scar on medial aspect of knee"
}}

============================================================
6. SUBJECTIVE GOALS
============================================================

Only extract patient-centered goals explicitly stated.

Do not invent goals.

Do not invent target dates.

If none are explicitly stated:

[]

============================================================
7. OBJECTIVE GOALS
============================================================

Extract ALL clinician-defined rehabilitation goals explicitly
stated in the transcript.

Extract each goal separately.

For example:

"restoring the extension"

becomes:

{{
  "goalName": "restore extension",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

"improving the stability"

becomes:

{{
  "goalName": "improve stability",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

"improving single leg stability"

becomes:

{{
  "goalName": "improve single leg stability",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

"strengthening the quadriceps"

becomes:

{{
  "goalName": "strengthen quadriceps",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

"strengthening the functional lower limb musculature"

becomes:

{{
  "goalName": "strengthen functional lower limb musculature",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

"improving ankle mobility"

becomes:

{{
  "goalName": "improve ankle mobility",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

"activating the posterior chain"

becomes:

{{
  "goalName": "activate posterior chain",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}}

IMPORTANT:
Do not omit explicitly stated rehabilitation goals.

Do not invent:
- target dates
- numerical values
- goal categories

============================================================
8. RECOMMENDATION
============================================================

Extract explicit treatment recommendations.

Example:

"physiotherapy was recommended once weekly for four sessions"

becomes:

{{
  "sessionType": "physiotherapy",
  "sessionFrequency": "once weekly for four sessions"
}}

============================================================
9. PATIENT ADVICE
============================================================

Only extract explicit instructions/advice given to the patient.

IMPORTANT:

"pain is relieved with rest"

is NOT patient advice.

It describes symptom behavior.

Therefore:

{{
  "adviceDetails": ""
}}

unless the clinician explicitly advises the patient to rest.

Never invent advice.

============================================================
10. CONFIDENCE
============================================================

Return confidence scores from 0 to 1.

Confidence must represent how clearly each extracted field is
supported by the transcript.

Do not use confidence to justify invented information.

============================================================
11. EXACT JSON STRUCTURE
============================================================

Return ONLY valid JSON.

No markdown.

No explanation.

No comments.

No additional fields.

The top-level object MUST contain exactly:

assessment
confidence

The assessment object MUST contain exactly:

clinicalDetails
subjectiveAssessments
objectiveAssessment
subjectiveGoals
objectiveGoals
recommendation
patientAdvice

The structure MUST be:

{{
  "assessment": {{
    "clinicalDetails": {{
      "clinicalHistory": "",
      "chiefComplaint": "",
      "duration": ""
    }},
    "subjectiveAssessments": [],
    "objectiveAssessment": {{
      "tests": []
    }},
    "subjectiveGoals": [],
    "objectiveGoals": [],
    "recommendation": [],
    "patientAdvice": {{
      "adviceDetails": ""
    }}
  }},
  "confidence": []
}}

============================================================
12. FINAL CHECK BEFORE RETURNING JSON
============================================================

Before returning the answer, verify:

1. clinicalHistory contains explicit history from the transcript.
2. Bilateral measurements have value="".
3. Left/right measurements are in left/right.
4. Negative measurements keep their "-" sign.
5. Objective rehabilitation goals are extracted.
6. Patient advice is not inferred from symptom relief.
7. No fields outside the required schema are included.
8. Missing information uses "" or [].
9. No clinical information has been invented.

============================================================
TRANSCRIPT
============================================================

{transcript}

============================================================
RETURN ONLY JSON
============================================================
"""



# ============================================================
# LLM RESPONSE PARSING
# ============================================================


def _parse_llm_json(
    content: Any,
) -> dict[str, Any]:
    """
    Parse ChatOllama's response safely.

    Ollama normally returns a JSON string, but different
    LangChain versions can represent content differently.
    """

    if isinstance(content, dict):
        return content

    if isinstance(content, list):

        parts: list[str] = []

        for item in content:

            if isinstance(item, dict):
                if "text" in item:
                    parts.append(
                        str(item["text"])
                    )
                elif "content" in item:
                    parts.append(
                        str(item["content"])
                    )
                else:
                    parts.append(
                        json.dumps(item)
                    )
            else:
                parts.append(str(item))

        content = "".join(parts)

    text = str(content).strip()

    if not text:
        raise ValueError(
            "LLM returned an empty response."
        )

    # Remove accidental markdown fences.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:

        raise ValueError(
            "LLM returned invalid JSON: "
            f"{exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "LLM JSON response must be an object."
        )

    return parsed


# ============================================================
# LANGGRAPH EXTRACTION NODE
# ============================================================


def _extraction_node(
    state: ExtractionState,
) -> ExtractionState:

    transcript = _clean_string(
        state.get("transcript")
    )

    if not transcript:
        raise ValueError(
            "Transcript is empty."
        )

    llm = _build_llm()

    prompt = _build_prompt(
        transcript
    )

    try:

        response = llm.invoke(
            prompt
        )

        raw = _parse_llm_json(
            response.content
        )

        assessment = _extract_assessment(
            raw
        )

        confidence = _extract_confidence(
            raw
        )

        return {
            "transcript": transcript,
            "raw_result": raw,
            "assessment": assessment,
            "confidence": confidence,
        }

    except ValidationError as exc:

        raise RuntimeError(
            "LLM output failed FirstAssessment "
            f"validation: {exc}"
        ) from exc

    except Exception as exc:

        raise RuntimeError(
            f"Clinical extraction failed: {exc}"
        ) from exc


# ============================================================
# VALIDATION NODE
# ============================================================


def _validation_node(
    state: ExtractionState,
) -> ExtractionState:

    assessment = FirstAssessment.model_validate(
        state["assessment"].model_dump(
            mode="json"
        )
    )

    return {
        **state,
        "assessment": assessment,
    }


# ============================================================
# LANGGRAPH
# ============================================================


def _build_graph():

    graph = StateGraph(
        ExtractionState
    )

    graph.add_node(
        "extract",
        _extraction_node,
    )

    graph.add_node(
        "validate",
        _validation_node,
    )

    graph.add_edge(
        START,
        "extract",
    )

    graph.add_edge(
        "extract",
        "validate",
    )

    graph.add_edge(
        "validate",
        END,
    )

    return graph.compile()


_graph = None


def get_extraction_graph():

    global _graph

    if _graph is None:
        _graph = _build_graph()

    return _graph


# ============================================================
# PUBLIC API
# ============================================================


def extract_assessment(
    transcript: str,
) -> ExtractionResult:

    transcript = _clean_string(
        transcript
    )

    if not transcript:
        raise ValueError(
            "Transcript is empty."
        )

    graph = get_extraction_graph()

    result = graph.invoke(
        {
            "transcript": transcript
        }
    )

    return ExtractionResult(
        assessment=result["assessment"],
        confidence=result.get(
            "confidence",
            [],
        ),
    )
