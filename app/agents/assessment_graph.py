from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)
from app.schemas.extraction import LowConfidenceField
from app.schemas.raw_extraction import RawFirstAssessment

SYSTEM_PROMPT = """You are a clinical documentation extraction assistant.

You will be given a transcript of a conversation between a clinician and a \
patient. Extract information into the given structured schema.

CRITICAL RULES — follow these exactly:
1. Only extract information that is explicitly stated or very clearly and \
unambiguously implied in the transcript. Do NOT infer, guess, estimate, or \
invent any clinical value, score, date, or detail that isn't actually there.
2. For every field, set `confidence` honestly:
   - 1.0 = explicitly and unambiguously stated in the transcript
   - 0.5-0.8 = reasonably implied but not stated in so many words
   - 0.0-0.3 = not mentioned at all, or you are guessing
3. If a field is not mentioned in the transcript at all, set `value` to an \
empty string "" and `confidence` to 0.0. Never fabricate a plausible-looking \
value just to fill the field.
4. `evidence` should be a SHORT quote or paraphrase (max ~10 words) from \
the transcript that supports the value. Keep it brief - it's just a pointer, \
not a full excerpt. If value is "", evidence should also be "".
5. For list sections (subjectiveAssessments, objectiveAssessment.tests, \
subjectiveGoals, objectiveGoals, recommendation): include one entry per \
distinct item actually discussed in the transcript. If none were discussed, \
return an empty list — do not invent a placeholder entry.
6. Never let your general medical knowledge fill in a value that the \
transcript doesn't support. Absence of information is not something to fix.

DO NOT UNDER-EXTRACT. Being overly conservative is also a mistake — if the \
transcript contains a real qualitative or descriptive detail, capture it, \
even if it isn't a number. Specifically:
- `objectiveAssessment.tests[].comments`: this field exists specifically \
for qualitative findings mentioned alongside a measurement — pain, \
restriction, swelling, quality of movement, overpressure findings, etc. \
If the clinician describes how a movement/test looked or felt (e.g. \
"restricted and painful on overpressure", "full and pain free", \
"with swelling"), that observation MUST go into `comments` for that test, \
not be dropped. An empty `comments` should only happen when literally \
nothing qualitative was said about that specific test.
- `subjectiveAssessments`: this captures the PATIENT's own reported \
symptoms/experience (pain level, irritability, what aggravates or relieves \
symptoms, functional limitations they describe). If the transcript reports \
what the patient says they feel or experience (e.g. "patient reports \
moderate pain, worse with prolonged walking, relieved by rest"), create an \
entry here with `testName` describing what's being assessed (e.g. "Pain \
and symptom irritability") and `conclusion` summarizing what was reported. \
Do not leave this empty just because no formal named test (like a VAS \
score) was administered — a patient's narrated symptom report counts.
- Re-read the transcript once specifically looking for sentences you have \
not yet used anywhere in your extraction. If a sentence describes a \
clinical finding, symptom, or observation that doesn't obviously map to a \
field you've already filled, find the closest correct field for it \
(comments, clinicalHistory, or a subjectiveAssessments entry) rather than \
discarding it.
- Do NOT truncate or shorten a field to just the first clause when the \
transcript lists multiple related items together (e.g. a chief complaint \
naming several symptoms/locations). Capture the full statement as given, \
not just the first symptom mentioned.
- `comments` should ONLY contain an actual qualitative observation from the \
transcript (pain, restriction, swelling, quality, "full and pain free", \
etc.). Never repeat the test name itself as the comment. If nothing \
qualitative was said about that specific test, leave `comments` as "".
- Numeric-looking fields (measurements, degrees, scores) should contain \
only the number/unit as clearly stated. If Whisper's transcription of a \
number seems garbled or ambiguous (e.g. an unclear word before a digit), \
use your best judgment to extract the numeric value that was clearly \
intended (e.g. a negative-sign word before a number means a negative \
value), rather than passing through unclear transcribed text verbatim.

CRITICAL — DO NOT MIX UP WHICH TEST A COMMENT BELONGS TO:
- Before writing a `comments` value for a specific test (e.g. ankle \
dorsiflexion), check that the qualitative observation you're about to use \
is actually about THAT body part/movement, not a different one mentioned \
elsewhere in the transcript. A comment about the hip must never end up on \
a knee or ankle test, and vice versa. If you are not sure which test a \
qualitative statement belongs to, leave `comments` as "" for the \
ambiguous test rather than guessing and attaching it to the wrong one — \
an incorrect attribution is worse than a blank field.
- Do not use filler words like "bilaterally" as a comment — that describes \
measurement symmetry, not a clinical finding, and belongs implicitly in \
the left/right values, not in `comments`.

TREATMENT GOALS — DO NOT SKIP THESE:
- If the clinician states specific things treatment should achieve or work \
on (e.g. "restoring extension", "improving stability", "strengthening the \
quadriceps", "improving ankle mobility"), these are treatment goals and \
belong in `objectiveGoals`, one entry per distinct goal mentioned. Use: \
`goalName` = the goal itself (e.g. "Restore knee extension"), \
`goalCategory` = a short category (e.g. "Range of Motion", "Strength", \
"Stability", "Mobility"), and leave `unitName`/`value`/`targetDate` as "" \
if no specific number or date was given for that goal. Do not leave \
`objectiveGoals` empty just because no numeric target was attached to the \
goals — a list of stated treatment aims still counts and must be captured.
"""

USER_PROMPT = "Transcript:\n\n{transcript}\n\nExtract the structured clinical assessment now."


class GraphState(TypedDict, total=False):
    transcript: str
    raw: RawFirstAssessment
    low_confidence_fields: list[LowConfidenceField]
    assessment: FirstAssessment | None


@dataclass
class AssessmentGraphResult:
    assessment: FirstAssessment | None
    low_confidence_fields: list[LowConfidenceField] = field(default_factory=list)
    transcript: str = ""

    @property
    def passed(self) -> bool:
        return self.assessment is not None and not self.low_confidence_fields


def _get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Set it in your .env file.")
    return ChatGoogleGenerativeAI(
        google_api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=0,
        max_output_tokens=8000,
    )


def _extract_node(state: GraphState) -> GraphState:
    llm = _get_llm().with_structured_output(RawFirstAssessment, include_raw=True)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    chain = prompt | llm

    last_error: Exception | str | None = None
    attempt = 0
    for attempt in range(3):  # retries if the model's tool-call JSON gets truncated/malformed/empty
        try:
            result = chain.invoke({"transcript": state["transcript"]})
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        # With include_raw=True, result is a dict: {"raw": <AIMessage>,
        # "parsed": RawFirstAssessment | None, "parsing_error": Exception | None}
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result

        if parsing_error is not None:
            last_error = parsing_error
            continue

        if parsed is None:
            # Some providers (observed with Gemini via langchain) can return
            # no parsed output without a parsing_error - treat as a failed
            # attempt too rather than silently propagating None.
            last_error = "LLM returned no structured output (None) with no parsing_error."
            continue

        return {"raw": parsed}

    raise RuntimeError(
        f"Clinical extraction failed after {attempt + 1} attempts (LLM output could not "
        f"be parsed into the expected schema, likely truncated or malformed): {last_error}"
    )


def _check_confidence_node(state: GraphState) -> GraphState:
    settings = get_settings()
    threshold = settings.confidence_threshold
    raw = state["raw"]
    if raw is None:
        raise RuntimeError(
            "Internal error: no extraction result reached the confidence gate. "
            "This should not happen - _extract_node should have raised first."
        )
    low_confidence: list[LowConfidenceField] = []

    def check(path: str, extraction) -> None:  # noqa: ANN001
        if extraction.value and extraction.confidence < threshold:
            low_confidence.append(
                LowConfidenceField(
                    field=path,
                    confidence=extraction.confidence,
                    reason=(
                        f"Model produced a value but confidence "
                        f"({extraction.confidence:.2f}) is below threshold "
                        f"({threshold:.2f}); refusing to include it to avoid "
                        f"hallucinating clinical data."
                    ),
                )
            )

    cd = raw.clinicalDetails
    check("clinicalDetails.clinicalHistory", cd.clinicalHistory)
    check("clinicalDetails.chiefComplaint", cd.chiefComplaint)
    check("clinicalDetails.duration", cd.duration)

    for i, sa in enumerate(raw.subjectiveAssessments):
        check(f"subjectiveAssessments[{i}].testName", sa.testName)
        check(f"subjectiveAssessments[{i}].conclusion", sa.conclusion)

    for i, t in enumerate(raw.objectiveAssessment.tests):
        check(f"objectiveAssessment.tests[{i}].testName", t.testName)
        check(f"objectiveAssessment.tests[{i}].unitName", t.unitName)
        check(f"objectiveAssessment.tests[{i}].value", t.value)
        check(f"objectiveAssessment.tests[{i}].left", t.left)
        check(f"objectiveAssessment.tests[{i}].right", t.right)
        check(f"objectiveAssessment.tests[{i}].comments", t.comments)

    for i, sg in enumerate(raw.subjectiveGoals):
        check(f"subjectiveGoals[{i}].goalDetails", sg.goalDetails)
        check(f"subjectiveGoals[{i}].targetDate", sg.targetDate)

    for i, og in enumerate(raw.objectiveGoals):
        check(f"objectiveGoals[{i}].goalName", og.goalName)
        check(f"objectiveGoals[{i}].goalCategory", og.goalCategory)
        check(f"objectiveGoals[{i}].unitName", og.unitName)
        check(f"objectiveGoals[{i}].value", og.value)
        check(f"objectiveGoals[{i}].targetDate", og.targetDate)

    for i, rec in enumerate(raw.recommendation):
        check(f"recommendation[{i}].sessionType", rec.sessionType)
        check(f"recommendation[{i}].sessionFrequency", rec.sessionFrequency)

    check("patientAdvice.adviceDetails", raw.patientAdvice.adviceDetails)

    if low_confidence:
        return {"low_confidence_fields": low_confidence, "assessment": None}

    assessment = _map_to_first_assessment(raw)
    return {"low_confidence_fields": [], "assessment": assessment}


def _map_to_first_assessment(raw: RawFirstAssessment) -> FirstAssessment:
    """Strip confidence/evidence, keep only the plain string values, and
    build the production FirstAssessment object. Values that weren't
    confidently extracted default to "" (never None) per the schema rule
    that string fields must always be strings."""

    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory=raw.clinicalDetails.clinicalHistory.value,
            chiefComplaint=raw.clinicalDetails.chiefComplaint.value,
            duration=raw.clinicalDetails.duration.value,
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(testName=sa.testName.value, conclusion=sa.conclusion.value)
            for sa in raw.subjectiveAssessments
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName=t.testName.value,
                    unitName=t.unitName.value,
                    value=t.value.value,
                    left=t.left.value,
                    right=t.right.value,
                    comments=t.comments.value,
                )
                for t in raw.objectiveAssessment.tests
            ]
        ),
        subjectiveGoals=[
            SubjectiveGoal(goalDetails=sg.goalDetails.value, targetDate=sg.targetDate.value)
            for sg in raw.subjectiveGoals
        ],
        objectiveGoals=[
            ObjectiveGoal(
                goalName=og.goalName.value,
                goalCategory=og.goalCategory.value,
                unitName=og.unitName.value,
                value=og.value.value,
                targetDate=og.targetDate.value,
            )
            for og in raw.objectiveGoals
        ],
        recommendation=[
            Recommendation(sessionType=r.sessionType.value, sessionFrequency=r.sessionFrequency.value)
            for r in raw.recommendation
        ],
        patientAdvice=PatientAdvice(adviceDetails=raw.patientAdvice.adviceDetails.value),
    )


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("extract", _extract_node)
    graph.add_node("check_confidence", _check_confidence_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "check_confidence")
    graph.add_edge("check_confidence", END)
    return graph.compile()


_compiled_graph = None


def run_assessment_graph(transcript: str) -> AssessmentGraphResult:
    """Run the full extraction graph on a transcript and return the result."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    final_state: dict[str, Any] = _compiled_graph.invoke({"transcript": transcript})
    return AssessmentGraphResult(
        assessment=final_state.get("assessment"),
        low_confidence_fields=final_state.get("low_confidence_fields", []),
        transcript=transcript,
    )
