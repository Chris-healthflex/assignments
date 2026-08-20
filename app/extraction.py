"""LangGraph extraction agent: transcript -> FirstAssessment + per-field evidence.

    START ─┬─> subjective ─┐
           ├─> objective ──┼─> assemble ─> ground ─> route ─> END
           └─> plan ───────┘        ^                 │
                                    └──── repair <────┘

**Three nodes, not one call and not seven.** A single call producing all seven
sections at once gets measurably lazier about the later ones -- Gemini degrades
as an output schema deepens. The other extreme, one call per section, was what
this started as, and it broke on contact with reality: the Gemini free tier
allows five requests per minute and a few dozen per day, so a seven-call fan-out
429s on arrival and burns a day's quota in three runs. Three nodes grouped by
what a clinician actually reasons about together -- the subjective picture, the
measurements, the plan -- keeps each call focused, fits inside the free tier,
and leaves headroom for the repair pass.

**Every value must be quoted.** A node returns its data *and* a list of
`Citation`s -- the verbatim transcript span each value came from. We then check
those quotes against the transcript **in our own code**. This is the difference
between asking a model to be honest and being able to tell whether it was: a
value whose quote is not in the transcript was invented, full stop, and it
scores zero no matter how confident the model claims to be.

**We compute the field paths, the model does not.** Citations carry a value and
a quote; the dotted path (``objectiveAssessment.tests[0].value``) is derived by
walking the assembled document ourselves. Asking an LLM to emit correct array
indices is a needless way to lose data.

**Repair targets hallucination specifically.** When grounding finds a quote that
is not in the transcript, the repair pass re-asks *only the affected group*,
naming the bad quote and instructing the model to either quote correctly or
leave the field empty. Leaving it empty is an acceptable answer here; guessing
is not.
"""

from __future__ import annotations

import logging
import operator
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import (
    GARBLED,
    SECTIONS,
    ClinicalDetails,
    ExtractionFlags,
    FieldEvidence,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class ExtractionFailed(RuntimeError):
    """Raised when the agent cannot produce anything usable at all."""


# --------------------------------------------------------------------------- #
# What the model is asked to return
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """One value and the transcript span the model claims it came from."""

    value: str = Field(description="The exact value you placed in a field.")
    evidence: str = Field(
        description=(
            "The verbatim span of the transcript that states this value. Copy it "
            "character for character. Do not paraphrase, summarise or correct it."
        )
    )
    confidence: float = Field(
        default=0.0,
        description=(
            "Required. How certain you are that this value is stated in the "
            "transcript, from 0.0 to 1.0. Always give a number."
        ),
    )


class SubjectiveOut(BaseModel):
    clinicalDetails: ClinicalDetails
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ObjectiveOut(BaseModel):
    objectiveAssessment: ObjectiveAssessment
    citations: list[Citation] = Field(default_factory=list)


class PlanOut(BaseModel):
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice
    citations: list[Citation] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You extract structured clinical data from a transcript of a physiotherapy
assessment. The transcript is the only source of truth available to you.

Absolute rules:
- Extract ONLY what the transcript explicitly states. Never infer, complete or
  correct a clinical value, score, measurement or date.
- If the transcript does not state something, leave the string empty ("") or
  the array empty. An empty field is a correct answer. A guessed field is not.
- Never invent a date. If no date or deadline was spoken, targetDate is "".
- Copy values as spoken. Do not convert units, normalise numbers, expand
  abbreviations or tidy up wording.
- If the transcript is garbled at some point, extract what is actually written
  there. Do not repair it into what you think was meant.
- Do not record the same finding twice under different names.

Citations are mandatory. For EVERY non-empty field you fill -- including names,
labels and categories you chose yourself, such as testName or goalCategory --
add one entry to `citations` containing the value you wrote and the verbatim
transcript span it came from.

- For a value taken from the transcript, quote the span that states it.
- For a label you chose yourself, quote the span that the label describes. For
  example, a testName of "Pain characteristics" cites the span describing the
  pain, because nobody said the words "pain characteristics" aloud.

Copy the span character for character from the transcript. Do not stitch words
together from different parts of a sentence, and do not tidy the wording: the
span is checked against the transcript automatically, and a field whose span
cannot be found there is discarded as unsupported.

Quote the shortest span that still establishes the value. Include the words
that give it meaning -- a side, a unit, a sign -- and nothing further."""

# group -> (response model, sections it fills, what to extract)
GROUP_SPECS: dict[str, tuple[type[BaseModel], tuple[str, ...], str]] = {
    "subjective": (
        SubjectiveOut,
        ("clinicalDetails", "subjectiveAssessments"),
        "`clinicalDetails`: the patient's clinical history, their chief "
        "complaint, and how long they have had it (`duration`).\n"
        "`subjectiveAssessments`: findings that were observed, reported or "
        "concluded rather than measured as a number. Each entry is a testName "
        "and the conclusion drawn. One entry per distinct finding.",
    ),
    "objective": (
        ObjectiveOut,
        ("objectiveAssessment",),
        "`objectiveAssessment.tests`: measured values only. Each test has a "
        "testName, a unitName (e.g. 'degrees'), and either a single `value` or "
        "separate `left`/`right` values when both sides were measured. Put "
        "anything qualifying the measurement in `comments`. Leave a side empty "
        "if it was not measured.",
    ),
    "plan": (
        PlanOut,
        ("subjectiveGoals", "objectiveGoals", "recommendation", "patientAdvice"),
        "`subjectiveGoals`: goals described in words, with no measured target. "
        "`objectiveGoals`: goals with a measurable target -- goalName, "
        "goalCategory (e.g. 'Range of motion', 'Strength'), unitName and target "
        "value. A goal belongs in exactly one of these two lists, never both. "
        "targetDate only if a date or deadline was actually spoken.\n"
        "`recommendation`: what kind of sessions were recommended (sessionType) "
        "and how often (sessionFrequency).\n"
        "`patientAdvice`: advice for the patient to follow themselves -- "
        "self-management, activity modification, home instructions. If the "
        "transcript contains none, leave it empty rather than repeating the "
        "treatment plan.",
    ),
}

SECTION_TO_GROUP: dict[str, str] = {
    section: group for group, (_, sections, _) in GROUP_SPECS.items() for section in sections
}


# --------------------------------------------------------------------------- #
# Model access
# --------------------------------------------------------------------------- #
@lru_cache
def _llm():
    """Build the chat model. Imported lazily so tests need no API key.

    Cached deliberately: the rate limiter lives on the model instance, so a
    fresh client per call would hand each node its own private quota and limit
    nothing at all. One shared instance is what makes the pacing real.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    if not settings.google_api_key:
        raise ExtractionFailed(
            "GOOGLE_API_KEY is not set; copy .env.example to .env and add your key."
        )
    return ChatGoogleGenerativeAI(
        model=settings.extraction_model,
        temperature=settings.extraction_temperature,
        google_api_key=settings.google_api_key,
        # Server-side 429s still happen (quota is per-project, not per-process),
        # so pace *and* retry rather than trusting either alone.
        max_retries=5,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=settings.extraction_requests_per_minute / 60.0,
            check_every_n_seconds=0.5,
            max_bucket_size=1,
        ),
    )


def _run_group(group: str, transcript: str, hint: str = "") -> BaseModel | None:
    """Ask the model for one group of sections. None if the call fails outright."""
    response_model, sections, description = GROUP_SPECS[group]
    prompt = (
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"Extract these sections: {', '.join(sections)}.\n\n{description}\n"
    )
    if hint:
        prompt += f"\n{hint}\n"
    try:
        model = _llm().with_structured_output(response_model)
        return model.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
    except Exception as exc:  # noqa: BLE001 -- one bad group must not sink the run
        logger.warning("Group %s failed: %s", group, str(exc)[:200])
        return None


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
def _merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class ExtractionState(TypedDict, total=False):
    transcript: str
    transcription: TranscriptionResult | None
    # Reducers, because the group nodes write concurrently.
    sections: Annotated[dict[str, Any], _merge]
    citations: Annotated[list[Citation], operator.add]
    assessment: FirstAssessment | None
    flags: ExtractionFlags | None
    errors: Annotated[list[str], operator.add]
    attempts: int


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def _sections_from(group: str, result: BaseModel) -> dict[str, Any]:
    _, sections, _ = GROUP_SPECS[group]
    return {section: getattr(result, section) for section in sections}


def _make_group_node(group: str):
    def node(state: ExtractionState) -> dict[str, Any]:
        result = _run_group(group, state["transcript"])
        if result is None:
            return {"errors": [f"Group {group} could not be extracted."]}
        return {
            "sections": _sections_from(group, result),
            "citations": list(result.citations),
        }

    node.__name__ = f"extract_{group}"
    return node


def assemble_node(state: ExtractionState) -> dict[str, Any]:
    """Merge the groups into one contract document.

    Any section that failed simply stays at its schema default -- an empty
    string or an empty array -- which is the honest representation of "we do not
    know" and keeps the document valid rather than half-formed.
    """
    sections = state.get("sections") or {}
    try:
        assessment = FirstAssessment.model_validate(
            {k: v for k, v in sections.items() if k in SECTIONS}
        )
    except Exception as exc:  # noqa: BLE001
        return {"assessment": FirstAssessment(), "errors": [f"Assembly failed: {exc}"]}
    return {"assessment": assessment}


def ground_node(state: ExtractionState) -> dict[str, Any]:
    """Verify every populated field against the transcript, in our own code."""
    assessment = state.get("assessment") or FirstAssessment()
    fields = ground(
        assessment,
        state["transcript"],
        state.get("transcription"),
        state.get("citations") or [],
    )
    unresolved = [
        path for path, value in _leaves(assessment.model_dump()) if not str(value).strip()
    ]
    return {
        "flags": ExtractionFlags.summarise(
            fields, unresolved=unresolved, warnings=state.get("errors") or []
        )
    }


def repair_node(state: ExtractionState) -> dict[str, Any]:
    """Re-ask only the groups that produced unverifiable quotes."""
    flags = state.get("flags") or ExtractionFlags()
    by_group: dict[str, list[FieldEvidence]] = {}
    for field in flags.ungrounded():
        group = SECTION_TO_GROUP.get(_section_of(field.field))
        if group:
            by_group.setdefault(group, []).append(field)

    updates: dict[str, Any] = {}
    citations: list[Citation] = []
    for group, offenders in by_group.items():
        detail = "\n".join(
            f'- Field `{f.field}`: you gave "{f.value}" citing '
            f'"{f.evidence or "(no quote at all)"}", which does not appear in the transcript.'
            for f in offenders
        )
        hint = (
            "Your previous attempt at this section was rejected:\n"
            f"{detail}\n"
            "Quote the transcript exactly, or leave the field empty. Leaving it "
            "empty is the correct answer when the transcript does not say it. "
            "Do not substitute a different guess."
        )
        result = _run_group(group, state["transcript"], hint=hint)
        if result is not None:
            updates.update(_sections_from(group, result))
            citations.extend(result.citations)

    return {
        "sections": updates,
        "citations": citations,
        "attempts": state.get("attempts", 0) + 1,
    }


def _route(state: ExtractionState) -> str:
    """Repair only while something is ungrounded and retries remain."""
    settings = get_settings()
    flags = state.get("flags") or ExtractionFlags()
    if not flags.ungrounded():
        return "done"
    if state.get("attempts", 0) >= settings.extraction_max_retries:
        logger.info(
            "Out of repair attempts; %d field(s) stay ungrounded.", len(flags.ungrounded())
        )
        return "done"
    return "repair"


# --------------------------------------------------------------------------- #
# Grounding (plain code -- no model involved, which is the point)
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def _leaves(node: Any, path: str = ""):
    """Yield every (dotted_path, value) leaf of a dumped document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaves(value, f"{path}[{index}]")
    else:
        yield path, node


def _section_of(path: str) -> str:
    return path.split(".")[0].split("[")[0]


def _match_citation(value: str, citations: list[Citation]) -> Citation | None:
    """Find the citation the model raised for this value.

    Exact match first; then a containment check, because a model will sometimes
    cite "124 degrees" for a field it filled with "124". Anything looser than
    that would start manufacturing the evidence we are trying to verify.
    """
    target = _normalise(value)
    for citation in citations:
        if _normalise(citation.value) == target:
            return citation
    for citation in citations:
        cited = _normalise(citation.value)
        if cited and (cited in target or target in cited):
            return citation
    return None


def ground(
    assessment: FirstAssessment,
    transcript: str,
    transcription: TranscriptionResult | None,
    citations: list[Citation],
) -> list[FieldEvidence]:
    """Build one `FieldEvidence` per populated field of the assessment."""
    haystack = _normalise(transcript)
    fields: list[FieldEvidence] = []

    for path, value in _leaves(assessment.model_dump()):
        if not isinstance(value, str) or not value.strip():
            continue  # an empty field claims nothing, so there is nothing to check

        citation = _match_citation(value, citations)
        if citation is None:
            fields.append(
                FieldEvidence(
                    field=path,
                    value=value,
                    evidenceFound=False,
                    reason="No source quoted for this value.",
                )
            )
            continue

        found = bool(citation.evidence.strip()) and _normalise(citation.evidence) in haystack
        # Only ask the audio how sure it was once we know the quote is real --
        # scoring a fabricated span would be meaningless.
        #
        # Score the whole quoted span, not just the value's own words. It is
        # tempting to score only the value -- a clearly-heard "124" should not
        # be dragged down by a mumbled "degrees" beside it -- but that reasoning
        # fails on the case this system exists for: the recording says "knee gig
        # 5 degrees" where "5" is heard perfectly and the ruined word before it
        # is almost certainly "negative". Score the number alone and a sign
        # error sails through. The span is what establishes the meaning, so the
        # span is what gets graded, and a noisy neighbour flags the field for
        # review. In a clinical setting that is the right direction to be wrong.
        audio = context = None
        if found and transcription:
            audio = transcription.confidence_for(citation.evidence)
            # Widen the view a few words either side. A model that obeys the
            # "quote the shortest span" instruction can quote straight past a
            # hole in the transcript, and a clean quote sitting next to a
            # destroyed word is exactly the case worth a human's attention.
            context = transcription.context_confidence(citation.evidence)

        if not found:
            reason = "Quoted evidence does not appear in the transcript."
        elif context is not None and context < GARBLED:
            reason = (
                f"The recording is badly unclear next to this value "
                f"({context:.0%} on a nearby word); the quote itself is clean, "
                "so its meaning may not be."
            )
        else:
            reason = ""

        fields.append(
            FieldEvidence(
                field=path,
                value=value,
                evidence=citation.evidence,
                evidenceFound=found,
                modelConfidence=max(0.0, min(1.0, citation.confidence)),
                audioConfidence=audio,
                contextConfidence=context,
                reason=reason,
            )
        )
    return fields


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ExtractionState)
    for group in GROUP_SPECS:
        graph.add_node(f"extract_{group}", _make_group_node(group))
        graph.add_edge(START, f"extract_{group}")  # fan out, all groups at once
        graph.add_edge(f"extract_{group}", "assemble")

    graph.add_node("assemble", assemble_node)
    graph.add_node("ground", ground_node)
    graph.add_node("repair", repair_node)

    graph.add_edge("assemble", "ground")
    graph.add_conditional_edges("ground", _route, {"repair": "repair", "done": END})
    graph.add_edge("repair", "assemble")
    return graph.compile()


class ExtractionResult(BaseModel):
    """What the pipeline hands to the API layer."""

    assessment: FirstAssessment
    flags: ExtractionFlags

    def failing(self) -> list[FieldEvidence]:
        """Fields below the configured bar -- the body of a 422."""
        return self.flags.below(get_settings().extraction_confidence_threshold)


def extract(
    transcript: str, transcription: TranscriptionResult | None = None
) -> ExtractionResult:
    """Run the agent over a transcript.

    `transcription` is optional but strongly recommended: without it there is no
    per-word audio confidence, so a value extracted perfectly from a misheard
    transcript looks indistinguishable from a good one.
    """
    if not transcript.strip():
        raise ExtractionFailed("Empty transcript.")

    final = build_graph().invoke(
        {
            "transcript": transcript,
            "transcription": transcription,
            "sections": {},
            "citations": [],
            "errors": [],
            "attempts": 0,
        }
    )
    assessment = final.get("assessment")
    if assessment is None:
        raise ExtractionFailed("; ".join(final.get("errors") or ["No assessment produced."]))
    return ExtractionResult(
        assessment=assessment, flags=final.get("flags") or ExtractionFlags()
    )
