"""LangGraph extraction agent: transcript -> FirstAssessment + per-field evidence.

    START ─┬─> subjective ─┐
           ├─> objective ──┼─> assemble ─> ground ─> route ─> END
           └─> plan ───────┘        ^                 │
                                    └──── repair <────┘

**Three nodes, not one call and not seven.** A single call producing all seven
sections at once gets measurably lazier about the later ones: Gemini degrades
as an output schema deepens. The other extreme, one call per section, was what
this started as, and it broke on contact with reality: the Gemini free tier
allows five requests per minute and a few dozen per day, so a seven-call fan-out
429s on arrival and burns a day's quota in three runs. Three nodes grouped by
what a clinician actually reasons about together (the subjective picture, the
measurements, the plan) keeps each call focused, fits inside the free tier, and
leaves headroom for the repair pass.

**Every value must be quoted.** A node returns its data *and* a list of
`Citation`s: the verbatim transcript span each value came from. We then check
those quotes against the transcript **in our own code**. This is the difference
between asking a model to be honest and being able to tell whether it was: a
value whose quote is not in the transcript was invented, full stop, and it
scores zero no matter how confident the model claims to be.

**We compute the field paths, the model does not.** Citations carry a value and
a quote; the dotted path (``objectiveAssessment.tests[0].value``) is derived by
walking the assembled document ourselves. Asking an LLM to emit correct array
indices is a needless way to lose data.

**Citations stay with the group that produced them.** They are kept per group
rather than pooled, so a field is only ever checked against the quotes from the
call that filled it. Values repeat across sections (``unitName`` is "degrees" in
both ``objectiveAssessment.tests`` and ``objectiveGoals``, and those come from
different calls), so a single pooled list lets one group's quote decide the fate
of another group's field. Keeping them separate is also what lets a repair
*replace* a group's quotes instead of adding to them.

**Repair targets hallucination specifically.** When grounding finds a quote that
is not in the transcript, the repair pass re-asks *only the affected group*,
naming the bad quote and instructing the model to either quote correctly or
leave the field empty. Leaving it empty is an acceptable answer here; guessing
is not.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping, Sequence
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
    _bare,
)

logger = logging.getLogger(__name__)


class ExtractionFailed(RuntimeError):
    """Raised when the agent cannot produce anything usable at all."""


class ExtractionUnavailable(ExtractionFailed):
    """Every group call failed. There is no partial result worth reviewing.

    Separate from `ExtractionFailed` so the API can answer 502 rather than 422:
    nothing was wrong with the request, the model provider was simply not
    answering, and "try again" is the useful advice rather than "fix your input".
    """


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

Citations are mandatory. For EVERY non-empty field you fill, including names,
labels and categories you chose yourself such as testName or goalCategory, add
one entry to `citations` containing the value you wrote and the verbatim
transcript span it came from.

- For a value taken from the transcript, quote the span that states it.
- For a label you chose yourself, quote the span that the label describes. For
  example, a testName of "Pain characteristics" cites the span describing the
  pain, because nobody said the words "pain characteristics" aloud.

Copy the span character for character from the transcript. Do not stitch words
together from different parts of a sentence, and do not tidy the wording: the
span is checked against the transcript automatically, and a field whose span
cannot be found there is discarded as unsupported.

Quote the shortest span that does both of these things: establishes the value,
and matches one place in the transcript and no other. Never answer with the
value by itself. A bare number, or a bare word such as "degrees", occurs in
many places at once and so points at none of them. Add only the few neighbouring
words needed to make the span unique, such as the body part, the side, the unit
or a sign, and nothing beyond that.

Good: "knee flexion of 124 degrees"
Good: "knee gig 5 degrees on the right"
Bad:  "124"
Bad:  "degrees"
"""

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
        "`objectiveGoals`: goals with a measurable target: goalName, "
        "goalCategory (e.g. 'Range of motion', 'Strength'), unitName and target "
        "value. A goal belongs in exactly one of these two lists, never both. "
        "targetDate only if a date or deadline was actually spoken.\n"
        "`recommendation`: what kind of sessions were recommended (sessionType) "
        "and how often (sessionFrequency).\n"
        "`patientAdvice`: advice for the patient to follow themselves: "
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


def _run_group(group: str, transcript: str, hint: str = "") -> tuple[BaseModel | None, str]:
    """Ask the model for one group of sections.

    Returns the parsed result and an empty string, or ``None`` and the reason it
    failed. The reason is carried rather than only logged because a section left
    empty by a failed call has to stay distinguishable from one left empty
    because the clinician never mentioned it. Those look identical in the
    finished document, and only one of them is trustworthy.
    """
    response_model, sections, description = GROUP_SPECS[group]
    prompt = (
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"Extract these sections: {', '.join(sections)}.\n\n{description}\n"
    )
    if hint:
        prompt += f"\n{hint}\n"
    try:
        model = _llm().with_structured_output(response_model)
        result = model.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
        return result, ""
    except Exception as exc:  # noqa: BLE001 (one bad group must not sink the run)
        reason = str(exc).strip()[:200] or exc.__class__.__name__
        logger.warning("Group %s failed: %s", group, reason)
        return None, reason


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
    # group -> the quotes that group gave us. Keyed by group and merged rather
    # than appended, so a repair replaces a group's quotes outright. Appending
    # them left the rejected quote in the list, and since matching takes the
    # first quote that fits, a corrected one could never win.
    citations: Annotated[dict[str, list[Citation]], _merge]
    # group -> why its call failed, or "" once it has succeeded. A dict for the
    # same reason: a repair can clear an earlier failure, and an append-only
    # reducer has no way to take one back.
    failures: Annotated[dict[str, str], _merge]
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
        result, reason = _run_group(group, state["transcript"])
        if result is None:
            # Recorded only in `failures`, which a repair can clear. Writing it
            # to the append-only `errors` list as well meant a group that failed
            # once and succeeded on the retry still carried "could not be
            # extracted" into the saved record, contradicting the very section
            # sitting beside it.
            return {"failures": {group: reason}}
        return {
            "sections": _sections_from(group, result),
            "citations": {group: list(result.citations)},
            "failures": {group: ""},
        }

    node.__name__ = f"extract_{group}"
    return node


def assemble_node(state: ExtractionState) -> dict[str, Any]:
    """Merge the groups into one contract document.

    Any section that failed simply stays at its schema default, an empty string
    or an empty array, which is the honest representation of "we do not know"
    and keeps the document valid rather than half-formed.
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
        state.get("citations") or {},
    )
    failures = {g: r for g, r in (state.get("failures") or {}).items() if r}
    failed_sections = sorted(
        section for section, group in SECTION_TO_GROUP.items() if group in failures
    )

    # A field inside a failed section is not "unresolved". That word means the
    # transcript did not support it, which is a statement about the recording we
    # are in no position to make when the call never returned.
    unresolved = [
        path
        for path, value in _leaves(assessment.model_dump())
        if not str(value).strip() and _section_of(path) not in failed_sections
    ]

    # Built from the current `failures`, never from an accumulated log: a
    # warning describes the document being returned, and a group that has since
    # recovered has nothing left to warn about.
    warnings = list(state.get("errors") or [])
    warnings += [
        f"Group {group} could not be extracted: {reason}"
        for group, reason in sorted(failures.items())
    ]
    if failed_sections:
        warnings.append(
            f"{', '.join(failed_sections)} could not be extracted after retries. "
            "These sections are empty because the model call for them failed, "
            "not because the recording was silent about them."
        )

    return {
        "flags": ExtractionFlags.summarise(
            fields,
            unresolved=unresolved,
            warnings=warnings,
            failed_sections=failed_sections,
        )
    }


def _repair_hint(offenders: list[FieldEvidence]) -> str:
    """Tell the model exactly which of its quotes did not check out.

    Empty when there are no offenders, which is the case for a group whose call
    never returned at all: there is nothing to correct, so it is simply asked
    again rather than accused of something it did not do.
    """
    if not offenders:
        return ""
    detail = "\n".join(
        f'- Field `{f.field}`: you gave "{f.value}" citing '
        f'"{f.evidence or "(no quote at all)"}", which does not appear in the transcript.'
        for f in offenders
    )
    return (
        "Your previous attempt at this section was rejected:\n"
        f"{detail}\n"
        "Quote the transcript exactly, or leave the field empty. Leaving it "
        "empty is the correct answer when the transcript does not say it. "
        "Do not substitute a different guess."
    )


def repair_node(state: ExtractionState) -> dict[str, Any]:
    """Re-ask the groups that produced unverifiable quotes, or nothing at all."""
    flags = state.get("flags") or ExtractionFlags()
    by_group: dict[str, list[FieldEvidence]] = {}
    for field in flags.ungrounded():
        group = SECTION_TO_GROUP.get(_section_of(field.field))
        if group:
            by_group.setdefault(group, []).append(field)

    # A group whose call failed outright contributed no fields, so it has no
    # ungrounded ones either and the loop above cannot see it. Without this a
    # transient 503 costs a whole section permanently, while a single badly
    # quoted value gets two more chances. Exactly backwards.
    for group, reason in (state.get("failures") or {}).items():
        if reason:
            by_group.setdefault(group, [])

    updates: dict[str, Any] = {}
    citations: dict[str, list[Citation]] = {}
    failures: dict[str, str] = {}
    for group, offenders in by_group.items():
        result, reason = _run_group(group, state["transcript"], hint=_repair_hint(offenders))
        if result is None:
            failures[group] = reason
            continue
        updates.update(_sections_from(group, result))
        # Replaces, not extends. The rejected quotes are exactly what this call
        # was made to get rid of.
        citations[group] = list(result.citations)
        failures[group] = ""

    return {
        "sections": updates,
        "citations": citations,
        "failures": failures,
        "attempts": state.get("attempts", 0) + 1,
    }


def _route(state: ExtractionState) -> str:
    """Repair only while something is ungrounded and retries remain."""
    settings = get_settings()
    flags = state.get("flags") or ExtractionFlags()
    stalled = [group for group, reason in (state.get("failures") or {}).items() if reason]

    if not flags.ungrounded() and not stalled:
        return "done"
    if state.get("attempts", 0) >= settings.extraction_max_retries:
        logger.info(
            "Out of repair attempts; %d field(s) ungrounded, %d group(s) still failing.",
            len(flags.ungrounded()),
            len(stalled),
        )
        return "done"
    return "repair"


# --------------------------------------------------------------------------- #
# Grounding (plain code, no model involved, which is the point)
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


def _tokens(text: str) -> list[str]:
    """Lowercased words with the punctuation stripped off each one."""
    return [token for token in (_bare(w) for w in _normalise(text).split()) if token]


def _runs_through(haystack: list[str], needle: list[str]) -> bool:
    """Is `needle` a run of whole words inside `haystack`?"""
    size = len(needle)
    return bool(needle) and any(
        haystack[i : i + size] == needle for i in range(len(haystack) - size + 1)
    )


def _match_citation(value: str, citations: Sequence[Citation]) -> Citation | None:
    """Find the citation the model raised for this value.

    Exact match first, then a whole-word containment check, because a model will
    sometimes cite "124 degrees" for a field it filled with "124".

    Containment is measured in words rather than characters. On raw substrings
    "5" is inside "15", so the citation for one measurement would vouch for a
    different one: a wrong number, carrying a quote that really is in the
    transcript, scoring high enough to be returned without review. That is the
    precise failure this module exists to prevent, so the loose end that allowed
    it is closed rather than merely narrowed.

    Where several quotes fit, the longest wins. It is the most specific claim on
    the value, whereas taking the first that fits makes the answer depend on the
    order the model happened to list its citations in.
    """
    target = _normalise(value)
    for citation in citations:
        if _normalise(citation.value) == target:
            return citation

    target_words = _tokens(value)
    best: Citation | None = None
    for citation in citations:
        cited_words = _tokens(citation.value)
        if _runs_through(target_words, cited_words) or _runs_through(
            cited_words, target_words
        ):
            if best is None or len(cited_words) > len(_tokens(best.value)):
                best = citation
    return best


def ground(
    assessment: FirstAssessment,
    transcript: str,
    transcription: TranscriptionResult | None,
    citations: Mapping[str, Sequence[Citation]],
) -> list[FieldEvidence]:
    """Build one `FieldEvidence` per populated field of the assessment.

    `citations` is keyed by group, and a field is checked only against the
    quotes from the group that filled its section. A quote from another call is
    not weaker evidence for this field, it is evidence about a different one.
    """
    haystack = _normalise(transcript)
    fields: list[FieldEvidence] = []

    for path, value in _leaves(assessment.model_dump()):
        if not isinstance(value, str) or not value.strip():
            continue  # an empty field claims nothing, so there is nothing to check

        group = SECTION_TO_GROUP.get(_section_of(path))
        citation = _match_citation(value, citations.get(group, ()) if group else ())
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
        # Only ask the audio how sure it was once we know the quote is real:
        # scoring a fabricated span would be meaningless.
        #
        # Score the whole quoted span, not just the value's own words. It is
        # tempting to score only the value, since a clearly-heard "124" should
        # not be dragged down by a mumbled "degrees" beside it, but that reasoning
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
        """Fields below the configured bar: the body of a 422."""
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
            "citations": {},
            "errors": [],
            "failures": {},
            "attempts": 0,
        }
    )
    assessment = final.get("assessment")

    # Every group failing is not a partial result, it is no result. Returning an
    # empty seven-section document would be indistinguishable from a recording
    # in which the clinician said nothing at all, and the caller would have no
    # way to tell that the provider, rather than the audio, was the problem.
    stalled = {g: r for g, r in (final.get("failures") or {}).items() if r}
    if stalled and len(stalled) == len(GROUP_SPECS):
        raise ExtractionUnavailable(
            "No section could be extracted; every model call failed. "
            + "; ".join(f"{group}: {reason}" for group, reason in sorted(stalled.items()))
        )

    if assessment is None:
        raise ExtractionFailed("; ".join(final.get("errors") or ["No assessment produced."]))
    return ExtractionResult(
        assessment=assessment, flags=final.get("flags") or ExtractionFlags()
    )
