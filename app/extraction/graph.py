"""The LangGraph extraction agent (D3).

Graph shape::

    extract_clinical_details
        -> extract_subjective
        -> extract_objective
        -> extract_goals
        -> extract_plan
        -> verify_grounding      (deterministic, no LLM)
        -> assemble              (into FirstAssessment)
        -> score_confidence      (deterministic, no LLM)

Two design points are worth stating explicitly.

**Section by section, not one call.** A 3B model holds a small flat schema
reliably and a seven-section nested one poorly. Splitting also contains
failure: if the objective measurements fail to parse, the rest of the
assessment still arrives, with that section flagged instead of the whole
request failing.

**The last three nodes use no LLM.** Grounding, assembly and scoring are pure
functions of the transcript and the extracted values. The guarantee that
nothing was invented therefore does not depend on the model cooperating - it
is enforced after the fact by code that can be tested directly.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings, get_settings
from app.extraction.confidence import ConfidenceReport, score
from app.extraction.grounding import GroundingIssue, ground_payload
from app.extraction.llm import StructuredOutputError, build_llm, structured_call
from app.extraction.prompts import SECTION_SPECS, SYSTEM_PROMPT, build_user_prompt
from app.schemas.first_assessment import FirstAssessment, empty_assessment

logger = logging.getLogger(__name__)


def _merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class ExtractionState(TypedDict, total=False):
    """State threaded through the graph."""

    transcript: str
    sections: Annotated[dict[str, Any], _merge]
    issues: list[GroundingIssue]
    assessment: FirstAssessment
    confidence: ConfidenceReport
    errors: Annotated[dict[str, str], _merge]
    timings: Annotated[dict[str, float], _merge]


class ExtractionResult(TypedDict):
    assessment: FirstAssessment
    confidence: ConfidenceReport
    issues: list[GroundingIssue]
    errors: dict[str, str]
    timings: dict[str, float]


def _make_extract_node(spec, llm, settings: Settings):
    """Build one extraction node for a section spec."""

    def node(state: ExtractionState) -> dict:
        started = time.perf_counter()
        try:
            result = structured_call(
                llm,
                system=SYSTEM_PROMPT,
                user=build_user_prompt(spec, state["transcript"]),
                model_cls=spec.model_cls,
                max_retries=settings.llm_max_retries,
            )
            payload = result.model_dump()
            error: dict[str, str] = {}
            logger.info("Extracted %s in %.1fs", spec.label, time.perf_counter() - started)
        except StructuredOutputError as exc:
            # Contain the failure: an empty section plus a recorded error beats
            # failing the whole assessment, and the section will be flagged.
            payload = spec.model_cls().model_dump()
            error = {spec.key: str(exc)}
            logger.warning("Section %r failed to extract: %s", spec.key, exc)

        return {
            "sections": {spec.key: payload},
            "errors": error,
            "timings": {spec.key: round(time.perf_counter() - started, 2)},
        }

    return node


def _verify_grounding(state: ExtractionState) -> dict:
    """Clear every value that cannot be traced to the transcript."""
    started = time.perf_counter()
    transcript = state["transcript"]

    cleaned: dict[str, Any] = {}
    issues: list[GroundingIssue] = []
    for key, payload in state.get("sections", {}).items():
        section, section_issues = ground_payload(payload, transcript)
        cleaned[key] = section
        issues.extend(section_issues)

    if issues:
        logger.info(
            "Grounding rejected %d value(s): %s",
            len(issues),
            ", ".join(issue.path for issue in issues),
        )

    return {
        "sections": cleaned,
        "issues": issues,
        "timings": {"grounding": round(time.perf_counter() - started, 2)},
    }


#: Findings a small model tends to emit as one blob. Matching an extracted
#: finding against this controlled vocabulary is *classification of text the
#: model already produced* - no clinical content is introduced, and an
#: unmatched finding keeps an empty name rather than being given a guessed one.
#: Order is load-bearing: first match wins, so the most specific signal has to
#: come first. A finding that reads "healed surgical scar ... painful flexion"
#: is about the scar, and one that reads "knee pain ... during walking" is
#: about pain, not gait - matching anatomy first mislabels both.
_FINDING_LABELS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("scar", "incision", "wound"), "Surgical scar"),
    (("patellar", "patella"), "Patellar mobility"),
    (("pain", "ache", "irritability", "tender"), "Pain"),
    (("swelling", "oedema", "edema"), "Swelling"),
    (("gait", "walking", "limp"), "Gait"),
    (("dorsiflex", "plantarflex", "ankle"), "Ankle range"),
    (("hip",), "Hip range"),
    (("flexion", "extension", "range of motion"), "Knee range"),
)


def _is_narrative(text: str) -> bool:
    """A label is a couple of words; anything sentence-shaped is a finding.

    Length alone was too blunt - "Patellar mobility was good." is short but is
    plainly a finding, not the name of a test.
    """
    return len(text.split()) > 3 or any(mark in text for mark in ".,;")


def _label_for(text: str, taken: set[str] | None = None) -> str:
    """First matching label, skipping ones already used on this assessment.

    Two findings both mentioning pain would otherwise both come back "Pain".
    Falling through to the next match distinguishes them - a finding about
    restricted flexion becomes "Knee range" rather than a second "Pain".
    """
    lowered = text.lower()
    fallback = ""
    for needles, label in _FINDING_LABELS:
        if any(needle in lowered for needle in needles):
            if taken is None or label not in taken:
                return label
            fallback = fallback or label
    return fallback


def _normalise_subjective(items: list) -> list:
    """Put the finding in `conclusion` and a short label in `testName`.

    The model reliably crams the whole finding into testName and leaves
    conclusion empty, which renders as a paragraph-length field label beside
    the word "not stated". Moving the text is a rearrangement of what was
    already extracted, not new content.
    """
    out: list[dict] = []
    taken: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("testName") or "").strip()
        conclusion = str(item.get("conclusion") or "").strip()

        if not conclusion and _is_narrative(name):
            # The whole finding landed in the label. Move it, and derive a name.
            conclusion, name = name, _label_for(name, taken)
        elif _is_narrative(name):
            # Both fields are filled but the label is a sentence. Relabel it,
            # keeping the original only when the vocabulary has no match, since
            # a long label still beats losing the text.
            name = _label_for(name, taken) or name
        elif not name and conclusion:
            name = _label_for(conclusion, taken)

        if name or conclusion:
            taken.add(name)
            out.append({"testName": name, "conclusion": conclusion})
    return out


def _merge_sided_tests(tests: list) -> list:
    """Collapse "Left knee flexion" and "Right knee flexion" into one row.

    The model emits a row per side, each carrying one number and an empty
    opposite side, so a bilateral measurement arrives as two half-blank rows.
    The side is stated in the row's own name, so merging recovers a complete
    measurement without inventing anything.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []

    for test in tests:
        name = str(test.get("testName") or "").strip()
        lowered = name.lower()

        side = ""
        base = name
        for prefix, which in (("left ", "left"), ("right ", "right")):
            if lowered.startswith(prefix):
                side, base = which, name[len(prefix):].strip()
                break

        key = base.lower() or lowered
        if key not in merged:
            merged[key] = {
                "testName": base[:1].upper() + base[1:] if base else name,
                "unitName": "", "value": "", "left": "", "right": "", "comments": "",
            }
            order.append(key)
        row = merged[key]

        left = _strip_unit(str(test.get("left") or ""))
        right = _strip_unit(str(test.get("right") or ""))
        value = _strip_unit(str(test.get("value") or ""))

        if side == "left":
            # A row named "Left ..." may still carry both sides.
            row["left"] = row["left"] or left or value
            row["right"] = row["right"] or right
        elif side == "right":
            row["right"] = row["right"] or right or left or value
            row["left"] = row["left"] or (left if right else "")
        else:
            row["left"] = row["left"] or left
            row["right"] = row["right"] or right
            if not (left or right):
                row["value"] = row["value"] or value

        row["unitName"] = row["unitName"] or str(test.get("unitName") or "").strip()
        row["comments"] = row["comments"] or str(test.get("comments") or "").strip()

    return [merged[key] for key in order]


def _split_goals(objective: list, subjective: list) -> tuple[list, list]:
    """Move goals with no measurable target into subjectiveGoals.

    An objective goal is one with a number to hit. "Improving ankle mobility"
    has none, and filing it as an objective goal leaves goalCategory, unitName
    and value permanently blank - three empty fields per goal that no recording
    could ever fill. As a subjective goal it is complete.
    """
    kept, moved = [], list(subjective)

    for goal in objective:
        if not isinstance(goal, dict):
            continue
        measurable = str(goal.get("value") or "").strip() or str(goal.get("unitName") or "").strip()
        name = str(goal.get("goalName") or "").strip()

        if measurable:
            kept.append(goal)
        elif name:
            moved.append({"goalDetails": name, "targetDate": goal.get("targetDate") or ""})

    return kept, moved


_MEASUREMENT_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_unit(value: str) -> str:
    """Keep the number, drop the unit that rides along with it.

    unitName already carries the unit, so "124 degrees" in `left` duplicates
    it and makes the field harder for a consumer to use - it has to be parsed
    rather than read. A value with no number at all is left untouched.
    """
    text = (value or "").strip()
    if not text:
        return ""
    match = _MEASUREMENT_NUMBER.search(text)
    return match.group(0) if match else text


def _normalise_tests(tests: list) -> list:
    """Make objective measurements internally consistent.

    ``value`` is for a measurement with no side, while ``left``/``right`` carry
    a sided one. A small model routinely fills both, producing entries where
    value merely duplicates left. Rather than prompt-tune a 3B model into
    compliance, the rule is enforced here where it always holds: if either side
    is present, value is cleared.

    Entries with no measurement at all are dropped - an objective test with no
    number is noise on a clinician's screen, and confidence still records that
    nothing was captured.
    """
    normalised = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        entry = dict(test)
        left = str(entry.get("left") or "").strip()
        right = str(entry.get("right") or "").strip()
        value = str(entry.get("value") or "").strip()

        if left or right:
            entry["value"] = ""
        else:
            entry["value"] = value

        has_measurement = bool(left or right or entry["value"])
        has_name = bool(str(entry.get("testName") or "").strip())
        if has_measurement or has_name:
            normalised.append(entry)

    return normalised


def _assemble(state: ExtractionState) -> dict:
    """Map the verified sections onto the strict FirstAssessment schema (S4)."""
    started = time.perf_counter()
    sections = state.get("sections", {})

    clinical = sections.get("clinicalDetails", {})
    subjective = sections.get("subjective", {})
    objective = sections.get("objective", {})
    goals = sections.get("goals", {})
    plan = sections.get("plan", {})

    objective_goals, subjective_goals = _split_goals(
        goals.get("objectiveGoals", []) or [], goals.get("subjectiveGoals", []) or []
    )

    payload = {
        "clinicalDetails": clinical,
        "subjectiveAssessments": _normalise_subjective(
            subjective.get("subjectiveAssessments", []) or []
        ),
        "objectiveAssessment": {
            "tests": _merge_sided_tests(_normalise_tests(objective.get("tests", []) or []))
        },
        "subjectiveGoals": subjective_goals,
        "objectiveGoals": objective_goals,
        "recommendation": plan.get("recommendation", []),
        "patientAdvice": plan.get("patientAdvice", {}),
    }

    try:
        assessment = FirstAssessment.model_validate(payload)
    except Exception as exc:
        # The schema is the contract with the frontend, so a malformed
        # assembly returns a valid empty assessment rather than a broken one.
        logger.error("Assembly failed, returning an empty assessment: %s", exc)
        return {
            "assessment": empty_assessment(),
            "errors": {"assemble": str(exc)},
            "timings": {"assemble": round(time.perf_counter() - started, 2)},
        }

    return {
        "assessment": assessment,
        "timings": {"assemble": round(time.perf_counter() - started, 2)},
    }


def _score_confidence(state: ExtractionState, settings: Settings) -> dict:
    started = time.perf_counter()
    report = score(
        state["assessment"],
        state.get("issues", []),
        threshold=settings.confidence_threshold,
        transcript=state.get("transcript", ""),
    )
    logger.info(
        "Confidence %.2f (threshold %.2f), %d flagged, %d rejected",
        report.overall, report.threshold, len(report.flaggedFields), report.rejectedCount,
    )
    return {
        "confidence": report,
        "timings": {"confidence": round(time.perf_counter() - started, 2)},
    }


def build_graph(llm=None, settings: Settings | None = None):
    """Compile the extraction graph.

    ``llm`` is injectable so tests can drive the whole graph with a stub and
    no network, model download, or GPU.
    """
    settings = settings or get_settings()
    llm = llm if llm is not None else build_llm(settings)

    graph = StateGraph(ExtractionState)

    for spec in SECTION_SPECS:
        graph.add_node(f"extract_{spec.key}", _make_extract_node(spec, llm, settings))

    graph.add_node("verify_grounding", _verify_grounding)
    graph.add_node("assemble", _assemble)
    graph.add_node("score_confidence", lambda state: _score_confidence(state, settings))

    entry = f"extract_{SECTION_SPECS[0].key}"
    graph.set_entry_point(entry)

    # Sequential rather than parallel: the extraction model is a single
    # GPU-resident process, so fanning out would queue at the daemon anyway
    # while making failures harder to attribute to a section.
    for current, following in zip(SECTION_SPECS, SECTION_SPECS[1:]):
        graph.add_edge(f"extract_{current.key}", f"extract_{following.key}")

    graph.add_edge(f"extract_{SECTION_SPECS[-1].key}", "verify_grounding")
    graph.add_edge("verify_grounding", "assemble")
    graph.add_edge("assemble", "score_confidence")
    graph.add_edge("score_confidence", END)

    return graph.compile()


def extract_assessment(
    transcript: str, *, llm=None, settings: Settings | None = None
) -> ExtractionResult:
    """Run the agent over a transcript and return the assessment plus metadata."""
    settings = settings or get_settings()
    compiled = build_graph(llm=llm, settings=settings)

    started = time.perf_counter()
    final = compiled.invoke(
        {"transcript": transcript, "sections": {}, "errors": {}, "timings": {}}
    )
    total = round(time.perf_counter() - started, 2)

    timings = dict(final.get("timings", {}))
    timings["total"] = total

    return {
        "assessment": final["assessment"],
        "confidence": final["confidence"],
        "issues": final.get("issues", []),
        "errors": final.get("errors", {}),
        "timings": timings,
    }
