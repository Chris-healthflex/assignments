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
    """Merge two dictionaries used by LangGraph state reducers."""
    return {**left, **right}


class ExtractionState(TypedDict, total=False):
    """State threaded through the extraction graph."""

    transcript: str

    sections: Annotated[
        dict[str, Any],
        _merge,
    ]

    issues: list[GroundingIssue]

    assessment: FirstAssessment

    confidence: ConfidenceReport

    errors: Annotated[
        dict[str, str],
        _merge,
    ]

    timings: Annotated[
        dict[str, float],
        _merge,
    ]


class ExtractionResult(TypedDict):
    """Public result returned by extract_assessment()."""

    assessment: FirstAssessment
    confidence: ConfidenceReport
    issues: list[GroundingIssue]
    errors: dict[str, str]
    timings: dict[str, float]


def _make_extract_node(spec, llm, settings: Settings):
    """Build one extraction node for a section specification."""

    def node(state: ExtractionState) -> dict:
        started = time.perf_counter()

        try:
            result = structured_call(
                llm,
                system=SYSTEM_PROMPT,
                user=build_user_prompt(
                    spec,
                    state["transcript"],
                ),
                model_cls=spec.model_cls,
                max_retries=settings.llm_max_retries,
            )

            payload = result.model_dump()

            errors: dict[str, str] = {}

            logger.info(
                "Extracted %s in %.1fs",
                spec.label,
                time.perf_counter() - started,
            )

        except StructuredOutputError as exc:
            # Failure containment:
            # a failed section should not destroy the whole assessment.
            #
            # Return an empty section and preserve the error so the caller
            # knows exactly which extraction section failed.
            payload = spec.model_cls().model_dump()

            errors = {
                spec.key: str(exc),
            }

            logger.warning(
                "Section %r failed to extract: %s",
                spec.key,
                exc,
            )

        return {
            "sections": {
                spec.key: payload,
            },
            "errors": errors,
            "timings": {
                spec.key: round(
                    time.perf_counter() - started,
                    2,
                )
            },
        }

    return node


def _verify_grounding(state: ExtractionState) -> dict:
    """Clear every value that cannot be traced to the transcript."""

    started = time.perf_counter()

    transcript = state["transcript"]

    cleaned: dict[str, Any] = {}

    issues: list[GroundingIssue] = []

    for key, payload in state.get("sections", {}).items():
        section, section_issues = ground_payload(
            payload,
            transcript,
        )

        cleaned[key] = section
        issues.extend(section_issues)

    if issues:
        logger.info(
            "Grounding rejected %d value(s): %s",
            len(issues),
            ", ".join(
                issue.path
                for issue in issues
            ),
        )

    return {
        "sections": cleaned,
        "issues": issues,
        "timings": {
            "grounding": round(
                time.perf_counter() - started,
                2,
            )
        },
    }


def _normalise_tests(tests: list) -> list:
    """Make objective measurements internally consistent.

    ``value`` is used for a measurement with no side, while ``left`` and
    ``right`` carry sided measurements.

    If either side is present, ``value`` is always cleared.

    Entries with no measurement but with a test name are preserved so that
    the extraction result does not silently lose a named finding.
    """

    normalised = []

    for test in tests:
        if not isinstance(test, dict):
            continue

        entry = dict(test)

        left = str(
            entry.get("left") or ""
        ).strip()

        right = str(
            entry.get("right") or ""
        ).strip()

        value = str(
            entry.get("value") or ""
        ).strip()

        # Sided measurement.
        if left or right:
            entry["value"] = ""

        # Non-sided measurement.
        else:
            entry["value"] = value

        has_measurement = bool(
            left
            or right
            or entry["value"]
        )

        has_name = bool(
            str(
                entry.get("testName") or ""
            ).strip()
        )

        # Keep measurements and named tests.
        if has_measurement or has_name:
            normalised.append(entry)

    return normalised


def _assemble(state: ExtractionState) -> dict:
    """Map verified sections onto the strict FirstAssessment schema."""

    started = time.perf_counter()

    sections = state.get(
        "sections",
        {},
    )

    clinical = sections.get(
        "clinicalDetails",
        {},
    )

    subjective = sections.get(
        "subjective",
        {},
    )

    objective = sections.get(
        "objective",
        {},
    )

    goals = sections.get(
        "goals",
        {},
    )

    plan = sections.get(
        "plan",
        {},
    )

    payload = {
        "clinicalDetails": clinical,

        "subjectiveAssessments": subjective.get(
            "subjectiveAssessments",
            [],
        ),

        "objectiveAssessment": {
            "tests": _normalise_tests(
                objective.get(
                    "tests",
                    [],
                )
            )
        },

        "subjectiveGoals": goals.get(
            "subjectiveGoals",
            [],
        ),

        "objectiveGoals": goals.get(
            "objectiveGoals",
            [],
        ),

        "recommendation": plan.get(
            "recommendation",
            [],
        ),

        "patientAdvice": plan.get(
            "patientAdvice",
            {},
        ),
    }

    try:
        assessment = FirstAssessment.model_validate(
            payload
        )

    except Exception as exc:
        # The schema is the contract with the frontend.
        # If assembly somehow produces an invalid object, return a valid
        # empty assessment rather than breaking the whole pipeline.

        logger.error(
            "Assembly failed, returning an empty assessment: %s",
            exc,
        )

        return {
            "assessment": empty_assessment(),
            "errors": {
                "assemble": str(exc),
            },
            "timings": {
                "assemble": round(
                    time.perf_counter() - started,
                    2,
                )
            },
        }

    return {
        "assessment": assessment,
        "timings": {
            "assemble": round(
                time.perf_counter() - started,
                2,
            )
        },
    }


def _score_confidence(
    state: ExtractionState,
    settings: Settings,
) -> dict:
    """Calculate deterministic confidence after extraction."""

    started = time.perf_counter()

    report = score(
        state["assessment"],
        state.get(
            "issues",
            [],
        ),
        threshold=settings.confidence_threshold,
    )

    logger.info(
        "Confidence %.2f (threshold %.2f), %d flagged, %d rejected",
        report.overall,
        report.threshold,
        len(report.flaggedFields),
        report.rejectedCount,
    )

    return {
        "confidence": report,
        "timings": {
            "confidence": round(
                time.perf_counter() - started,
                2,
            )
        },
    }


def build_graph(
    llm=None,
    settings: Settings | None = None,
):
    """Compile the extraction graph.

    ``llm`` is injectable so tests can drive the whole graph with a stub
    without requiring Ollama, a GPU, or a model download.
    """

    settings = settings or get_settings()

    llm = (
        llm
        if llm is not None
        else build_llm(settings)
    )

    graph = StateGraph(
        ExtractionState
    )

    # ---------------------------------------------------------------
    # Extraction nodes
    # ---------------------------------------------------------------

    for spec in SECTION_SPECS:
        graph.add_node(
            f"extract_{spec.key}",
            _make_extract_node(
                spec,
                llm,
                settings,
            ),
        )

    # ---------------------------------------------------------------
    # Deterministic nodes
    # ---------------------------------------------------------------

    graph.add_node(
        "verify_grounding",
        _verify_grounding,
    )

    graph.add_node(
        "assemble",
        _assemble,
    )

    graph.add_node(
        "score_confidence",
        lambda state: _score_confidence(
            state,
            settings,
        ),
    )

    # ---------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------

    entry = (
        f"extract_{SECTION_SPECS[0].key}"
    )

    graph.set_entry_point(entry)

    # ---------------------------------------------------------------
    # Sequential extraction
    # ---------------------------------------------------------------

    for current, following in zip(
        SECTION_SPECS,
        SECTION_SPECS[1:],
    ):
        graph.add_edge(
            f"extract_{current.key}",
            f"extract_{following.key}",
        )

    # ---------------------------------------------------------------
    # Final deterministic pipeline
    # ---------------------------------------------------------------

    graph.add_edge(
        f"extract_{SECTION_SPECS[-1].key}",
        "verify_grounding",
    )

    graph.add_edge(
        "verify_grounding",
        "assemble",
    )

    graph.add_edge(
        "assemble",
        "score_confidence",
    )

    graph.add_edge(
        "score_confidence",
        END,
    )

    return graph.compile()


def extract_assessment(
    transcript: str,
    *,
    llm=None,
    settings: Settings | None = None,
) -> ExtractionResult:
    """Run the extraction graph over a transcript."""

    settings = settings or get_settings()

    compiled = build_graph(
        llm=llm,
        settings=settings,
    )

    started = time.perf_counter()

    final = compiled.invoke(
        {
            "transcript": transcript,
            "sections": {},
            "errors": {},
            "timings": {},
        }
    )

    total = round(
        time.perf_counter() - started,
        2,
    )

    timings = dict(
        final.get(
            "timings",
            {},
        )
    )

    timings["total"] = total

    return {
        "assessment": final["assessment"],
        "confidence": final["confidence"],
        "issues": final.get(
            "issues",
            [],
        ),
        "errors": final.get(
            "errors",
            {},
        ),
        "timings": timings,
    }
