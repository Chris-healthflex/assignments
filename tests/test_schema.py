"""Contract tests for FirstAssessment.

These are the tests that make "exact match" checkable rather than hopeful, and
they must pass with no Mongo, no Whisper and no API key. The brief's three
hard rules each get their own group below:

* no extra fields / no renamed keys
* arrays stay arrays, even with one item
* string fields are strings, never null
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas import (
    SECTIONS,
    ExtractionFlags,
    FieldEvidence,
    FirstAssessment,
    StoredAssessment,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)

# A hand-written document in the exact shape the frontend consumes. If the
# models drift, this fixture is what notices first.
SAMPLE: dict = {
    "clinicalDetails": {
        "clinicalHistory": "Desk-based role, no prior back injury, no red flags reported.",
        "chiefComplaint": "Lower back pain radiating into the left leg.",
        "duration": "About three weeks",
    },
    "subjectiveAssessments": [
        {"testName": "Pain (NPRS)", "conclusion": "6 out of 10 at worst, worse in the mornings."},
        {"testName": "Straight leg raise", "conclusion": "Reproduces left-sided symptoms."},
    ],
    "objectiveAssessment": {
        "tests": [
            {
                "testName": "Lumbar flexion",
                "unitName": "degrees",
                "value": "40",
                "left": "",
                "right": "",
                "comments": "Limited by pain at end range.",
            },
            {
                "testName": "Hip abduction strength",
                "unitName": "MMT grade",
                "value": "",
                "left": "4",
                "right": "5",
                "comments": "Left weaker than right.",
            },
        ]
    },
    "subjectiveGoals": [
        {"goalDetails": "Sit through a full workday without pain.", "targetDate": "in about six weeks"}
    ],
    "objectiveGoals": [
        {
            "goalName": "Restore lumbar flexion",
            "goalCategory": "Range of motion",
            "unitName": "degrees",
            "value": "60",
            "targetDate": "2026-10-01",
        }
    ],
    "recommendation": [{"sessionType": "Outpatient physiotherapy", "sessionFrequency": "Twice weekly"}],
    "patientAdvice": {"adviceDetails": "Avoid prolonged sitting; walk for five minutes every hour."},
}


def _walk(node, path=""):
    """Yield every (path, value) leaf in a nested dump."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")
    else:
        yield path, node


# --------------------------------------------------------------------------- #
# Exact match: keys, nesting, ordering
# --------------------------------------------------------------------------- #
def test_top_level_sections_are_exactly_the_seven():
    """No missing sections, and just as important, no extra ones."""
    assert tuple(FirstAssessment.model_fields) == SECTIONS
    assert tuple(FirstAssessment().model_dump()) == SECTIONS


def test_sample_round_trips_byte_for_byte():
    """The canonical document survives validate -> dump unchanged."""
    assert FirstAssessment.model_validate(SAMPLE).model_dump() == SAMPLE


def test_json_round_trip_is_stable():
    parsed = FirstAssessment.model_validate(SAMPLE)
    assert FirstAssessment.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("clinicalDetails", {"clinicalHistory", "chiefComplaint", "duration"}),
        ("subjectiveAssessments", {"testName", "conclusion"}),
        ("objectiveAssessment.tests", {"testName", "unitName", "value", "left", "right", "comments"}),
        ("subjectiveGoals", {"goalDetails", "targetDate"}),
        ("objectiveGoals", {"goalName", "goalCategory", "unitName", "value", "targetDate"}),
        ("recommendation", {"sessionType", "sessionFrequency"}),
        ("patientAdvice", {"adviceDetails"}),
    ],
)
def test_section_keys_match_the_brief(path, expected):
    """Walk each section of a fully-populated dump and check its key set."""
    dumped = FirstAssessment.model_validate(SAMPLE).model_dump()
    node = dumped
    for part in path.split("."):
        node = node[part]
    items = node if isinstance(node, list) else [node]
    assert items, f"{path} has no items to check"
    for item in items:
        assert set(item) == expected


def test_extra_key_is_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({**SAMPLE, "notes": "extra"})


def test_extra_key_in_a_nested_section_is_rejected():
    payload = json.loads(json.dumps(SAMPLE))
    payload["objectiveAssessment"]["tests"][0]["severity"] = "high"
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate(payload)


@pytest.mark.parametrize("renamed", ["clinical_details", "recommendations", "patientAdvise"])
def test_renamed_key_is_rejected(renamed):
    """A renamed key arrives as an unknown key, which is exactly the point."""
    payload = json.loads(json.dumps(SAMPLE))
    payload[renamed] = payload.pop(
        {"clinical_details": "clinicalDetails", "recommendations": "recommendation", "patientAdvise": "patientAdvice"}[renamed]
    )
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate(payload)


# --------------------------------------------------------------------------- #
# Rule: arrays stay arrays
# --------------------------------------------------------------------------- #
def test_empty_assessment_still_renders_every_array():
    dumped = FirstAssessment().model_dump()
    for key in ("subjectiveAssessments", "subjectiveGoals", "objectiveGoals", "recommendation"):
        assert dumped[key] == []
    assert dumped["objectiveAssessment"]["tests"] == []


def test_single_item_stays_an_array():
    parsed = FirstAssessment.model_validate(SAMPLE)
    assert isinstance(parsed.model_dump()["recommendation"], list)
    assert len(parsed.recommendation) == 1


def test_null_array_becomes_empty_array():
    payload = json.loads(json.dumps(SAMPLE))
    payload["subjectiveGoals"] = None
    payload["objectiveAssessment"] = None
    dumped = FirstAssessment.model_validate(payload).model_dump()
    assert dumped["subjectiveGoals"] == []
    assert dumped["objectiveAssessment"] == {"tests": []}


def test_a_bare_object_where_an_array_belongs_is_rejected():
    payload = json.loads(json.dumps(SAMPLE))
    payload["recommendation"] = {"sessionType": "Outpatient", "sessionFrequency": "Weekly"}
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate(payload)


# --------------------------------------------------------------------------- #
# Rule: strings are strings, never null
# --------------------------------------------------------------------------- #
def test_no_leaf_is_ever_null():
    """The strongest form of the rule: walk every leaf of an empty document."""
    for path, value in _walk(FirstAssessment().model_dump()):
        assert value is not None, f"{path} is null"


def test_null_string_becomes_empty_string():
    payload = json.loads(json.dumps(SAMPLE))
    payload["clinicalDetails"]["duration"] = None
    payload["patientAdvice"]["adviceDetails"] = None
    dumped = FirstAssessment.model_validate(payload).model_dump()
    assert dumped["clinicalDetails"]["duration"] == ""
    assert dumped["patientAdvice"]["adviceDetails"] == ""


def test_numeric_value_is_coerced_to_string():
    """LLMs return `"value": 40` constantly; that must not become an int on the wire."""
    payload = json.loads(json.dumps(SAMPLE))
    payload["objectiveAssessment"]["tests"][0]["value"] = 40
    payload["objectiveAssessment"]["tests"][1]["left"] = 4.5
    tests = FirstAssessment.model_validate(payload).model_dump()["objectiveAssessment"]["tests"]
    assert tests[0]["value"] == "40"
    assert tests[1]["left"] == "4.5"


def test_structural_value_where_a_string_belongs_is_rejected():
    """Coercion is for scalars only: a dict in a string slot is a real error."""
    payload = json.loads(json.dumps(SAMPLE))
    payload["clinicalDetails"]["chiefComplaint"] = {"text": "back pain"}
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate(payload)


def test_assignment_is_normalised_too():
    """validate_assignment keeps the invariants alive after construction."""
    parsed = FirstAssessment.model_validate(SAMPLE)
    parsed.clinicalDetails.duration = None
    assert parsed.clinicalDetails.duration == ""


# --------------------------------------------------------------------------- #
# Provenance lives outside the contract
# --------------------------------------------------------------------------- #
def test_confidence_metadata_is_not_part_of_the_contract():
    """The 422 path needs confidence; the contract has no room for it."""
    for name in ("flags", "confidence", "meta", "unresolvedFields"):
        assert name not in FirstAssessment.model_fields
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({**SAMPLE, "flags": {"overallConfidence": 0.4}})


def test_stored_envelope_wraps_the_contract_without_touching_it():
    stored = StoredAssessment(
        id="65f0c0ffee",
        transcript="...",
        audioFilename="clinical_assessment.wav",
        flags=ExtractionFlags.summarise(
            [_evidence(field="objectiveGoals[0].targetDate", modelConfidence=0.2)],
            unresolved=["subjectiveGoals[0].targetDate"],
        ),
        assessment=FirstAssessment.model_validate(SAMPLE),
    )
    assert stored.model_dump()["assessment"] == SAMPLE
    assert isinstance(stored.createdAt, datetime)
    assert stored.flags.fields[0].confidence == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_confidence_must_be_a_probability(bad):
    with pytest.raises(ValidationError):
        ExtractionFlags(overallConfidence=bad)


# --------------------------------------------------------------------------- #
# Per-field confidence: grounding gate + the two independent signals
# --------------------------------------------------------------------------- #
def _evidence(**kw) -> FieldEvidence:
    """A grounded, high-confidence field unless the test says otherwise."""
    return FieldEvidence(
        **{
            "field": "clinicalDetails.duration",
            "value": "about three weeks",
            "evidence": "it started about three weeks ago",
            "evidenceFound": True,
            "modelConfidence": 0.9,
            **kw,
        }
    )


def test_ungrounded_field_scores_zero_however_sure_the_model_claims_to_be():
    """The anti-hallucination gate: an unquotable value is worth nothing."""
    field = _evidence(evidenceFound=False, modelConfidence=1.0, audioConfidence=1.0)
    assert field.confidence == 0.0


def test_weakest_signal_decides():
    """Confident extraction from badly-heard audio must not score well."""
    misheard = _evidence(modelConfidence=0.95, audioConfidence=0.31)
    assert misheard.confidence == pytest.approx(0.31)


def test_missing_audio_signal_falls_back_to_the_model_score():
    assert _evidence(audioConfidence=None).confidence == pytest.approx(0.9)


def test_a_silent_model_is_not_treated_as_a_screaming_one():
    """Models are erratic about filling in an optional confidence number.

    Reading an omitted score as 0.0 would zero out every well-grounded field in
    a section and bury the genuinely suspect ones in noise, which is exactly
    what happened on the first live run.
    """
    quiet = _evidence(modelConfidence=0.0, audioConfidence=0.94)
    assert quiet.confidence == pytest.approx(0.94)


def test_a_field_with_no_signals_at_all_scores_zero():
    assert _evidence(modelConfidence=0.0, audioConfidence=None).confidence == 0.0


def test_destroyed_audio_beside_a_clean_quote_drags_the_field_down():
    field = _evidence(modelConfidence=0.95, audioConfidence=0.93, contextConfidence=0.05)
    assert field.confidence == pytest.approx(0.05)


@pytest.mark.parametrize("nearby", [0.26, 0.5, 0.9])
def test_merely_imperfect_audio_nearby_is_recorded_but_not_charged(nearby):
    field = _evidence(modelConfidence=0.9, audioConfidence=0.9, contextConfidence=nearby)
    assert field.confidence == pytest.approx(0.9)


def test_the_422_gate_is_per_field_not_overall():
    """One bad measurement in an otherwise clean assessment must still fail."""
    flags = ExtractionFlags.summarise(
        [_evidence(field=f"f{i}") for i in range(9)]
        + [_evidence(field="objectiveAssessment.tests[0].value", audioConfidence=0.15)]
    )
    assert flags.overallConfidence > 0.8  # the average looks perfectly healthy
    failed = flags.below(0.5)  # the per-field gate is not fooled
    assert [f.field for f in failed] == ["objectiveAssessment.tests[0].value"]


def test_ungrounded_fields_are_reported_separately():
    flags = ExtractionFlags.summarise(
        [_evidence(), _evidence(field="patientAdvice.adviceDetails", evidenceFound=False)]
    )
    assert [f.field for f in flags.ungrounded()] == ["patientAdvice.adviceDetails"]


def test_an_extraction_that_found_nothing_scores_zero_not_one():
    assert ExtractionFlags.summarise([]).overallConfidence == 0.0


# --------------------------------------------------------------------------- #
# Locating a quote in the audio
# --------------------------------------------------------------------------- #
def _transcription() -> TranscriptionResult:
    words = [("flexion", 0.98), ("was", 0.97), ("about", 0.95), ("forty", 0.41), ("degrees", 0.93)]
    return TranscriptionResult(
        text="flexion was about forty degrees",
        language="en",
        durationSec=12.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=12.0,
                text="flexion was about forty degrees",
                confidence=0.88,
                words=[
                    TranscriptWord(start=i, end=i + 1, word=w, confidence=c)
                    for i, (w, c) in enumerate(words)
                ],
            )
        ],
    )


def test_span_confidence_is_local_to_the_quote():
    """The whole segment reads 0.88; the number inside it does not."""
    result = _transcription()
    assert result.confidence_for("flexion was about") == pytest.approx(0.95)
    assert result.confidence_for("about forty degrees") == pytest.approx(0.41)


def test_a_quote_carrying_punctuation_still_matches():
    """The quote has commas, Whisper's word list does not. Strip both or nothing
    longer than a few words ever matches."""
    result = TranscriptionResult(
        text="flexion was about forty degrees",
        segments=[
            TranscriptSegment(
                text="flexion was about forty degrees",
                confidence=0.9,
                words=[
                    TranscriptWord(word=w, confidence=c)
                    for w, c in [("flexion,", 0.98), ("was", 0.97), ("about", 0.95)]
                ],
            )
        ],
    )
    assert result.confidence_for("flexion, was about") == pytest.approx(0.95)


def test_a_repeated_quote_scores_its_best_occurrence():
    """"degrees" occurs all over a recording; we cannot tell which one a value
    came from, so an unrelated bad occurrence must not manufacture a warning."""
    result = TranscriptionResult(
        text="124 degrees and 130 degrees",
        segments=[
            TranscriptSegment(
                text="124 degrees and 130 degrees",
                confidence=0.9,
                words=[
                    TranscriptWord(word=w, confidence=c)
                    for w, c in [
                        ("124", 1.0), ("degrees", 0.52), ("and", 0.99),
                        ("130", 1.0), ("degrees", 0.99),
                    ]
                ],
            )
        ],
    )
    assert result.confidence_for("degrees") == pytest.approx(0.99)
    # But a span that is weak everywhere it appears stays weak.
    assert result.confidence_for("124 degrees") == pytest.approx(0.52)


def test_context_widens_the_view_by_a_few_words():
    result = _transcription()  # flexion(98) was(97) about(95) forty(41) degrees(93)
    assert result.confidence_for("degrees") == pytest.approx(0.93)
    # "forty" sits three words back, inside the window.
    assert result.context_confidence("degrees") == pytest.approx(0.41)


def test_context_window_does_not_reach_the_whole_sentence():
    words = [("one", 0.02)] + [(w, 0.99) for w in "a b c d e target".split()]
    result = TranscriptionResult(
        text="one a b c d e target",
        segments=[
            TranscriptSegment(
                text="one a b c d e target",
                confidence=0.9,
                words=[TranscriptWord(word=w, confidence=c) for w, c in words],
            )
        ],
    )
    # The 2% word is six places away, outside a three-word window, and so not
    # this value's problem.
    assert result.context_confidence("target") == pytest.approx(0.99)
    assert result.context_confidence("target", window=6) == pytest.approx(0.02)


def test_a_misheard_function_word_nearby_is_not_an_alarm():
    """A mangled "and" between two goals puts neither of them in doubt.

    Function words are what Whisper mangles most and what carries least clinical
    meaning. On the real recording this exact case produced two false alarms.
    """
    words = [("improving", 0.99), ("ankle", 0.99), ("mobility", 1.0), ("and", 0.18), ("activating", 0.99)]
    result = TranscriptionResult(
        text="improving ankle mobility and activating",
        segments=[
            TranscriptSegment(
                text="improving ankle mobility and activating",
                confidence=0.9,
                words=[TranscriptWord(word=w, confidence=c) for w, c in words],
            )
        ],
    )
    assert result.context_confidence("improving ankle mobility") == pytest.approx(0.99)


def test_a_misheard_content_word_nearby_is_still_an_alarm():
    """"negative" before a measurement changes what the measurement says."""
    words = [("with", 1.0), ("negative", 0.05), ("5", 0.93), ("degrees", 0.99)]
    result = TranscriptionResult(
        text="with negative 5 degrees",
        segments=[
            TranscriptSegment(
                text="with negative 5 degrees",
                confidence=0.9,
                words=[TranscriptWord(word=w, confidence=c) for w, c in words],
            )
        ],
    )
    assert result.context_confidence("5 degrees") == pytest.approx(0.05)


def test_context_of_an_unlocatable_span_is_none():
    assert _transcription().context_confidence("never said this") is None


def test_unlocatable_span_returns_none_not_zero():
    """'No signal' and 'bad audio' are different answers and must stay different."""
    assert _transcription().confidence_for("patient reported nausea") is None


def test_span_lookup_falls_back_to_segments_without_word_timestamps():
    result = TranscriptionResult(
        text="flexion was about forty degrees",
        segments=[TranscriptSegment(text="flexion was about forty degrees", confidence=0.62)],
    )
    assert result.confidence_for("about forty") == pytest.approx(0.62)
    assert result.confidence_for("not in here") is None
