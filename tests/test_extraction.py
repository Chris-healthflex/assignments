"""Tests for the extraction agent.

No API key and no network: the model is stubbed, because what needs proving
here is not "does Gemini work" but "does our grounding catch a model that
lies". A stub is the only way to test the lying case deliberately.

The transcript below is a trimmed excerpt of the real recording, including its
real mishearing ("knee gig 5 degrees"), so the tests exercise the same
ambiguity the pipeline actually meets.
"""

from __future__ import annotations

import pytest

from app import extraction
from app.extraction import Citation, ExtractionResult, ground
from app.schemas import (
    ClinicalDetails,
    ExtractionFlags,
    FieldEvidence,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)

TRANSCRIPT = (
    "She reports moderate pain with mild irritability particularly during "
    "prolonged walking and standing which is relieved with rest. Objective "
    "measurements showed left knee flexion of 124 degrees compared with 130 "
    "degrees on the right, left knee extension of 20 degrees compared with "
    "knee gig 5 degrees on the right. Physiotherapy was recommended once "
    "weekly for four sessions."
)


def _transcription() -> TranscriptionResult:
    """Word probabilities mirroring the real run: 'knee' before '5' is 5%."""
    spoken = [
        ("left", 0.99), ("knee", 1.0), ("flexion", 0.99), ("of", 1.0),
        ("124", 1.0), ("degrees", 1.0), ("compared", 1.0), ("with", 1.0),
        ("knee", 0.05), ("gig", 0.86), ("5", 0.93), ("degrees", 0.99),
    ]
    return TranscriptionResult(
        text=TRANSCRIPT,
        segments=[
            TranscriptSegment(
                text=TRANSCRIPT,
                confidence=0.90,  # the segment looks perfectly healthy
                words=[
                    TranscriptWord(start=i, end=i + 1, word=w, confidence=c)
                    for i, (w, c) in enumerate(spoken)
                ],
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Grounding: the anti-hallucination check, in plain code
# --------------------------------------------------------------------------- #
def test_a_quoted_value_is_grounded():
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[ObjectiveTest(testName="Knee flexion", unitName="degrees", left="124")]
        )
    )
    citations = [
        Citation(value="Knee flexion", evidence="left knee flexion", confidence=0.9),
        Citation(value="degrees", evidence="124 degrees", confidence=0.9),
        Citation(value="124", evidence="left knee flexion of 124 degrees", confidence=0.95),
    ]
    fields = ground(assessment, TRANSCRIPT, None, {"objective": citations})
    left = next(f for f in fields if f.field.endswith(".left"))
    assert left.evidenceFound is True
    assert left.confidence == pytest.approx(0.95)


def test_an_invented_value_is_caught_by_its_missing_quote():
    """The model claims a source that is not in the transcript. Score zero."""
    assessment = FirstAssessment(clinicalDetails=ClinicalDetails(duration="eight months"))
    citations = [
        Citation(
            value="eight months",
            evidence="the injury occurred eight months ago",  # never said
            confidence=0.99,
        )
    ]
    fields = ground(assessment, TRANSCRIPT, None, {"subjective": citations})
    duration = next(f for f in fields if f.field == "clinicalDetails.duration")
    assert duration.evidenceFound is False
    assert duration.confidence == 0.0
    assert "does not appear" in duration.reason


def test_a_value_with_no_citation_at_all_is_treated_as_unsupported():
    assessment = FirstAssessment(patientAdvice=PatientAdvice(adviceDetails="Apply ice twice daily"))
    fields = ground(assessment, TRANSCRIPT, None, {})
    assert fields[0].evidenceFound is False
    assert fields[0].reason == "No source quoted for this value."


def test_empty_fields_are_not_graded():
    """An empty field claims nothing, so there is nothing to verify."""
    assert ground(FirstAssessment(), TRANSCRIPT, None, {}) == []


def test_field_paths_are_computed_by_us_including_array_indices():
    assessment = FirstAssessment(
        subjectiveAssessments=[
            SubjectiveAssessment(testName="Pain", conclusion="moderate"),
            SubjectiveAssessment(testName="Irritability", conclusion="mild"),
        ]
    )
    paths = [f.field for f in ground(assessment, TRANSCRIPT, None, {})]
    assert "subjectiveAssessments[1].testName" in paths
    assert "subjectiveAssessments[0].conclusion" in paths


def test_citation_matching_tolerates_a_wider_quoted_value():
    """A model citing "124 degrees" for a field holding "124" still counts."""
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(tests=[ObjectiveTest(left="124")])
    )
    citations = [Citation(value="124 degrees", evidence="124 degrees", confidence=0.8)]
    left = next(
        f
        for f in ground(assessment, TRANSCRIPT, None, {"objective": citations})
        if f.field.endswith(".left")
    )
    assert left.evidenceFound is True


def test_a_citation_from_one_group_cannot_ground_another_groups_field():
    """Quotes are matched inside the group that gave them.

    `unitName` is "degrees" in both `objectiveAssessment.tests` (the objective
    call) and `objectiveGoals` (the plan call). Pooled into one list, the plan
    call's invented quote was matched first and decided the fate of a field the
    objective call had quoted correctly. The two are not competing accounts of
    one value, they are evidence about two different ones.
    """
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[ObjectiveTest(testName="Knee flexion", unitName="degrees", left="124")]
        ),
        objectiveGoals=[
            ObjectiveGoal(goalName="Improve flexion", unitName="degrees", value="140")
        ],
    )
    citations = {
        "objective": [
            Citation(value="Knee flexion", evidence="left knee flexion", confidence=0.9),
            Citation(value="degrees", evidence="124 degrees", confidence=0.95),
            Citation(value="124", evidence="knee flexion of 124 degrees", confidence=0.95),
        ],
        "plan": [
            # Never said. The plan call invented its deadline.
            Citation(value="degrees", evidence="a target of 140 degrees by June", confidence=0.9),
        ],
    }
    fields = {f.field: f for f in ground(assessment, TRANSCRIPT, _transcription(), citations)}

    unit = fields["objectiveAssessment.tests[0].unitName"]
    assert unit.evidence == "124 degrees"  # its own call's quote, not the other one's
    assert unit.evidenceFound is True
    # And the invented quote still condemns the field it was actually given for.
    assert fields["objectiveGoals[0].unitName"].evidenceFound is False


def test_a_citation_for_a_shorter_number_does_not_vouch_for_a_longer_one():
    """"5" must not stand as evidence for "15".

    The containment fallback exists so a model citing "124 degrees" for a field
    holding "124" still counts. Measured on raw characters it also let the quote
    for one measurement ground a different one, which produces the worst output
    this pipeline can produce: a wrong number carrying a quote that really is in
    the transcript, scoring well above the review threshold.
    """
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(tests=[ObjectiveTest(left="15")])
    )
    citations = {
        "objective": [
            Citation(value="5", evidence="knee gig 5 degrees on the right", confidence=0.9)
        ]
    }
    left = ground(assessment, TRANSCRIPT, _transcription(), citations)[0]

    assert left.evidenceFound is False
    assert left.reason == "No source quoted for this value."
    assert left.confidence == 0.0


def test_the_most_specific_quote_wins_when_several_fit():
    """Order of arrival must not decide which quote is used.

    Nothing cites this value exactly, so both quotes reach the containment pass
    and both fit. The vaguer one is listed first, which is what taking the first
    match would have picked.
    """
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[ObjectiveTest(testName="left knee flexion of 124 degrees")]
        )
    )
    citations = {
        "objective": [
            Citation(value="knee", evidence="knee", confidence=0.5),
            Citation(value="left knee flexion", evidence="left knee flexion", confidence=0.9),
        ]
    }
    name = ground(assessment, TRANSCRIPT, _transcription(), citations)[0]
    assert name.evidence == "left knee flexion"


# --------------------------------------------------------------------------- #
# The case only audio confidence can catch
# --------------------------------------------------------------------------- #
def test_a_perfect_extraction_from_misheard_audio_is_still_flagged():
    """The whole design in one test.

    The agent behaves impeccably: it extracts "5", quotes the transcript
    correctly, and is rightly confident. The transcript itself is wrong: the
    words around that number scored 5% in Whisper. Nothing but the audio signal
    can see this.
    """
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[ObjectiveTest(testName="Knee extension", right="5", unitName="degrees")]
        )
    )
    citations = [
        Citation(value="Knee extension", evidence="left knee extension", confidence=0.95),
        Citation(value="degrees", evidence="20 degrees", confidence=0.95),
        Citation(value="5", evidence="knee gig 5 degrees", confidence=0.95),
    ]
    fields = ground(assessment, TRANSCRIPT, _transcription(), {"objective": citations})
    right = next(f for f in fields if f.field.endswith(".right"))

    assert right.evidenceFound is True          # the agent did nothing wrong
    assert right.modelConfidence == 0.95        # and is sure of itself
    assert right.audioConfidence == pytest.approx(0.05)  # but Whisper was not
    assert right.confidence == pytest.approx(0.05)       # weakest signal decides

    flags = ExtractionFlags.summarise(fields)
    assert [f.field for f in flags.below(0.6)] == ["objectiveAssessment.tests[0].right"]


def test_a_model_cannot_quote_its_way_around_a_hole_in_the_transcript():
    """The live run's escape hatch, closed.

    Told to quote the shortest span that establishes the value, the model
    quotes "5 degrees on the right", every word of which Whisper heard at 93%
    or better. It obeyed perfectly, and in doing so stepped straight over the
    5% word that probably reads "negative". Scoring the quote alone passes it.
    """
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(tests=[ObjectiveTest(right="5")])
    )
    citations = [Citation(value="5", evidence="5 degrees", confidence=0.95)]
    right = ground(assessment, TRANSCRIPT, _transcription(), {"objective": citations})[0]

    assert right.audioConfidence == pytest.approx(0.93)  # the quote is clean
    assert right.contextConfidence == pytest.approx(0.05)  # its surroundings are not
    assert right.confidence == pytest.approx(0.05)  # so the field is still flagged
    assert "badly unclear next to this value" in right.reason


def test_an_ordinary_mumble_nearby_does_not_flag_a_good_value():
    """Context is a tripwire, not a term.

    Everyday speech dips to 0.5 or 0.6 constantly. Folding that into the score
    would flag half the document and bury the real warnings in noise.
    """
    transcription = TranscriptionResult(
        text="knee flexion of 124 degrees",
        segments=[
            TranscriptSegment(
                text="knee flexion of 124 degrees",
                confidence=0.9,
                words=[
                    TranscriptWord(word=w, confidence=c)
                    for w, c in [
                        ("knee", 0.55), ("flexion", 0.61), ("of", 0.58),
                        ("124", 0.99), ("degrees", 0.97),
                    ]
                ],
            )
        ],
    )
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(tests=[ObjectiveTest(left="124")])
    )
    citations = [Citation(value="124", evidence="124 degrees", confidence=0.95)]
    left = ground(
        assessment, "knee flexion of 124 degrees", transcription, {"objective": citations}
    )[0]

    assert left.contextConfidence == pytest.approx(0.55)  # mediocre, and recorded
    assert left.confidence == pytest.approx(0.95)  # but it does not drag the score
    assert left.reason == ""


# --------------------------------------------------------------------------- #
# The graph, with the model stubbed out
# --------------------------------------------------------------------------- #
def _canned(**overrides):
    """Group outputs a well-behaved model would return for TRANSCRIPT."""
    base = {
        "subjective": extraction.SubjectiveOut(
            clinicalDetails=ClinicalDetails(chiefComplaint="moderate pain"),
            subjectiveAssessments=[
                SubjectiveAssessment(testName="Pain", conclusion="relieved with rest")
            ],
            citations=[
                Citation(value="moderate pain", evidence="moderate pain", confidence=0.9),
                Citation(value="Pain", evidence="moderate pain", confidence=0.9),
                Citation(value="relieved with rest", evidence="which is relieved with rest", confidence=0.9),
            ],
        ),
        "objective": extraction.ObjectiveOut(
            objectiveAssessment=ObjectiveAssessment(
                tests=[ObjectiveTest(testName="Knee flexion", unitName="degrees", left="124", right="130")]
            ),
            citations=[
                Citation(value="Knee flexion", evidence="left knee flexion", confidence=0.95),
                Citation(value="degrees", evidence="124 degrees", confidence=0.95),
                Citation(value="124", evidence="knee flexion of 124 degrees", confidence=0.95),
                Citation(value="130", evidence="compared with 130 degrees on the right", confidence=0.95),
            ],
        ),
        "plan": extraction.PlanOut(
            recommendation=[
                Recommendation(sessionType="Physiotherapy", sessionFrequency="once weekly for four sessions")
            ],
            patientAdvice=PatientAdvice(),
            citations=[
                Citation(value="Physiotherapy", evidence="Physiotherapy was recommended", confidence=0.95),
                Citation(value="once weekly for four sessions", evidence="once weekly for four sessions", confidence=0.95),
            ],
        ),
    }
    base.update(overrides)
    return base


def _plan_with_invented_advice() -> "extraction.PlanOut":
    """A plan group that invents patient advice and cites a quote that is not there."""
    plan = _canned()["plan"]
    return extraction.PlanOut(
        recommendation=plan.recommendation,
        patientAdvice=PatientAdvice(adviceDetails="Apply ice twice daily"),
        citations=list(plan.citations)
        + [Citation(value="Apply ice twice daily", evidence="apply ice twice daily", confidence=0.9)],
    )


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the LLM call, recording which groups were asked and how often."""
    calls: list[tuple[str, str]] = []
    responses = _canned()

    def fake(group, transcript, hint=""):
        calls.append((group, hint))
        return responses[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    return calls, responses


def test_the_graph_assembles_every_section(stub_model):
    calls, _ = stub_model
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert isinstance(result, ExtractionResult)
    assert sorted(g for g, _ in calls) == ["objective", "plan", "subjective"]
    dumped = result.assessment.model_dump()
    assert tuple(dumped) == extraction.SECTIONS  # all seven, in contract order
    assert dumped["objectiveAssessment"]["tests"][0]["left"] == "124"
    assert dumped["recommendation"][0]["sessionType"] == "Physiotherapy"
    # A section the transcript never covered stays empty rather than invented.
    assert dumped["patientAdvice"]["adviceDetails"] == ""


def test_three_calls_not_seven(stub_model):
    """The free tier allows 5 requests/minute; the call count is a design constraint."""
    calls, _ = stub_model
    extraction.extract(TRANSCRIPT, _transcription())
    assert len(calls) == 3
    assert set(extraction.SECTION_TO_GROUP) == set(extraction.SECTIONS)


def test_untouched_sections_are_reported_as_unresolved_not_guessed(stub_model):
    result = extraction.extract(TRANSCRIPT, _transcription())
    assert "patientAdvice.adviceDetails" in result.flags.unresolvedFields
    assert "clinicalDetails.duration" in result.flags.unresolvedFields


def test_every_populated_field_gets_an_evidence_record(stub_model):
    result = extraction.extract(TRANSCRIPT, _transcription())
    populated = {
        path
        for path, value in extraction._leaves(result.assessment.model_dump())
        if str(value).strip()
    }
    assert {f.field for f in result.flags.fields} == populated


def test_a_hallucinating_group_triggers_a_targeted_repair(monkeypatch):
    """Repair re-asks only the offending group, and accepts an empty answer."""
    calls: list[tuple[str, str]] = []
    responses = _canned(plan=_plan_with_invented_advice())

    def fake(group, transcript, hint=""):
        calls.append((group, hint))
        if group == "plan" and hint:
            # Told its quote was unverifiable, the model backs off correctly.
            return _canned()["plan"], ""
        return responses[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    repairs = [(g, h) for g, h in calls if h]
    assert [g for g, _ in repairs] == ["plan"]  # nothing else re-run
    assert "does not appear in the transcript" in repairs[0][1]
    assert "patientAdvice.adviceDetails" in repairs[0][1]  # names the field
    assert result.assessment.patientAdvice.adviceDetails == ""
    assert result.flags.ungrounded() == []


def test_a_repair_that_corrects_its_quote_is_accepted(monkeypatch):
    """The other half of the repair loop.

    A model told its quote is unverifiable has two acceptable answers: empty the
    field, or quote properly. Only the first was ever exercised. Citations
    accumulated across attempts and matching took the first quote that fit, so
    the rejected one still won and a model that did exactly what it was asked
    was told a second and third time that it had invented the value. The field
    then reached the clinician marked as unsupported.
    """
    canned = _canned()
    quoted = "which is relieved with rest"

    def plan_citing(evidence: str) -> "extraction.PlanOut":
        return extraction.PlanOut(
            recommendation=canned["plan"].recommendation,
            patientAdvice=PatientAdvice(adviceDetails="relieved with rest"),
            citations=list(canned["plan"].citations)
            + [Citation(value="relieved with rest", evidence=evidence, confidence=0.9)],
        )

    calls: list[tuple[str, str]] = []

    def fake(group, transcript, hint=""):
        calls.append((group, hint))
        if group == "plan":
            # Corrects itself once told the first quote was not in the transcript.
            return plan_citing(quoted if hint else "the patient was told to rest"), ""
        return canned[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert [g for g, h in calls if h] == ["plan"]  # asked again exactly once
    advice = next(f for f in result.flags.fields if f.field == "patientAdvice.adviceDetails")
    assert advice.evidence == quoted  # the corrected quote, not the rejected one
    assert advice.evidenceFound is True
    assert advice.confidence > 0
    assert result.flags.ungrounded() == []


def test_a_group_that_recovers_leaves_no_warning_behind(monkeypatch):
    """A warning describes the document, not the journey.

    Warnings were copied from an append-only error list, so a group that failed
    once and succeeded on the retry still carried "could not be extracted" into
    the response and into the saved record, contradicting the fully populated
    section sitting next to it.
    """
    seen: list[str] = []

    def fake(group, transcript, hint=""):
        seen.append(group)
        if group == "objective" and seen.count("objective") == 1:
            return None, "503 UNAVAILABLE"  # transient
        return _canned()[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert result.assessment.objectiveAssessment.tests  # the section came back
    assert result.flags.failedSections == []
    assert result.flags.incomplete is False
    assert result.flags.warnings == []


def test_repair_gives_up_rather_than_looping_forever(monkeypatch):
    """A model that keeps inventing must not spin the graph indefinitely."""
    calls: list[str] = []

    def fake(group, transcript, hint=""):
        calls.append(group)
        result = _plan_with_invented_advice() if group == "plan" else _canned()[group]
        return result, ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert calls.count("plan") == 3  # first pass plus two repairs
    # It survives, and says plainly which field it could not stand behind.
    assert [f.field for f in result.flags.ungrounded()] == ["patientAdvice.adviceDetails"]


def test_a_failing_group_does_not_sink_the_run(monkeypatch):
    def fake(group, transcript, hint=""):
        if group == "objective":
            return None, "503 UNAVAILABLE"
        return _canned()[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert result.assessment.objectiveAssessment.tests == []
    assert result.assessment.recommendation[0].sessionType == "Physiotherapy"
    assert any("objective" in w for w in result.flags.warnings)


def test_a_group_that_returned_nothing_is_retried(monkeypatch):
    """The gap that lost a whole section to one transient 503.

    Repair used to look only at fields whose quotes did not check out. A group
    whose call failed produced no fields at all, so it was invisible to that
    loop and never asked again. A bad quote got two more chances while a
    dropped connection got none.
    """
    attempts: list[str] = []

    def fake(group, transcript, hint=""):
        attempts.append(group)
        if group == "subjective" and attempts.count("subjective") == 1:
            return None, "503 UNAVAILABLE"
        return _canned()[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert attempts.count("subjective") == 2  # asked again after the failure
    assert result.assessment.clinicalDetails.chiefComplaint  # and it came back
    assert result.flags.failedSections == []  # so nothing is reported missing
    assert result.flags.incomplete is False


def test_a_group_that_never_recovers_names_the_sections_it_cost(monkeypatch):
    def fake(group, transcript, hint=""):
        if group == "subjective":
            return None, "503 UNAVAILABLE"
        return _canned()[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    # Named by contract section, not by internal group: "subjective" means
    # nothing to a caller reading the response.
    assert result.flags.failedSections == ["clinicalDetails", "subjectiveAssessments"]
    assert result.flags.incomplete is True
    assert any("not because the recording was silent" in w for w in result.flags.warnings)


def test_a_section_lost_to_a_failed_call_is_not_called_unresolved(monkeypatch):
    """`unresolvedFields` means the transcript did not support it.

    That is a claim about the recording, and a call that never returned gives
    us no standing to make it. Conflating the two would let a network error
    masquerade as a clinical finding.
    """

    def fake(group, transcript, hint=""):
        if group == "subjective":
            return None, "503 UNAVAILABLE"
        return _canned()[group], ""

    monkeypatch.setattr(extraction, "_run_group", fake)
    result = extraction.extract(TRANSCRIPT, _transcription())

    assert not any(p.startswith("clinicalDetails") for p in result.flags.unresolvedFields)
    assert not any(p.startswith("subjectiveAssessments") for p in result.flags.unresolvedFields)
    # A field that is genuinely empty because nobody mentioned it still is.
    assert "patientAdvice.adviceDetails" in result.flags.unresolvedFields


def test_every_group_failing_is_no_result_rather_than_an_empty_one(monkeypatch):
    """An empty document would be indistinguishable from a silent recording."""

    def fake(group, transcript, hint=""):
        return None, "429 RESOURCE_EXHAUSTED"

    monkeypatch.setattr(extraction, "_run_group", fake)

    with pytest.raises(extraction.ExtractionUnavailable) as raised:
        extraction.extract(TRANSCRIPT, _transcription())

    assert "429" in str(raised.value)
    # Still an ExtractionFailed, so existing handlers keep working.
    assert isinstance(raised.value, extraction.ExtractionFailed)


def test_empty_transcript_is_refused():
    with pytest.raises(extraction.ExtractionFailed, match="Empty transcript"):
        extraction.extract("   ")


def test_failing_fields_drive_the_422(stub_model):
    result = extraction.extract(TRANSCRIPT, _transcription())
    result.flags.fields = [
        FieldEvidence(field="a", value="x", evidenceFound=True, modelConfidence=0.9),
        FieldEvidence(field="b", value="y", evidenceFound=True, modelConfidence=0.2),
    ]
    assert [f.field for f in result.failing()] == ["b"]
