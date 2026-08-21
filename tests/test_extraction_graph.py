from app.schemas.first_assessment import (
    ClinicalDetails,
    FirstAssessment,
    PatientAdvice,
    SubjectiveAssessment,
)
from app.services.extraction_graph import (
    ExtractionResult,
    FieldEvidence,
    populated_paths,
    run_extraction,
    validate_extraction,
)
from app.services.transcription import Transcript, TranscriptSegment


class FakeLLM:
    """Stands in for ChatGroq(...).with_structured_output(...) in tests."""

    def __init__(self, *results: ExtractionResult):
        self._results = list(results)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        # Repeat the last answer once the scripted responses run out.
        index = min(self.calls - 1, len(self._results) - 1)
        return self._results[index].model_copy(deep=True)


def _transcript() -> Transcript:
    return Transcript(
        text="Where does it hurt? My left knee, for about eight months.",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=2.0, text="Where does it hurt?"),
            TranscriptSegment(
                id=1, start=2.0, end=5.0, text="My left knee, for about eight months."
            ),
        ],
    )


# --------------------------------------------------------------------------
# Core behaviour
# --------------------------------------------------------------------------


def test_run_extraction_returns_assessment_and_confidence_flag():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Lower back pain"),
            subjectiveAssessments=[
                SubjectiveAssessment(testName="SLR", conclusion="Positive")
            ],
        ),
        low_confidence_sections=[],
    )
    llm = FakeLLM(fake_result)

    report, is_low_confidence = run_extraction("some transcript", llm=llm)

    assert llm.calls == 1
    assert report.assessment.clinicalDetails.chiefComplaint == "Lower back pain"
    assert is_low_confidence is False


def test_run_extraction_flags_low_confidence_past_threshold():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(),
        low_confidence_sections=["subjectiveGoals", "objectiveGoals"],
    )
    llm = FakeLLM(fake_result)

    _, is_low_confidence = run_extraction(
        "sparse transcript", llm=llm, confidence_threshold=2
    )

    assert is_low_confidence is True


def test_run_extraction_below_threshold_is_not_flagged():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(),
        low_confidence_sections=["subjectiveGoals"],
    )
    llm = FakeLLM(fake_result)

    _, is_low_confidence = run_extraction(
        "mostly complete transcript", llm=llm, confidence_threshold=2
    )

    assert is_low_confidence is False


def test_run_extraction_drops_unknown_section_names():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(),
        low_confidence_sections=["not_a_real_section"],
    )
    llm = FakeLLM(fake_result)

    report, _ = run_extraction("transcript", llm=llm)

    assert report.low_confidence_sections == []


def test_plain_string_transcript_skips_grounding_checks():
    """Without timestamps there are no segments to cite, so an uncited value
    is not evidence of hallucination and must not trigger a refine loop."""
    fake_result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
        ),
        evidence=[],
    )
    llm = FakeLLM(fake_result)

    report, _ = run_extraction("plain transcript", llm=llm)

    assert llm.calls == 1
    assert report.ungrounded_fields == []


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_populated_paths_reports_dotted_leaf_paths():
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain"),
        subjectiveAssessments=[
            SubjectiveAssessment(testName="SLR", conclusion="Positive")
        ],
    )

    assert set(populated_paths(assessment)) == {
        "clinicalDetails.chiefComplaint",
        "subjectiveAssessments[0].testName",
        "subjectiveAssessments[0].conclusion",
    }


def test_validate_flags_values_without_supporting_evidence():
    result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain", duration="8 months")
        ),
        evidence=[
            FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[1])
        ],
    )

    issues, ungrounded = validate_extraction(result, {0, 1}, check_grounding=True)

    assert ungrounded == ["clinicalDetails.duration"]
    assert any("no supporting transcript segment" in issue for issue in issues)


def test_validate_accepts_parent_level_evidence():
    result = ExtractionResult(
        assessment=FirstAssessment(
            subjectiveAssessments=[
                SubjectiveAssessment(testName="SLR", conclusion="Positive")
            ]
        ),
        evidence=[FieldEvidence(field="subjectiveAssessments[0]", segmentIds=[0])],
    )

    _, ungrounded = validate_extraction(result, {0, 1}, check_grounding=True)

    assert ungrounded == []


def test_validate_flags_placeholder_text():
    result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="N/A", duration="unknown"),
            patientAdvice=PatientAdvice(adviceDetails="not mentioned"),
        ),
        evidence=[
            FieldEvidence(field="clinicalDetails", segmentIds=[0]),
            FieldEvidence(field="patientAdvice", segmentIds=[0]),
        ],
    )

    issues, _ = validate_extraction(result, {0}, check_grounding=True)

    assert any("placeholder text" in issue for issue in issues)


def test_validate_flags_section_both_flagged_and_populated():
    result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
        ),
        low_confidence_sections=["clinicalDetails"],
        evidence=[FieldEvidence(field="clinicalDetails", segmentIds=[0])],
    )

    issues, _ = validate_extraction(result, {0}, check_grounding=True)

    assert any("listed in low_confidence_sections" in issue for issue in issues)


def test_validate_flags_citations_to_nonexistent_segments():
    result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
        ),
        evidence=[FieldEvidence(field="clinicalDetails", segmentIds=[99])],
    )

    issues, _ = validate_extraction(result, {0, 1}, check_grounding=True)

    assert any("do not exist" in issue for issue in issues)


def test_validate_flags_empty_assessment_that_claims_confidence():
    result = ExtractionResult(assessment=FirstAssessment(), low_confidence_sections=[])

    issues, _ = validate_extraction(result, {0}, check_grounding=True)

    assert any("entirely empty" in issue for issue in issues)


def test_validate_is_silent_on_a_clean_extraction():
    result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
        ),
        low_confidence_sections=["patientAdvice"],
        evidence=[
            FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[1])
        ],
    )

    issues, ungrounded = validate_extraction(result, {0, 1}, check_grounding=True)

    assert issues == []
    assert ungrounded == []


# --------------------------------------------------------------------------
# Self-correction loop
# --------------------------------------------------------------------------


def test_ungrounded_extraction_triggers_a_refine_pass():
    hallucinated = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain", duration="8 months")
        ),
        evidence=[FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[1])],
    )
    corrected = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain", duration="8 months")
        ),
        evidence=[
            FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[1]),
            FieldEvidence(field="clinicalDetails.duration", segmentIds=[1]),
        ],
    )
    llm = FakeLLM(hallucinated, corrected)

    report, _ = run_extraction(_transcript(), llm=llm)

    assert llm.calls == 2
    assert report.attempts == 2
    assert report.ungrounded_fields == []
    assert report.validation_issues == []


def test_refine_loop_is_bounded_and_reports_remaining_issues():
    """A model that never fixes its output must not loop forever — the graph
    gives up after max_refinements and surfaces what is still wrong."""
    stubborn = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Invented complaint")
        ),
        evidence=[],
    )
    llm = FakeLLM(stubborn)

    report, _ = run_extraction(_transcript(), llm=llm, max_refinements=1)

    assert llm.calls == 2  # initial extract + one refine, then stop
    assert report.ungrounded_fields == ["clinicalDetails.chiefComplaint"]
    assert report.validation_issues != []


def test_clean_extraction_never_calls_the_model_twice():
    clean = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
        ),
        low_confidence_sections=[
            "subjectiveAssessments",
            "objectiveAssessment",
            "subjectiveGoals",
            "objectiveGoals",
            "recommendation",
            "patientAdvice",
        ],
        evidence=[
            FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[1])
        ],
    )
    llm = FakeLLM(clean)

    report, _ = run_extraction(_transcript(), llm=llm, confidence_threshold=7)

    assert llm.calls == 1
    assert report.attempts == 1


def test_evidence_citing_missing_segments_is_stripped():
    result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
        ),
        evidence=[
            FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[1, 99])
        ],
    )
    llm = FakeLLM(result)

    report, _ = run_extraction(_transcript(), llm=llm)

    assert report.evidence[0].segmentIds == [1]
