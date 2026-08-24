"""Grounding blanks hallucinated values but keeps stated ones."""
from app.schemas.assessment import (
    FirstAssessment, ObjectiveAssessment, ObjectiveTest, ClinicalDetails,
)
from app.extraction.grounding import verify


def test_hallucinated_number_is_blanked():
    a = FirstAssessment(objectiveAssessment=ObjectiveAssessment(
        tests=[ObjectiveTest(testName="Knee flexion", left="124", right="999")]))
    transcript = "left knee flexion of 124 degrees compared with 130 on the right"
    report = verify(a, transcript)
    t = a.objectiveAssessment.tests[0]
    assert t.left == "124"       # grounded, kept
    assert t.right == ""         # 999 not in transcript -> blanked
    assert any(f["path"].endswith(".right") for f in report.ungrounded)


def test_grounded_text_survives():
    a = FirstAssessment(clinicalDetails=ClinicalDetails(duration="eight months"))
    report = verify(a, "the patient was normal eight months ago")
    assert a.clinicalDetails.duration == "eight months"
    assert report.ungrounded == []


def test_fabricated_text_flagged():
    a = FirstAssessment(clinicalDetails=ClinicalDetails(
        clinicalHistory="patient has a documented penicillin allergy and prior cardiac surgery"))
    report = verify(a, "the patient injured a knee in an accident")
    assert a.clinicalDetails.clinicalHistory == ""
    assert report.ungrounded


# --- measurement coverage: the four rubric defects ---------------------------
from app.schemas.assessment import ObjectiveTest as _OT, ObjectiveAssessment as _OA

_TRANSCRIPT_MEAS = (
    "left knee flexion of 124 degrees compared with 130 degrees on the right, "
    "left knee extension of 20 degrees compared with 5 degrees on the right, "
    "hip internal rotation of 45 degrees bilaterally, "
    "hip external rotation of 60 degrees bilaterally and "
    "ankle dorsiflexion of 4.5 degrees on the left compared with 12 degrees on the right."
)


def _correct_table():
    return FirstAssessment(objectiveAssessment=_OA(tests=[
        _OT(testName="Knee flexion", left="124", right="130"),
        _OT(testName="Knee extension", left="20", right="5"),
        _OT(testName="Hip internal rotation", left="45", right="45"),
        _OT(testName="Hip external rotation", left="60", right="60"),
        _OT(testName="Ankle dorsiflexion", left="4.5", right="12"),
    ]))


def _broken_table():
    # the other candidate's shape: 8 rows, split sides, dup rows, missing 60
    return FirstAssessment(objectiveAssessment=_OA(tests=[
        _OT(testName="Left knee flexion", left="124"),
        _OT(testName="Right knee flexion", right="130"),
        _OT(testName="Left knee extension", left="20", right="5"),
        _OT(testName="Right knee extension", left="5"),
        _OT(testName="Left hip internal rotation", left="45", right="45"),
        _OT(testName="Right hip internal rotation", left="45"),
        _OT(testName="Left ankle dorsiflexion", left="4.5"),
        _OT(testName="Right ankle dorsiflexion", right="12"),
    ]))


def test_correct_table_full_coverage():
    a = _correct_table()
    r = verify(a, _TRANSCRIPT_MEAS)
    assert r.coverage["objectiveAssessment"] == 1.0
    reasons = {f["reason"] for f in r.ungrounded}
    assert "missing_measurement" not in reasons
    assert "count_mismatch" not in reasons


def test_broken_table_flags_all_four_defects():
    a = _broken_table()
    r = verify(a, _TRANSCRIPT_MEAS)
    reasons = [f["reason"] for f in r.ungrounded]
    # count wrong (8 vs 5)
    assert "count_mismatch" in reasons
    # missing hip external rotation (60 degrees)
    assert any(f["reason"] == "missing_measurement" and "60" in f["detail"] for f in r.ungrounded)
    # confidence must NOT be 1.0 on a broken table
    assert r.coverage["objectiveAssessment"] < 1.0
