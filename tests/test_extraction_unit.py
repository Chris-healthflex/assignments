from app.services.extraction import (
    extract_clinical_details,
    extract_objective_assessment,
)


TRANSCRIPT = """
The patient presented with left knee pain, difficulty performing
functional activities and difficulty walking along with ankle and back
pain during prolonged walking following surgery. The patient was
apparently normal eight months ago when she was involved in a road
traffic accident resulting in a left tibial condyle fracture and an
avulsion ACL tear. Open reduction and internal fixation was performed.

Objective measurements showed left knee flexion of 124 degrees
compared with 130 degrees on the right. Left knee extension was 20
degrees compared with knee extension of 5 degrees on the right.
Hip internal rotation was 45 degrees bilaterally, hip external
rotation of 60 degrees bilaterally and ankle dorsiflexion of 4.5
degrees on the left compared with 12 degrees on the right.
"""


def test_extract_clinical_details_from_transcript():
    result = extract_clinical_details(
        {"transcript": TRANSCRIPT}
    )

    details = result["clinical_details"]

    assert details.clinicalHistory != "Not specified"
    assert details.chiefComplaint != "Not specified"
    assert details.duration.value == "8"
    assert details.duration.unit == "months"


def test_extract_objective_measurements_from_transcript():
    result = extract_objective_assessment(
        {"transcript": TRANSCRIPT}
    )

    tests = result["objective_assessment"].tests

    assert len(tests) >= 5

    test_names = {test.testName for test in tests}

    assert "Knee flexion" in test_names
    assert "Knee extension" in test_names
    assert "Hip internal rotation" in test_names
    assert "Hip external rotation" in test_names
    assert "Ankle dorsiflexion" in test_names