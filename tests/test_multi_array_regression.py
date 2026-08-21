import json
from pathlib import Path

from app.models.first_assessment import FirstAssessment


def test_multi_array_fixture_preserves_all_items() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "multi_array_assessment.json"
    assessment = FirstAssessment.model_validate(json.loads(fixture_path.read_text()))
    assert len(assessment.subjectiveGoals) == 5
    assert len(assessment.objectiveGoals) == 5
    assert len(assessment.objectiveAssessment.tests) == 5
