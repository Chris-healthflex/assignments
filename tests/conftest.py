import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.extraction import AssessmentDraft, Verification

TRANSCRIPT = (
    "The patient presented with left knee pain and difficulty walking following "
    "surgery eight months ago after a road traffic accident. Objective "
    "measurements showed left knee flexion of 124 degrees compared with 130 "
    "degrees on the right. Physiotherapy was recommended once weekly for four "
    "sessions."
)

DRAFT = AssessmentDraft.model_validate(
    {
        "clinicalDetails": {
            "clinicalHistory": "Road traffic accident eight months ago, then surgery.",
            "chiefComplaint": "Left knee pain and difficulty walking",
            "duration": "Eight months",
        },
        "subjectiveAssessments": [
            {"testName": "Pain report", "conclusion": "Left knee pain when walking"}
        ],
        "objectiveTests": [
            {
                "testName": "Knee flexion",
                "unitName": "degrees",
                "left": "124",
                "right": "130",
            }
        ],
        "subjectiveGoals": [{"goalDetails": "Walk without pain"}],
        "recommendation": [
            {
                "sessionType": "Physiotherapy",
                "sessionFrequency": "Once weekly for four sessions",
            }
        ],
    }
)


class StubLLM:
    """Returns canned structured objects instead of calling a model.

    The workflow only uses ``with_structured_output``, so this keeps the tests
    offline while still running the real LangGraph nodes and mapping.
    """

    def __init__(
        self,
        draft: Any = None,
        confidence: float = 0.9,
        unsupported: Optional[List[str]] = None,
        notes: str = "",
    ):
        self.draft = draft if draft is not None else DRAFT.model_copy(deep=True)
        self.verification = Verification(
            confidence=confidence, unsupportedFields=unsupported or [], notes=notes
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        value = self.draft if schema is AssessmentDraft else self.verification

        def run(_: Any) -> Any:
            if isinstance(value, Exception):
                raise value
            return value

        return RunnableLambda(run)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        google_api_key="test-key",
        mongodb_database="clinical_test",
        extraction_confidence_threshold=0.6,
        _env_file=None,
    )


@pytest.fixture
def wav_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "clinical_assessment.wav"
    if not path.is_file():
        pytest.skip("clinical_assessment.wav is not available")
    return path
