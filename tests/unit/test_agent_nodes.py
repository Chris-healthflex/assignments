import pytest
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableLambda

from app.agent.nodes.clinical_details import extract_clinical_details
from app.agent.state import AgentState
from app.schemas.extraction import ExtractionResult, ExtractionField, ClinicalDetailsExtraction


@pytest.mark.asyncio
async def test_extract_clinical_details_mock():
    mock_out = ClinicalDetailsExtraction(
        clinicalHistory=ExtractionField(
            value="chronic pain",
            is_mentioned=True,
            confidence=0.9,
            source_quote="pain for years",
        ),
        chiefComplaint=ExtractionField(
            value="knee pain",
            is_mentioned=True,
            confidence=0.95,
            source_quote="knee hurts",
        ),
        duration=ExtractionField(
            value="3 months",
            is_mentioned=True,
            confidence=0.8,
            source_quote="three months",
        ),
    )

    async def fake_chain(_input):
        return mock_out

    runnable = RunnableLambda(fake_chain)

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = runnable

    with patch("app.agent.nodes.clinical_details.get_llm", return_value=mock_llm):
        state = AgentState(
            transcript="some transcript",
            result=ExtractionResult(),
            retry_count=0,
            section_errors=[],
            retry_needed=False,
        )
        new_state = await extract_clinical_details(state)

    assert new_state["result"].clinicalDetails.chiefComplaint.value == "knee pain"