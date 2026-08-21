from app.agent.llm_client import get_llm
from app.agent.prompts.objective_assessment import OBJECTIVE_ASSESSMENT_PROMPT
from app.schemas.extraction import ObjectiveTestExtraction
from app.agent.state import AgentState
from pydantic import BaseModel, Field
from typing import List

class ObjectiveTestsList(BaseModel):
    tests: List[ObjectiveTestExtraction] = Field(default_factory=list)

async def extract_objective_assessment(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = OBJECTIVE_ASSESSMENT_PROMPT | llm.with_structured_output(ObjectiveTestsList)
    out = await chain.ainvoke({"transcript": state["transcript"]})
    state["result"].objectiveAssessment.tests = out.tests
    return state