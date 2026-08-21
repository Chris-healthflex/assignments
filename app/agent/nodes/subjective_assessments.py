from app.agent.llm_client import get_llm
from app.agent.prompts.subjective_assessments import SUBJECTIVE_ASSESSMENTS_PROMPT
from app.schemas.extraction import SubjectiveAssessmentExtraction
from app.agent.state import AgentState
from pydantic import BaseModel, Field
from typing import List

class SubjectiveAssessmentsList(BaseModel):
    assessments: List[SubjectiveAssessmentExtraction] = Field(default_factory=list)

async def extract_subjective_assessments(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = SUBJECTIVE_ASSESSMENTS_PROMPT | llm.with_structured_output(SubjectiveAssessmentsList)
    out = await chain.ainvoke({"transcript": state["transcript"]})
    state["result"].subjectiveAssessments = out.assessments
    return state