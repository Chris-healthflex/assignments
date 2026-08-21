from app.agent.llm_client import get_llm
from app.agent.prompts.clinical_details import CLINICAL_DETAILS_PROMPT
from app.schemas.extraction import ClinicalDetailsExtraction
from app.agent.state import AgentState

async def extract_clinical_details(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = CLINICAL_DETAILS_PROMPT | llm.with_structured_output(ClinicalDetailsExtraction)
    out = await chain.ainvoke({"transcript": state["transcript"]})
    state["result"].clinicalDetails = out
    return state