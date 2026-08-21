from app.agent.llm_client import get_llm
from app.agent.prompts.patient_advice import PATIENT_ADVICE_PROMPT
from app.schemas.extraction import PatientAdviceExtraction
from app.agent.state import AgentState

async def extract_patient_advice(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = PATIENT_ADVICE_PROMPT | llm.with_structured_output(PatientAdviceExtraction)
    out = await chain.ainvoke({"transcript": state["transcript"]})
    state["result"].patientAdvice = out
    return state