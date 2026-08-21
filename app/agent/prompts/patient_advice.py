from langchain_core.prompts import ChatPromptTemplate
from app.agent.prompts.system_prompt import SYSTEM_PROMPT

PATIENT_ADVICE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract the patient advice section (adviceDetails)."),
    ("human", "{transcript}")
])