from langchain_core.prompts import ChatPromptTemplate
from app.agent.prompts.system_prompt import SYSTEM_PROMPT

CLINICAL_DETAILS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract the clinical details section."),
    ("human", "{transcript}")
])