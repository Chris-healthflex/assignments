from langchain_core.prompts import ChatPromptTemplate
from app.agent.prompts.system_prompt import SYSTEM_PROMPT

SUBJECTIVE_ASSESSMENTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract all subjective assessments as an array. Each item has testName and conclusion."),
    ("human", "{transcript}")
])