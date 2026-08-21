from langchain_core.prompts import ChatPromptTemplate
from app.agent.prompts.system_prompt import SYSTEM_PROMPT

OBJECTIVE_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract the objective assessment tests as an array. Each test has testName, unitName, value, left, right, comments."),
    ("human", "{transcript}")
])