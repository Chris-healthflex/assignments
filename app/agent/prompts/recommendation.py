from langchain_core.prompts import ChatPromptTemplate
from app.agent.prompts.system_prompt import SYSTEM_PROMPT

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract recommendations as an array. Each item has sessionType and sessionFrequency."),
    ("human", "{transcript}")
])