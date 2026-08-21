from langchain_core.prompts import ChatPromptTemplate
from app.agent.prompts.system_prompt import SYSTEM_PROMPT

# For subjective goals
SUBJECTIVE_GOALS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract subjective goals as an array. Each goal has goalDetails and targetDate."),
    ("human", "{transcript}")
])

# For objective goals
OBJECTIVE_GOALS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\nExtract objective goals as an array. Each goal has goalName, goalCategory, unitName, value, targetDate."),
    ("human", "{transcript}")
])