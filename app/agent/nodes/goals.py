from app.agent.llm_client import get_llm
from app.agent.prompts.goals import SUBJECTIVE_GOALS_PROMPT, OBJECTIVE_GOALS_PROMPT
from app.schemas.extraction import SubjectiveGoalExtraction, ObjectiveGoalExtraction
from app.agent.state import AgentState
from pydantic import BaseModel, Field
from typing import List

class SubjectiveGoalsList(BaseModel):
    goals: List[SubjectiveGoalExtraction] = Field(default_factory=list)

class ObjectiveGoalsList(BaseModel):
    goals: List[ObjectiveGoalExtraction] = Field(default_factory=list)

async def extract_goals(state: AgentState) -> AgentState:
    llm = get_llm()
    
    # Subjective goals
    chain_subj = SUBJECTIVE_GOALS_PROMPT | llm.with_structured_output(SubjectiveGoalsList)
    subj_out = await chain_subj.ainvoke({"transcript": state["transcript"]})
    state["result"].subjectiveGoals = subj_out.goals

    # Objective goals
    chain_obj = OBJECTIVE_GOALS_PROMPT | llm.with_structured_output(ObjectiveGoalsList)
    obj_out = await chain_obj.ainvoke({"transcript": state["transcript"]})
    state["result"].objectiveGoals = obj_out.goals

    return state