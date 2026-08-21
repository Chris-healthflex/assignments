from app.agent.llm_client import get_llm
from app.agent.prompts.recommendation import RECOMMENDATION_PROMPT
from app.schemas.extraction import RecommendationExtraction
from app.agent.state import AgentState
from pydantic import BaseModel, Field
from typing import List

class RecommendationList(BaseModel):
    recommendations: List[RecommendationExtraction] = Field(default_factory=list)

async def extract_recommendation(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = RECOMMENDATION_PROMPT | llm.with_structured_output(RecommendationList)
    out = await chain.ainvoke({"transcript": state["transcript"]})
    state["result"].recommendation = out.recommendations
    return state