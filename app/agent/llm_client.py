from langchain_groq import ChatGroq
from app.core.config import settings

def get_llm() -> ChatGroq:
    """Return a Groq LLM instance with temperature 0 and JSON mode if needed."""
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )