from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging import logger

async def aggregate(state: AgentState) -> AgentState:
    """Validate confidence and set retry flag."""
    threshold = settings.CONFIDENCE_THRESHOLD
    errors = []

    def walk(obj, prefix=""):
        from app.schemas.extraction import ExtractionField
        if isinstance(obj, ExtractionField):
            if obj.is_mentioned and obj.value is not None and obj.confidence < threshold:
                errors.append({
                    "field": prefix,
                    "confidence": round(obj.confidence, 3),
                    "reason": f"Confidence below threshold ({threshold})"
                })
        elif hasattr(obj, "__dict__"):   # Pydantic BaseModel
            for name, value in obj.__dict__.items():
                if name.startswith("_"):
                    continue
                walk(value, f"{prefix}.{name}" if prefix else name)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{prefix}[{i}]")

    walk(state["result"])
    state["section_errors"] = errors

    # Retry if errors and we haven't hit max retries
    if errors and state["retry_count"] < settings.MAX_RETRIES:
        state["retry_needed"] = True
        state["retry_count"] += 1
        logger.info(f"Retry triggered: count={state['retry_count']}, errors={len(errors)}")
    else:
        state["retry_needed"] = False

    return state