import logging
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import settings
from app.graph.state import AssessmentState
from app.services.extraction import run_extraction_pipeline, get_llm, clean_json_text, invoke_llm_with_retry
from app.services.confidence import evaluate_confidence


logger = logging.getLogger(__name__)

# --- Node Implementations ---

def extract_node(state: AssessmentState) -> Dict[str, Any]:
    """
    1. Runs the LLM extraction pipeline with retry mechanism.
    2. Stores raw_extraction and any fallback errors in confidence_metadata.
    """
    logger.info("Entering extract_node")
    raw_ext, fallback_errors = run_extraction_pipeline(state["transcript"])
    
    # Store initial confidence metadata for fields that failed schema validation
    return {
        "raw_extraction": raw_ext,
        "confidence_metadata": fallback_errors
    }

NORMALIZE_SYSTEM_PROMPT = """You are a clinical data normalization assistant.
Your job is to look at the raw extracted clinical JSON and clean up formatting where necessary:
- Format any dates (e.g. 'October 5th, 2026') to 'YYYY-MM-DD' if a specific day is clear, or leave as relative if relative (e.g. '2 weeks').
- Ensure consistent unit naming (e.g. 'degrees', 'cm', 'kg') and spelling.
- Do not invent, alter, or remove clinical details.
- Output ONLY the updated JSON conforming strictly to the FirstAssessment schema.

Start directly with the JSON object. Do not include markdown fences or any introductory text.
"""

def normalize_node(state: AssessmentState) -> Dict[str, Any]:
    """
    Standardizes and cleans up formatting of the extracted data using a light LLM pass.
    """
    logger.info("Entering normalize_node")
    raw_ext = state.get("raw_extraction")
    if not raw_ext:
        return {"normalized_extraction": {}}
        
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=NORMALIZE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(raw_ext))
        ]
        response = invoke_llm_with_retry(llm, messages)

        cleaned = clean_json_text(response.content)
        normalized = json.loads(cleaned)
        
        # Verify it still conforms to structure
        from app.models.assessment import FirstAssessment
        assessment = FirstAssessment.model_validate(normalized)
        logger.info("Normalization successful.")
        return {"normalized_extraction": assessment.model_dump()}
    except Exception as e:
        logger.warning(f"Normalization failed or returned invalid schema: {e}. Using raw extraction.")
        return {"normalized_extraction": raw_ext}

def confidence_node(state: AssessmentState) -> Dict[str, Any]:
    """
    Compares the normalized extraction against the raw transcript to calculate
    field-by-field confidence scores. Merges with any extract-time fallback errors.
    """
    logger.info("Entering confidence_node")
    normalized = state.get("normalized_extraction") or state.get("raw_extraction") or {}
    transcript = state["transcript"]
    
    # Calculate confidence scores via LLM
    computed_conf = evaluate_confidence(transcript, normalized)
    
    # Merge computed confidence with any pre-existing fallback errors (which have confidence 0.0)
    current_conf = state.get("confidence_metadata") or {}
    merged_conf = {**current_conf, **computed_conf}
    
    return {"confidence_metadata": merged_conf}

def validate_node(state: AssessmentState) -> Dict[str, Any]:
    """
    Scans confidence metadata. If any field score falls below CONFIDENCE_THRESHOLD,
    appends it to validation_errors. Sets first_assessment to None on validation failure.
    """
    logger.info("Entering validate_node")
    conf_metadata = state.get("confidence_metadata") or {}
    threshold = settings.confidence_threshold
    
    validation_errors = []
    for field_path, audit in conf_metadata.items():
        score = audit.get("confidence", 1.0)
        reason = audit.get("reason", "No audit details")
        if score < threshold:
            validation_errors.append({
                "field": field_path,
                "reason": reason,
                "confidence": score
            })
            
    if validation_errors:
        logger.warning(f"Validation failed: {len(validation_errors)} fields below threshold {threshold}")
        return {
            "validation_errors": validation_errors,
            "first_assessment": None
        }
    else:
        logger.info("Validation passed successfully.")
        return {
            "validation_errors": [],
            "first_assessment": state.get("normalized_extraction") or state.get("raw_extraction")
        }

# --- State Machine Compilation ---

workflow = StateGraph(AssessmentState)

workflow.add_node("extract_node", extract_node)
workflow.add_node("normalize_node", normalize_node)
workflow.add_node("confidence_node", confidence_node)
workflow.add_node("validate_node", validate_node)

workflow.add_edge(START, "extract_node")
workflow.add_edge("extract_node", "normalize_node")
workflow.add_edge("normalize_node", "confidence_node")
workflow.add_edge("confidence_node", "validate_node")
workflow.add_edge("validate_node", END)

app_graph = workflow.compile()
