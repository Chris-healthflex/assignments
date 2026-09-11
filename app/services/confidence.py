import json
import logging
from typing import Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
from app.services.extraction import get_llm, clean_json_text, invoke_llm_with_retry


logger = logging.getLogger(__name__)

def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flattens a nested dictionary into dot-notation paths.
    Lists are indexed using brackets like 'recommendation[0].sessionType'.
    """
    items = []
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        elif isinstance(v, list):
            if not v:
                items.append((new_key, v))
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]").items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

CONFIDENCE_SYSTEM_PROMPT = """You are a clinical documentation auditor.
Your job is to compare a list of extracted clinical fields against the raw clinical transcript.
For each extracted field, determine if it is explicitly and accurately mentioned in the transcript.
You must assign a confidence score between 0.0 and 1.0 for each field.

Guidelines:
- 1.0 confidence: The value is directly stated or explicitly supported by the transcript.
- < 0.70 confidence: The value is hallucinated, assumed, guessed, or contains external medical knowledge not stated by the patient/clinician in the transcript.
- If a value is empty or default, it is correct (automatically 1.0).

Return a JSON object where the keys are the EXACT field paths provided, and values contain 'confidence' (float) and 'reason' (string). E.g.:
{
  "clinicalDetails.clinicalHistory": {
    "confidence": 1.0,
    "reason": "Matches the transcript perfectly."
  }
}

Do not wrap the JSON or include any Markdown formatting fences. Start directly with the JSON object.
"""

def evaluate_confidence(transcript: str, assessment_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates confidence score for all fields in the assessment dictionary.
    Populated fields are evaluated by the LLM. Empty/default fields are assigned 1.0.
    """
    flat = flatten_dict(assessment_dict)
    
    # Separate empty vs populated fields
    populated_fields = {}
    confidence_metadata = {}
    
    for path, val in flat.items():
        # Check if empty/default
        is_empty = False
        if val == "" or val == [] or val is None:
            is_empty = True
        
        if is_empty:
            confidence_metadata[path] = {
                "confidence": 1.0,
                "reason": "Field is empty (not mentioned in transcript)"
            }
        else:
            populated_fields[path] = val

    if not populated_fields:
        logger.info("No populated fields to evaluate for confidence.")
        return confidence_metadata

    logger.info(f"Evaluating confidence for {len(populated_fields)} populated fields.")
    
    try:
        llm = get_llm()
        
        user_prompt = f"Transcript:\n{transcript}\n\nPopulated Fields to Audit:\n{json.dumps(populated_fields, indent=2)}"
        
        messages = [
            SystemMessage(content=CONFIDENCE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = invoke_llm_with_retry(llm, messages)

        cleaned_content = clean_json_text(response.content)
        evaluation = json.loads(cleaned_content)
        
        # Merge LLM results back
        for path in populated_fields.keys():
            if path in evaluation:
                conf = evaluation[path].get("confidence", 1.0)
                reason = evaluation[path].get("reason", "")
                confidence_metadata[path] = {
                    "confidence": float(conf),
                    "reason": str(reason)
                }
            else:
                # Default to 0.0 if LLM failed to return an evaluation for a populated field
                confidence_metadata[path] = {
                    "confidence": 0.0,
                    "reason": "Missing confidence audit response"
                }
                
    except Exception as e:
        logger.error(f"Confidence evaluation LLM call failed: {e}")
        # Default populated fields to 0.0 if the LLM audit fails completely
        for path in populated_fields.keys():
            confidence_metadata[path] = {
                "confidence": 0.0,
                "reason": f"Audit failed: {str(e)}"
            }

    return confidence_metadata
