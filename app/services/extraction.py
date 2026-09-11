import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from app.core.config import settings
from app.models.assessment import (
    FirstAssessment, ClinicalDetails, SubjectiveAssessment,
    ObjectiveAssessment, ObjectiveTest, SubjectiveGoal,
    ObjectiveGoal, Recommendation, PatientAdvice
)

logger = logging.getLogger(__name__)

def get_llm():
    """
    Returns the configured LLM client.
    """
    provider = settings.llm_provider.lower().strip()
    logger.info(f"Initializing LLM provider: {provider}")
    
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set for Groq provider.")
        from langchain_groq import ChatGroq
        
        api_key = settings.groq_api_key.strip('"').strip("'")
        base_url = settings.groq_base_url.strip('"').strip("'")
        
        if "api.groq.com" in base_url:
            return ChatGroq(
                api_key=api_key,
                model=settings.llm_model,
                temperature=0.0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        else:
            return ChatGroq(
                api_key=api_key,
                base_url=base_url,
                model=settings.llm_model,
                temperature=0.0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.llm_model,
                temperature=0.0,
                format="json"
            )
        except ImportError:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.llm_model,
                temperature=0.0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def invoke_llm_with_retry(llm, messages, max_retries: int = 5, initial_delay: float = 4.0):
    import time
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate_limit" in err_str or "rate limit" in err_str) and attempt < max_retries - 1:
                logger.warning(f"Rate limit hit (429). Retrying in {delay:.1f}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def clean_json_text(text: str) -> str:
    """
    Cleans raw markdown formatting code fences from a JSON string if present.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def fallback_validate(json_data: dict) -> Tuple[dict, Dict[str, Any]]:
    """
    Validates field-by-field, replacing failing fields with defaults
    and returning the valid dict along with metadata flagging these as low-confidence.
    """
    fallback_errors = {}
    
    def validate_field(model_cls, data, field_name: str, default_factory):
        try:
            return model_cls.model_validate(data).model_dump()
        except Exception as e:
            logger.warning(f"Fallback validation failed for {field_name}: {e}")
            fallback_errors[field_name] = {
                "confidence": 0.0,
                "reason": f"Failed schema validation: {str(e)}"
            }
            return default_factory().model_dump()

    clinical_details = validate_field(
        ClinicalDetails,
        json_data.get("clinicalDetails", {}),
        "clinicalDetails",
        ClinicalDetails
    )
    
    # subjectiveAssessments
    sub_assessments = []
    sub_assessments_raw = json_data.get("subjectiveAssessments")
    if not isinstance(sub_assessments_raw, list):
        fallback_errors["subjectiveAssessments"] = {
            "confidence": 0.0,
            "reason": "Expected list for subjectiveAssessments"
        }
        sub_assessments_raw = []
    for idx, item in enumerate(sub_assessments_raw):
        sub_assessments.append(
            validate_field(
                SubjectiveAssessment,
                item,
                f"subjectiveAssessments[{idx}]",
                SubjectiveAssessment
            )
        )
        
    # objectiveAssessment
    obj_assessment = validate_field(
        ObjectiveAssessment,
        json_data.get("objectiveAssessment", {}),
        "objectiveAssessment",
        ObjectiveAssessment
    )
    
    # subjectiveGoals
    sub_goals = []
    sub_goals_raw = json_data.get("subjectiveGoals")
    if not isinstance(sub_goals_raw, list):
        fallback_errors["subjectiveGoals"] = {
            "confidence": 0.0,
            "reason": "Expected list for subjectiveGoals"
        }
        sub_goals_raw = []
    for idx, item in enumerate(sub_goals_raw):
        sub_goals.append(
            validate_field(
                SubjectiveGoal,
                item,
                f"subjectiveGoals[{idx}]",
                SubjectiveGoal
            )
        )
        
    # objectiveGoals
    obj_goals = []
    obj_goals_raw = json_data.get("objectiveGoals")
    if not isinstance(obj_goals_raw, list):
        fallback_errors["objectiveGoals"] = {
            "confidence": 0.0,
            "reason": "Expected list for objectiveGoals"
        }
        obj_goals_raw = []
    for idx, item in enumerate(obj_goals_raw):
        obj_goals.append(
            validate_field(
                ObjectiveGoal,
                item,
                f"objectiveGoals[{idx}]",
                ObjectiveGoal
            )
        )
        
    # recommendation
    recommendation = []
    rec_raw = json_data.get("recommendation")
    if not isinstance(rec_raw, list):
        fallback_errors["recommendation"] = {
            "confidence": 0.0,
            "reason": "Expected list for recommendation"
        }
        rec_raw = []
    for idx, item in enumerate(rec_raw):
        recommendation.append(
            validate_field(
                Recommendation,
                item,
                f"recommendation[{idx}]",
                Recommendation
            )
        )
        
    # patientAdvice
    patient_advice = validate_field(
        PatientAdvice,
        json_data.get("patientAdvice", {}),
        "patientAdvice",
        PatientAdvice
    )
    
    valid_dict = {
        "clinicalDetails": clinical_details,
        "subjectiveAssessments": sub_assessments,
        "objectiveAssessment": obj_assessment,
        "subjectiveGoals": sub_goals,
        "objectiveGoals": obj_goals,
        "recommendation": recommendation,
        "patientAdvice": patient_advice
    }
    
    return valid_dict, fallback_errors

EXTRACTION_SYSTEM_PROMPT = """You are a clinical assessment data extraction assistant.
Your task is to extract structured details from the provided clinical transcript and return them strictly in JSON format conforming to the Pydantic schema for `FirstAssessment`.

Strict Invariants you MUST follow:
1. Extract ONLY information explicitly stated in the transcript.
2. NEVER invent a diagnosis, score, date, or measurement not present in the transcript.
3. Preserve numbers, units, and left/right sidedness exactly as spoken (e.g. do not convert 120 degrees to radians or normalize it).
4. Use empty strings ("") or empty lists ([]) for any schema fields not explicitly stated.
5. NEVER use null values.
6. The JSON must exactly match the schema structure. Do not include any extra fields or wrapper keys other than the FirstAssessment fields.

Here is the FirstAssessment Pydantic Schema representation for your guidance:
- clinicalDetails: { clinicalHistory: str, chiefComplaint: str, duration: str }
- subjectiveAssessments: List of { testName: str, conclusion: str }
- objectiveAssessment: { tests: List of { testName: str, unitName: str, value: str, left: str, right: str, comments: str } }
- subjectiveGoals: List of { goalDetails: str, targetDate: str }
- objectiveGoals: List of { goalName: str, goalCategory: str, unitName: str, value: str, targetDate: str }
- recommendation: List of { sessionType: str, sessionFrequency: str }
- patientAdvice: { adviceDetails: str }

Ensure your response is valid, parsable JSON. Start directly with the JSON object. Do not include any markdown fences or introductory text.
"""

def run_extraction_pipeline(transcript: str) -> Tuple[dict, Dict[str, Any]]:
    """
    Runs the LLM extraction with up to 3 retry attempts on Pydantic validation failures.
    Returns (extracted_dict, fallback_errors_dict).
    """
    llm = get_llm()
    attempts = 3
    feedback_msg = ""
    
    for attempt in range(attempts):
        logger.info(f"LLM extraction attempt {attempt + 1} of {attempts}")
        
        user_prompt = f"Transcript:\n{transcript}\n"
        if feedback_msg:
            user_prompt += f"\n{feedback_msg}"
            
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = invoke_llm_with_retry(llm, messages)
            cleaned_content = clean_json_text(response.content)

            data = json.loads(cleaned_content)
            
            # Validate completely using FirstAssessment
            assessment = FirstAssessment.model_validate(data)
            logger.info("Successfully validated FirstAssessment output on LLM attempt.")
            return assessment.model_dump(), {}
            
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                # Last attempt failed, parse what we can and return fallback dict
                logger.error("All extraction retries failed. Falling back to field-by-field parsing.")
                try:
                    data = json.loads(clean_json_text(response.content)) if 'response' in locals() else {}
                except Exception:
                    data = {}
                return fallback_validate(data)
            
            # Formulate feedback for next try
            feedback_msg = f"Your previous attempt was invalid. Error details:\n{str(e)}\n\nPlease correct this and return clean, schema-compliant JSON."
            
    # Absolute fallback block
    return fallback_validate({})
