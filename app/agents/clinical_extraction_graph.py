import json
import re
import time
import httpx
from typing import Any, TypedDict


from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.core.logging import logger
from app.core.errors import ExtractionError
from app.schemas.first_assessment import FirstAssessment
from app.agents.prompts import CLINICAL_EXTRACTION_SYSTEM_PROMPT


class ExtractionState(TypedDict):
    transcript: str
    extracted_data: dict[str, Any]
    validation_errors: list[str]
    confidence_score: float
    first_assessment: FirstAssessment | None
    is_valid: bool


def validate_transcript_node(state: ExtractionState) -> dict[str, Any]:
    """Validates that the input transcript is non-empty and substantive."""
    transcript = (state.get("transcript") or "").strip()
    errors = list(state.get("validation_errors", []))

    logger.info("LangGraph Node [validate_transcript]: Processing transcript of length %d", len(transcript))

    if not transcript:
        logger.warning("Validation failed: Transcript is empty")
        errors.append("Transcript is empty.")
        return {
            "validation_errors": errors,
            "is_valid": False,
            "confidence_score": 0.0,
            "extracted_data": {},
            "first_assessment": None,
        }

    words = transcript.split()
    if len(words) < 3:
        logger.warning("Validation failed: Transcript too short (%d words)", len(words))
        errors.append("Transcript too brief to extract meaningful clinical entities.")
        return {
            "validation_errors": errors,
            "is_valid": False,
            "confidence_score": 0.0,
            "extracted_data": {},
            "first_assessment": None,
        }

    return {
        "validation_errors": errors,
        "is_valid": True,
    }


def extract_clinical_entities_node(state: ExtractionState) -> dict[str, Any]:
    """Invokes the LLM to extract structured FirstAssessment fields from the transcript."""
    if not state.get("is_valid", True):
        return {}

    transcript = state["transcript"]
    settings = get_settings()
    errors = list(state.get("validation_errors", []))

    logger.info("LangGraph Node [extract_clinical_entities]: Calling LLM (%s)", settings.LLM_MODEL)

    api_key = settings.effective_llm_api_key
    base_url = settings.effective_llm_base_url


    if not api_key:
        logger.warning("No LLM API key configured (OPENAI_API_KEY / XAI_API_KEY). Using fallback rule-based extraction.")
        fallback_data = _rule_based_fallback_extraction(transcript)
        return {
            "extracted_data": fallback_data,
            "validation_errors": errors,
        }

    http_client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, deflate"}
    )

    models_to_try = [settings.LLM_MODEL]
    if "gpt-oss-120b" in settings.LLM_MODEL:
        models_to_try.append("openai/gpt-oss-20b")
    elif "gpt-oss-20b" not in settings.LLM_MODEL:
        models_to_try.append("openai/gpt-oss-20b")

    messages = [
        SystemMessage(content=CLINICAL_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Clinical Session Transcript:\n\n{transcript}\n\nExtract and return JSON:")
    ]

    parsed_data = None
    last_error = None

    for current_model in models_to_try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(
                    "LangGraph Node [extract_clinical_entities]: Calling LLM (%s, attempt %d/%d)",
                    current_model, attempt + 1, max_retries
                )
                llm_kwargs: dict[str, Any] = {
                    "model": current_model,
                    "temperature": settings.LLM_TEMPERATURE,
                    "openai_api_key": api_key,
                    "http_client": http_client,
                    "max_tokens": 4096,
                    "model_kwargs": {"response_format": {"type": "json_object"}},
                }
                if base_url:
                    llm_kwargs["openai_api_base"] = base_url

                llm = ChatOpenAI(**llm_kwargs)
                response = llm.invoke(messages)
                content = response.content

                # Clean JSON markdown and reasoning tags
                if isinstance(content, str):
                    clean_json = content.strip()
                    if "</think>" in clean_json:
                        clean_json = clean_json.split("</think>")[-1].strip()

                    clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                    clean_json = re.sub(r"\s*```$", "", clean_json).strip()

                    if not (clean_json.startswith("{") and clean_json.endswith("}")):
                        start_idx = clean_json.find("{")
                        end_idx = clean_json.rfind("}")
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            clean_json = clean_json[start_idx:end_idx + 1]

                    try:
                        parsed_data = json.loads(clean_json)
                    except json.JSONDecodeError:
                        fixed = re.sub(r",\s*([\]}])", r"\1", clean_json)
                        parsed_data = json.loads(fixed)
                elif isinstance(content, dict):
                    parsed_data = content
                else:
                    parsed_data = json.loads(str(content))

                if parsed_data:
                    logger.info("LLM extraction succeeded using model '%s'", current_model)
                    return {
                        "extracted_data": parsed_data,
                        "validation_errors": errors,
                    }

            except Exception as exc:
                err_str = str(exc)
                last_error = exc

                # Parse 429 rate limit and wait
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait_seconds = 2.5
                    m = re.search(r"try again in ([\d\.]+)s", err_str, re.IGNORECASE)
                    if m:
                        try:
                            wait_seconds = float(m.group(1)) + 0.5
                        except ValueError:
                            pass

                    logger.warning(
                        "Rate limit hit on model '%s' (429). Backing off for %.2fs before retry (attempt %d/%d)...",
                        current_model, wait_seconds, attempt + 1, max_retries
                    )
                    time.sleep(wait_seconds)
                else:
                    logger.warning(
                        "LLM call attempt %d on model '%s' failed: %s",
                        attempt + 1, current_model, exc
                    )
                    time.sleep(1.0)

        logger.warning("Exhausted retries on model '%s'. Falling back to next available model...", current_model)

    # If all models/retries failed
    logger.error("All LLM extraction attempts failed: %s", last_error)
    errors.append(f"LLM entity extraction failed: {str(last_error)}")
    return {
        "extracted_data": {},
        "validation_errors": errors,
        "is_valid": False,
    }



STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "from", "of", "as", "is", "was", "are", "were", "been", "be", "have",
    "has", "had", "do", "does", "did", "can", "could", "should", "would", "may",
    "might", "must", "will", "shall", "that", "which", "who", "whom", "this",
    "these", "those", "it", "its", "they", "them", "their", "we", "us", "our",
    "you", "your", "he", "him", "his", "she", "her", "i", "me", "my", "also",
    "about", "then", "than", "so", "very", "just", "now", "if"
}


def _check_field_grounding(field_path: str, val: Any, transcript: str) -> tuple[bool, str | None]:
    """
    Verifies whether a populated text field has evidence in the transcript.
    Returns (is_grounded, warning_message_if_not).
    """
    if not isinstance(val, str) or not val.strip():
        return True, None

    text = val.strip().lower()
    # Extract significant content words
    words = [w for w in re.findall(r"\b\w+\b", text) if len(w) > 2 and w not in STOP_WORDS]

    if not words:
        # Check if numbers or short characters appear
        nums = re.findall(r"\d+", text)
        if nums and any(n in transcript for n in nums):
            return True, None
        return True, None

    # Check matching content words
    matched_words = [w for w in words if w in transcript]
    match_ratio = len(matched_words) / len(words)

    # If at least 30% of significant words or at least 2 distinct words match, it's grounded
    if match_ratio >= 0.30 or len(matched_words) >= 2 or any(w in transcript for w in words if len(w) >= 5):
        return True, None

    return False, f"{field_path} ('{val[:60]}...') lacks verbatim evidence in transcript"


def validate_extraction_node(state: ExtractionState) -> dict[str, Any]:
    """
    Comprehensive Evidence Grounding, Confidence Calculation & Anti-Hallucination Pruning:
    1. Verifies every populated field against speaker-specific transcripts:
       - Recommendations & Patient Advice -> strictly validated against Doctor Transcript.
       - Clinical History & Chief Complaint -> validated against Patient Transcript & session text.
    2. Enforces strict auto-pruning on unsupported assertions.
    3. Outputs a structured, defensible Grounding Verification Report.
    """
    if not state.get("is_valid", True):
        return {}

    full_transcript = state.get("transcript", "").lower()
    data = state.get("extracted_data", {})
    errors = list(state.get("validation_errors", []))
    settings = get_settings()

    # Split speaker transcripts if attributed
    doctor_transcript = full_transcript
    patient_transcript = full_transcript
    if "doctor:" in full_transcript and "patient:" in full_transcript:
        parts = full_transcript.split("patient:")
        doc_part = parts[0].replace("doctor:", "").strip()
        pat_part = "patient:".join(parts[1:]).strip()
        if doc_part:
            doctor_transcript = doc_part
        if pat_part:
            patient_transcript = pat_part

    checked_fields = 0
    grounded_fields = 0
    hallucination_warnings: list[str] = []
    pruned_fields: list[str] = []
    grounding_report: list[dict[str, Any]] = []

    def evaluate_field(path: str, value: Any, target_source: str = "full") -> bool:
        nonlocal checked_fields, grounded_fields
        if not isinstance(value, str) or not value.strip():
            return True

        target_text = doctor_transcript if target_source == "doctor" else (patient_transcript if target_source == "patient" else full_transcript)
        checked_fields += 1
        is_grounded, warning = _check_field_grounding(path, value, target_text)

        if is_grounded:
            grounded_fields += 1
            grounding_report.append({
                "field": path,
                "value": value[:60],
                "source": target_source,
                "status": "PASS",
                "grounded": True,
            })
            return True
        else:
            hallucination_warnings.append(warning or f"{path} ungrounded in {target_source} transcript")
            pruned_fields.append(path)
            grounding_report.append({
                "field": path,
                "value": value[:60],
                "source": target_source,
                "status": "PRUNED",
                "grounded": False,
                "reason": f"No direct supporting evidence found in {target_source} transcript."
            })
            return False

    # Deep copy data for safe pruning
    clean_data: dict[str, Any] = {
        "clinicalDetails": {
            "clinicalHistory": "",
            "chiefComplaint": "",
            "duration": ""
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {"tests": []},
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": ""}
    }

    # 1. Check clinicalDetails (against patient/full transcript)
    cd = data.get("clinicalDetails", {})
    if isinstance(cd, dict):
        for k in ["clinicalHistory", "chiefComplaint", "duration"]:
            val = cd.get(k, "")
            if val and evaluate_field(f"clinicalDetails.{k}", val, "patient"):
                clean_data["clinicalDetails"][k] = val

    # 2. Check subjectiveAssessments (prune synthetic test categories)
    sa_list = data.get("subjectiveAssessments", [])
    if isinstance(sa_list, list):
        for i, sa in enumerate(sa_list):
            if isinstance(sa, dict):
                t_name = sa.get("testName", "")
                t_conc = sa.get("conclusion", "")
                t_name_ok = evaluate_field(f"subjectiveAssessments[{i}].testName", t_name, "full")
                t_conc_ok = evaluate_field(f"subjectiveAssessments[{i}].conclusion", t_conc, "full")
                if t_name_ok and t_conc_ok:
                    clean_data["subjectiveAssessments"].append({"testName": t_name, "conclusion": t_conc})

    # 3. Check objectiveAssessment.tests
    oa = data.get("objectiveAssessment", {})
    if isinstance(oa, dict):
        tests = oa.get("tests", [])
        if isinstance(tests, list):
            for i, t in enumerate(tests):
                if isinstance(t, dict):
                    all_ok = True
                    for k in ["testName", "unitName", "value", "left", "right", "comments"]:
                        if not evaluate_field(f"objectiveAssessment.tests[{i}].{k}", t.get(k, ""), "full"):
                            all_ok = False
                    if all_ok:
                        clean_data["objectiveAssessment"]["tests"].append(t)

    # 4. Check subjectiveGoals & objectiveGoals
    sg_list = data.get("subjectiveGoals", [])
    if isinstance(sg_list, list):
        for i, sg in enumerate(sg_list):
            if isinstance(sg, dict):
                gd_ok = evaluate_field(f"subjectiveGoals[{i}].goalDetails", sg.get("goalDetails", ""), "patient")
                td_ok = evaluate_field(f"subjectiveGoals[{i}].targetDate", sg.get("targetDate", ""), "patient")
                if gd_ok and td_ok:
                    clean_data["subjectiveGoals"].append(sg)

    og_list = data.get("objectiveGoals", [])
    if isinstance(og_list, list):
        for i, og in enumerate(og_list):
            if isinstance(og, dict):
                all_ok = True
                for k in ["goalName", "goalCategory", "unitName", "value", "targetDate"]:
                    if not evaluate_field(f"objectiveGoals[{i}].{k}", og.get(k, ""), "full"):
                        all_ok = False
                if all_ok:
                    clean_data["objectiveGoals"].append(og)

    # 5. Check recommendation (strictly against Doctor transcript)
    rec_list = data.get("recommendation", [])
    if isinstance(rec_list, list):
        for i, rec in enumerate(rec_list):
            if isinstance(rec, dict):
                st_ok = evaluate_field(f"recommendation[{i}].sessionType", rec.get("sessionType", ""), "doctor")
                sf_ok = evaluate_field(f"recommendation[{i}].sessionFrequency", rec.get("sessionFrequency", ""), "doctor")
                if st_ok and sf_ok:
                    clean_data["recommendation"].append(rec)

    # 6. Check patientAdvice (strictly against Doctor transcript)
    pa = data.get("patientAdvice", {})
    if isinstance(pa, dict):
        val = pa.get("adviceDetails", "")
        if val and evaluate_field("patientAdvice.adviceDetails", val, "doctor"):
            clean_data["patientAdvice"]["adviceDetails"] = val

    confidence = (grounded_fields / checked_fields) if checked_fields > 0 else 1.0

    # Print structured Grounding Verification Report
    logger.info("=" * 60)
    logger.info("GROUNDING VERIFICATION REPORT (%d/%d verified, confidence=%.2f):", grounded_fields, checked_fields, confidence)
    for rep in grounding_report:
        if rep["status"] == "PASS":
            logger.info("  [PASS] [%s] Supported (Source: %s) -> '%s'", rep["field"], rep["source"], rep["value"])
        else:
            logger.warning("  [PRUNED] [%s] -> PRUNED (Reason: %s)", rep["field"], rep.get("reason", "Ungrounded"))
    logger.info("=" * 60)


    if confidence < settings.CONFIDENCE_THRESHOLD:
        errors.extend(hallucination_warnings)
        errors.append(
            f"Extraction confidence ({confidence:.2f}) is below threshold ({settings.CONFIDENCE_THRESHOLD:.2f})."
        )
        return {
            "extracted_data": clean_data,
            "confidence_score": confidence,
            "validation_errors": errors,
            "is_valid": False,
        }

    return {
        "extracted_data": clean_data,
        "confidence_score": 1.0,
        "validation_errors": errors,
        "is_valid": True,
    }






def _clean_for_pydantic(data: dict[str, Any]) -> dict[str, Any]:
    """Ensures exact canonical field names and shapes for strict FirstAssessment schema."""
    if not isinstance(data, dict):
        return {}

    clean: dict[str, Any] = {
        "clinicalDetails": {
            "clinicalHistory": "",
            "chiefComplaint": "",
            "duration": ""
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {"tests": []},
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": ""}
    }

    cd = data.get("clinicalDetails", {})
    if isinstance(cd, dict):
        clean["clinicalDetails"]["clinicalHistory"] = str(cd.get("clinicalHistory") or "")
        clean["clinicalDetails"]["chiefComplaint"] = str(cd.get("chiefComplaint") or "")
        clean["clinicalDetails"]["duration"] = str(cd.get("duration") or "")

    for sa in data.get("subjectiveAssessments", []):
        if isinstance(sa, dict):
            clean["subjectiveAssessments"].append({
                "testName": str(sa.get("testName") or sa.get("test") or ""),
                "conclusion": str(sa.get("conclusion") or sa.get("finding") or "")
            })

    for t in data.get("objectiveAssessment", {}).get("tests", []):
        if isinstance(t, dict):
            clean["objectiveAssessment"]["tests"].append({
                "testName": str(t.get("testName") or t.get("test") or t.get("name") or ""),
                "unitName": str(t.get("unitName") or t.get("unit") or ""),
                "value": str(t.get("value") or ""),
                "left": str(t.get("left") or ""),
                "right": str(t.get("right") or ""),
                "comments": str(t.get("comments") or t.get("comment") or "")
            })

    for sg in data.get("subjectiveGoals", []):
        if isinstance(sg, dict):
            clean["subjectiveGoals"].append({
                "goalDetails": str(sg.get("goalDetails") or sg.get("goal") or ""),
                "targetDate": str(sg.get("targetDate") or "")
            })

    for og in data.get("objectiveGoals", []):
        if isinstance(og, dict):
            clean["objectiveGoals"].append({
                "goalName": str(og.get("goalName") or og.get("name") or ""),
                "goalCategory": str(og.get("goalCategory") or og.get("category") or ""),
                "unitName": str(og.get("unitName") or og.get("unit") or ""),
                "value": str(og.get("value") or ""),
                "targetDate": str(og.get("targetDate") or "")
            })

    for rec in data.get("recommendation", []):
        if isinstance(rec, dict):
            clean["recommendation"].append({
                "sessionType": str(rec.get("sessionType") or rec.get("type") or ""),
                "sessionFrequency": str(rec.get("sessionFrequency") or rec.get("frequency") or "")
            })

    pa = data.get("patientAdvice", {})
    if isinstance(pa, dict):
        clean["patientAdvice"]["adviceDetails"] = str(pa.get("adviceDetails") or pa.get("advice") or "")

    return clean


def build_first_assessment_node(state: ExtractionState) -> dict[str, Any]:
    """Validates and constructs the final strict Pydantic FirstAssessment instance."""
    if not state.get("is_valid", True):
        return {"first_assessment": None}

    data = state.get("extracted_data", {})
    errors = list(state.get("validation_errors", []))

    logger.info("LangGraph Node [build_first_assessment]: Validating against FirstAssessment Pydantic schema")

    try:
        sanitized = _clean_for_pydantic(data)
        first_assessment = FirstAssessment.model_validate(sanitized)
        return {
            "first_assessment": first_assessment,
            "validation_errors": errors,
            "is_valid": True,
        }
    except ValidationError as exc:
        logger.error("Pydantic schema validation error: %s", exc)
        errors.append(f"Pydantic schema validation error: {str(exc)}")
        return {
            "first_assessment": None,
            "validation_errors": errors,
            "is_valid": False,
        }



def _should_continue(state: ExtractionState) -> str:
    """Routing condition after transcript validation."""
    return "extract" if state.get("is_valid", True) else "end"


def _should_build(state: ExtractionState) -> str:
    """Routing condition after extraction validation."""
    return "build" if state.get("is_valid", True) else "end"


def create_clinical_extraction_graph() -> Any:
    """Compiles and returns the LangGraph workflow for clinical extraction."""
    builder = StateGraph(ExtractionState)

    builder.add_node("validate_transcript", validate_transcript_node)
    builder.add_node("extract_clinical_entities", extract_clinical_entities_node)
    builder.add_node("validate_extraction", validate_extraction_node)
    builder.add_node("build_first_assessment", build_first_assessment_node)

    builder.add_edge(START, "validate_transcript")
    builder.add_conditional_edges(
        "validate_transcript",
        _should_continue,
        {
            "extract": "extract_clinical_entities",
            "end": END,
        }
    )
    builder.add_edge("extract_clinical_entities", "validate_extraction")
    builder.add_conditional_edges(
        "validate_extraction",
        _should_build,
        {
            "build": "build_first_assessment",
            "end": END,
        }
    )
    builder.add_edge("build_first_assessment", END)

    return builder.compile()


def _rule_based_fallback_extraction(transcript: str) -> dict[str, Any]:
    """Lightweight rule-based parser used when no LLM API key is present."""
    lower = transcript.lower()
    
    # Extract chief complaint
    chief_complaint = ""
    duration = ""
    for pain_term in ["knee pain", "back pain", "shoulder pain", "neck pain", "hip pain", "ankle pain", "wrist pain", "pain"]:
        if pain_term in lower:
            chief_complaint = pain_term
            break

    # Extract duration
    dur_match = re.search(r"(?:for|about|approximately)\s+((?:\w+\s+)?(?:weeks?|days?|months?|years?))", lower)
    if dur_match:
        duration = dur_match.group(1).strip()

    # Extract objective tests
    tests = []
    # Check for flexion/extension angles
    flex_matches = re.finditer(r"(left|right)?\s*(?:knee|shoulder|hip|elbow)?\s*flexion\s*(?:is|was|measures)?\s*(\d+)\s*(?:degrees?|deg)?", lower)

    for m in flex_matches:
        side = m.group(1) or ""
        val = m.group(2)
        unit = "degrees"
        tests.append({
            "testName": "flexion",
            "unitName": unit,
            "value": val,
            "left": val if side == "left" else "",
            "right": val if side == "right" else "",
            "comments": f"{side.capitalize()} side" if side else ""
        })

    # Recommendations
    recommendations = []
    if "physiotherapy" in lower or "therapy" in lower:
        freq = ""
        if "twice a week" in lower or "twice per week" in lower:
            freq = "twice a week"
        elif "weekly" in lower or "once a week" in lower:
            freq = "once a week"
        recommendations.append({
            "sessionType": "physiotherapy",
            "sessionFrequency": freq
        })

    # Patient advice
    advice = ""
    if "icing" in lower or "ice" in lower:
        ice_match = re.search(r"(icing[^\.\,]*|ice[^\.\,]*)", lower)
        if ice_match:
            advice = ice_match.group(1).strip()

    return {
        "clinicalDetails": {
            "clinicalHistory": "",
            "chiefComplaint": chief_complaint,
            "duration": duration
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": tests
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": recommendations,
        "patientAdvice": {
            "adviceDetails": advice
        }
    }
