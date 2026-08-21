from typing import List, Dict, Any
from app.core.config import settings
from app.schemas.extraction import ExtractionResult, ExtractionField
from app.guardrails.source_match import adjust_confidence
from app.guardrails.date_validator import is_valid_date_string
from app.guardrails.numeric_validator import is_numeric, numeric_source_supported

def apply_source_verification(result: ExtractionResult, transcript: str) -> ExtractionResult:
    """Adjust confidence values based on source match and domain validators."""
    def adjust_field(field: ExtractionField):
        if field.is_mentioned and field.value is not None:
            str_value = str(field.value)
            new_conf = adjust_confidence(str_value, field.confidence, transcript)
            # Additional domain checks
            if is_numeric(str_value) and not numeric_source_supported(str_value, transcript):
                new_conf = min(new_conf, 0.5)
            if "date" in field.__class__.__name__.lower() or "targetdate" in field.__class__.__name__.lower():
                if not is_valid_date_string(str_value):
                    new_conf = min(new_conf, 0.5)
            field.confidence = round(new_conf, 3)
        return field

    # Recursively apply
    def walk(obj):
        if isinstance(obj, ExtractionField):
            adjust_field(obj)
        elif hasattr(obj, "__dict__"):
            for name, value in obj.__dict__.items():
                if name.startswith("_"):
                    continue
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
    walk(result)
    return result

def collect_field_errors(result: ExtractionResult) -> List[Dict[str, Any]]:
    """Return field-level errors for non-empty fields with low confidence."""
    errors = []
    threshold = settings.CONFIDENCE_THRESHOLD

    def walk(obj, prefix=""):
        if isinstance(obj, ExtractionField):
            if obj.is_mentioned and obj.value is not None and obj.confidence < threshold:
                errors.append({
                    "field": prefix,
                    "confidence": round(obj.confidence, 3),
                    "reason": f"Confidence below threshold ({threshold})"
                })
        elif hasattr(obj, "__dict__"):
            for name, value in obj.__dict__.items():
                if name.startswith("_"):
                    continue
                walk(value, f"{prefix}.{name}" if prefix else name)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{prefix}[{i}]")

    walk(result)
    return errors