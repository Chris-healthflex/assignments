from app.core.exceptions import ConfidenceError
from app.schemas.extraction import ExtractionResult, ExtractionField
from app.guardrails.confidence_gate import collect_field_errors, apply_source_verification

def test_low_confidence_trigger_error():
    result = ExtractionResult()
    # Set a field with low confidence
    result.clinicalDetails.chiefComplaint = ExtractionField(
        value="headache", is_mentioned=True, confidence=0.5, source_quote="head"
    )
    errors = collect_field_errors(result)
    assert len(errors) == 1
    assert "chiefComplaint" in errors[0]["field"]