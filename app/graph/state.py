from typing import TypedDict, Optional, List, Dict, Any

class AssessmentState(TypedDict):
    # The clinical transcript text input
    transcript: str
    
    # Raw JSON extraction parsed from LLM output
    raw_extraction: Optional[Dict[str, Any]]
    
    # Normalized fields after formatting and schema alignment
    normalized_extraction: Optional[Dict[str, Any]]
    
    # Tracked confidence scores and reasons per field/field path
    confidence_metadata: Dict[str, Dict[str, Any]]
    
    # List of validation errors (fields below confidence threshold)
    validation_errors: List[Dict[str, Any]]
    
    # The final validated FirstAssessment dictionary (serializable version of FirstAssessment)
    first_assessment: Optional[Dict[str, Any]]
