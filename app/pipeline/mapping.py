from app.models.first_assessment import FirstAssessment
from app.pipeline.grounding import ground_assessment


def map_and_validate(value: FirstAssessment | dict, transcript: str | None = None) -> tuple[FirstAssessment, list[dict]]:
    if isinstance(value, FirstAssessment):
        assessment = value
    else:
        assessment = FirstAssessment.model_validate(value)
    if transcript is None:
        return assessment, []
    return ground_assessment(assessment, transcript)


def low_confidence_fields(confidence: dict[str, float], threshold: float) -> list[dict]:
    return [
        {
            "field": field,
            "confidence": score,
            "message": f"Extraction confidence is below the threshold of {threshold:.2f}",
        }
        for field, score in confidence.items()
        if score < threshold
    ]
