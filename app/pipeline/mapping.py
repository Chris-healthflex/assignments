from app.models.first_assessment import FirstAssessment


def map_and_validate(value: FirstAssessment | dict) -> FirstAssessment:
    if isinstance(value, FirstAssessment):
        return value
    return FirstAssessment.model_validate(value)


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
