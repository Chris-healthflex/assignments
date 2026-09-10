"""Tests that the internal extraction shape mirrors the production schema."""

import pytest
from pydantic import ValidationError

from clinical_assessment.extraction_models import ExtractedValue, RawExtraction
from clinical_assessment.schema import FirstAssessment


def leaf_paths(model: type, prefix: str = "") -> set[str]:
    """Collect dotted paths to every leaf of a Pydantic model tree.

    An ``ExtractedValue`` counts as a leaf: it is the wrapper being compared
    against a plain ``str`` on the production side.
    """
    if model is ExtractedValue or model is str:
        return {prefix}

    paths: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        path = f"{prefix}.{name}" if prefix else name
        item_type = getattr(annotation, "__args__", None)
        if item_type and getattr(annotation, "__origin__", None) is list:
            paths |= leaf_paths(item_type[0], f"{path}[]")
            continue
        paths |= leaf_paths(annotation, path)
    return paths


def test_extraction_mirrors_the_production_schema_field_for_field() -> None:
    assert leaf_paths(RawExtraction) == leaf_paths(FirstAssessment)


def test_every_leaf_carries_its_own_evidence() -> None:
    extracted = ExtractedValue(value="120", evidence="abduction was 120 degrees")

    dumped = extracted.model_dump()

    assert dumped == {"value": "120", "evidence": "abduction was 120 degrees"}


def test_absent_datum_is_representable_as_empty_strings() -> None:
    extracted = ExtractedValue(value="", evidence="")

    assert (extracted.value, extracted.evidence) == ("", "")


def test_value_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedValue(value="120")  # type: ignore[call-arg]


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ExtractedValue(value="120", evidence="120 degrees", confidence="high")  # type: ignore[call-arg]
