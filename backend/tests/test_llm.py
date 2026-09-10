"""Tests for the Anthropic extraction wrapper.

The client is stubbed throughout: the suite makes no network calls and spends
nothing. ``httpx2`` is the transport the installed SDK is built on, and its
request/response objects are what the SDK's own exceptions require.
"""

from typing import Any

import anthropic
import httpx2
import pytest

from clinical_assessment.errors import ExtractionError
from clinical_assessment.extraction_models import (
    ExtractedValue,
    RawClinicalDetails,
    RawExtraction,
    RawObjectiveAssessment,
    RawPatientAdvice,
)
from clinical_assessment.llm import EXTRACTION_MAX_TOKENS, request_extraction

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def blank(value: str = "", evidence: str = "") -> ExtractedValue:
    """Build an ExtractedValue, defaulting to the "not stated" pair."""
    return ExtractedValue(value=value, evidence=evidence)


def build_extraction() -> RawExtraction:
    """Build a minimal valid extraction for the stub to return."""
    return RawExtraction(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=blank(),
            chiefComplaint=blank("Right shoulder pain", "my right shoulder hurts"),
            duration=blank(),
        ),
        subjectiveAssessments=[],
        objectiveAssessment=RawObjectiveAssessment(tests=[]),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=RawPatientAdvice(adviceDetails=blank()),
    )


class StubResponse:
    """Stands in for anthropic's ParsedMessage."""

    def __init__(
        self,
        parsed_output: RawExtraction | None,
        stop_reason: str = "end_turn",
        stop_details: Any = None,
    ) -> None:
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class StubMessages:
    """Records the kwargs it was called with, then returns or raises."""

    def __init__(self, response: StubResponse | None, error: Exception | None) -> None:
        self._response = response
        self._error = error
        self.captured_kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> StubResponse:
        self.captured_kwargs = kwargs
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class StubClient:
    """Minimal stand-in for anthropic.Anthropic."""

    def __init__(
        self,
        response: StubResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.messages = StubMessages(response, error)


def test_extract_returns_parsed_output_from_client() -> None:
    extraction = build_extraction()
    client = StubClient(StubResponse(extraction))

    returned = request_extraction(client, "extract this")  # type: ignore[arg-type]

    assert returned is extraction


def test_request_is_constrained_to_the_extraction_schema() -> None:
    client = StubClient(StubResponse(build_extraction()))

    request_extraction(client, "extract this")  # type: ignore[arg-type]

    assert client.messages.captured_kwargs["output_format"] is RawExtraction


def test_request_sends_the_prompt_as_the_user_turn() -> None:
    client = StubClient(StubResponse(build_extraction()))

    request_extraction(client, "extract this")  # type: ignore[arg-type]

    assert client.messages.captured_kwargs["messages"] == [
        {"role": "user", "content": "extract this"}
    ]


def test_request_caps_output_tokens() -> None:
    client = StubClient(StubResponse(build_extraction()))

    request_extraction(client, "extract this")  # type: ignore[arg-type]

    assert client.messages.captured_kwargs["max_tokens"] == EXTRACTION_MAX_TOKENS


def test_status_error_is_wrapped_in_extraction_error() -> None:
    rate_limited = anthropic.RateLimitError(
        "slow down",
        response=httpx2.Response(429, request=REQUEST),
        body=None,
    )
    client = StubClient(error=rate_limited)

    with pytest.raises(ExtractionError, match="HTTP 429"):
        request_extraction(client, "extract this")  # type: ignore[arg-type]


def test_connection_error_is_wrapped_in_extraction_error() -> None:
    client = StubClient(
        error=anthropic.APIConnectionError(message="no route", request=REQUEST)
    )

    with pytest.raises(ExtractionError, match="Could not reach"):
        request_extraction(client, "extract this")  # type: ignore[arg-type]


def test_refusal_stop_reason_raises_rather_than_returning_empty() -> None:
    refusal = anthropic.types.RefusalStopDetails(type="refusal", category="bio")
    client = StubClient(
        StubResponse(build_extraction(), stop_reason="refusal", stop_details=refusal)
    )

    with pytest.raises(ExtractionError, match="declined"):
        request_extraction(client, "extract this")  # type: ignore[arg-type]


def test_missing_parsed_output_raises_rather_than_returning_none() -> None:
    client = StubClient(StubResponse(None, stop_reason="max_tokens"))

    with pytest.raises(ExtractionError, match="no schema-valid extraction"):
        request_extraction(client, "extract this")  # type: ignore[arg-type]
