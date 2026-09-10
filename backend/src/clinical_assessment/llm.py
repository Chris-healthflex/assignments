"""Anthropic Messages API wrapper for schema-constrained extraction.

``client.messages.parse(output_format=...)`` constrains decoding to the schema
server-side and validates the response, so the model cannot rename a key or add
one. The API key is read from the environment by the SDK; it is never passed
through this module and never logged.
"""

import logging

import anthropic

from .config import load_settings
from .errors import ExtractionError
from .extraction_models import RawExtraction
from .prompts import EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)

# The evidence-carrying extraction is roughly twice the size of the assessment
# it produces, and a response truncated at the cap parses to nothing at all.
EXTRACTION_MAX_TOKENS = 16_000


def request_extraction(client: anthropic.Anthropic, user_prompt: str) -> RawExtraction:
    """Ask the model for an evidence-carrying extraction.

    Args:
        client: An Anthropic client; injected so callers and tests control it.
        user_prompt: The user-turn text, built by :mod:`.prompts`.

    Returns:
        The schema-validated extraction.

    Raises:
        ExtractionError: If the request fails, the model refuses, or no
            schema-valid output comes back.
    """
    settings = load_settings()
    # Length only: the transcript is clinical material and does not belong in
    # application logs.
    logger.info(
        "Requesting extraction from %s (%d prompt characters)",
        settings.anthropic_model,
        len(user_prompt),
    )

    try:
        response = client.messages.parse(
            model=settings.anthropic_model,
            max_tokens=EXTRACTION_MAX_TOKENS,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=RawExtraction,
        )
    except anthropic.APIStatusError as exc:
        # Covers the whole 4xx/5xx family, RateLimitError included, since those
        # all subclass APIStatusError.
        raise ExtractionError(
            f"Extraction request failed with HTTP {exc.status_code}: {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ExtractionError(f"Could not reach the extraction API: {exc}") from exc

    if response.stop_reason == "refusal":
        raise ExtractionError(
            "The model declined to extract from this transcript "
            f"(category={_refusal_category(response)})"
        )

    # parsed_output is a property that yields None when no text block carried
    # schema-valid JSON - a response truncated at max_tokens lands here.
    extraction = response.parsed_output
    if extraction is None:
        raise ExtractionError(
            "The model returned no schema-valid extraction "
            f"(stop_reason={response.stop_reason!r})"
        )
    return extraction


def _refusal_category(response: anthropic.types.ParsedMessage[RawExtraction]) -> str:
    """Read the refusal category, which is populated only for refusals."""
    if response.stop_details is None:
        return "unknown"
    return str(response.stop_details.category)
