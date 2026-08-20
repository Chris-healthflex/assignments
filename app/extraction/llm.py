"""LLM provider factory plus a provider-agnostic structured-output call.

Why not ``with_structured_output``: it resolves to native tool-calling on
Anthropic and OpenAI but degrades badly on a local 3B model, which is the
default here. Instead every provider goes through one path - ask for JSON,
parse it, validate against Pydantic, and on failure hand the validation error
back for a repair attempt. That path behaves identically across providers,
keeps the raw response available for debugging, and makes the agent trivial to
test with a stub LLM that needs no network.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """The configured provider could not be reached or is misconfigured.

    Surfaces as HTTP 503 with a message telling the caller how to fix it.
    """


class StructuredOutputError(RuntimeError):
    """The model never returned JSON matching the requested schema."""


def build_llm(settings: Settings | None = None):
    """Construct the chat model named by ``LLM_PROVIDER``."""
    settings = settings or get_settings()
    provider = settings.llm_provider
    model = settings.default_llm_model

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise LLMUnavailableError(
                "langchain-ollama is not installed. Run: pip install -r requirements.txt"
            ) from exc
        return ChatOllama(
            model=model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
            format="json",      # constrains decoding to syntactically valid JSON
            num_ctx=8192,       # the transcript plus the schema must both fit
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LLMUnavailableError(
                "langchain-anthropic is not installed. Run: "
                "pip install -r requirements-optional.txt"
            ) from exc
        if not settings.anthropic_api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(
            model=model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LLMUnavailableError(
                "langchain-openai is not installed. Run: "
                "pip install -r requirements-optional.txt"
            ) from exc
        if not settings.openai_api_key:
            raise LLMUnavailableError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    raise LLMUnavailableError(f"Unknown LLM_PROVIDER: {provider!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Small models wrap JSON in prose or code fences even when told not to, so
    strip fences first, then fall back to brace matching that respects string
    literals (a naive scan breaks on a brace inside a quoted clinical note).
    """
    text = raw.strip()

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break

    raise StructuredOutputError(f"No JSON object found in response: {raw[:300]!r}")


def _schema_hint(model_cls: type[BaseModel]) -> str:
    """A compact schema description.

    The full JSON Schema is verbose enough to crowd out the transcript in an
    8k context, so only the field names and types are shown.
    """
    schema = model_cls.model_json_schema()
    defs = schema.get("$defs", {})

    def describe(props: dict) -> dict:
        out = {}
        for name, spec in props.items():
            if "$ref" in spec:
                ref = defs.get(spec["$ref"].split("/")[-1], {})
                out[name] = describe(ref.get("properties", {}))
            elif spec.get("type") == "array":
                items = spec.get("items", {})
                if "$ref" in items:
                    ref = defs.get(items["$ref"].split("/")[-1], {})
                    out[name] = [describe(ref.get("properties", {}))]
                else:
                    out[name] = ["string"]
            else:
                out[name] = "string"
        return out

    return json.dumps(describe(schema.get("properties", {})), indent=2)


def structured_call(
    llm,
    *,
    system: str,
    user: str,
    model_cls: type[T],
    max_retries: int | None = None,
) -> T:
    """Call the LLM and return a validated instance of ``model_cls``.

    On a parse or validation failure the error text is fed back so the model
    can correct itself, up to ``max_retries`` times. The repair prompt repeats
    the no-invention rule, because the naive way for a model to satisfy a
    schema complaint is to fill every field with something plausible.
    """
    settings = get_settings()
    retries = settings.llm_max_retries if max_retries is None else max_retries

    prompt = (
        f"{user}\n\n"
        "Respond with a single JSON object and nothing else, in this shape:\n"
        f"{_schema_hint(model_cls)}"
    )

    last_error = ""
    for attempt in range(retries + 1):
        messages = [("system", system), ("human", prompt)]
        if last_error:
            messages.append(
                (
                    "human",
                    f"Your previous response was rejected: {last_error}\n"
                    "Return corrected JSON only. Do not invent values to satisfy "
                    'the schema - use "" for anything the transcript does not state.',
                )
            )

        try:
            response = llm.invoke(messages)
        except Exception as exc:      # network or daemon failure -> 503
            raise LLMUnavailableError(
                f"LLM provider call failed: {type(exc).__name__}: {exc}"
            ) from exc

        raw = getattr(response, "content", response)
        if isinstance(raw, list):
            # Anthropic returns a list of content blocks.
            raw = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )

        try:
            return model_cls.model_validate(extract_json_object(str(raw)))
        except (StructuredOutputError, ValidationError) as exc:
            last_error = str(exc)[:500]
            logger.warning(
                "Structured output attempt %d/%d failed: %s",
                attempt + 1,
                retries + 1,
                last_error.splitlines()[0],
            )

    raise StructuredOutputError(
        f"Model did not return valid {model_cls.__name__} after "
        f"{retries + 1} attempts. Last error: {last_error}"
    )
