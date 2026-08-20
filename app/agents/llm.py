
from __future__ import annotations
from typing import Any, Protocol
from app.config import Settings

class ConfigurationError(RuntimeError):
    """Raised when the configured provider is missing its credentials."""

class StructuredLLM(Protocol):
    """The only capability the graph needs from a model."""

    def invoke(self, messages: Any) -> Any: ...

def build_llm(settings: Settings, output_model: type) -> StructuredLLM:
    provider = settings.llm_provider

    if provider == "stub":
        from app.agents.stub import StubStructuredLLM

        return StubStructuredLLM(output_model)

    if provider == "groq":
        if not settings.groq_api_key:
            raise ConfigurationError(
            )
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            max_tokens=settings.llm_max_tokens,
            temperature=0,  
            timeout=180,
            max_retries=2,
        )
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigurationError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty. "
                "Add the key to .env, or set LLM_PROVIDER=openai."
            )
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key,
            timeout=180,
            max_retries=2,
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Add the key to .env, or set LLM_PROVIDER=anthropic."
            )
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            timeout=180,
            max_retries=2,
        )
    else: 
        raise ConfigurationError(f"Unknown LLM_PROVIDER: {provider!r}")
    return model.with_structured_output(
        output_model, method=settings.llm_structured_method, include_raw=False
    )
