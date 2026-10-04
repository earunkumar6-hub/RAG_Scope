"""Build the configured LLM provider (``LLM_PROVIDER``)."""

from app.core.config import Settings
from app.llm.base import LLMNotConfigured, LLMProvider


def build_llm(settings: Settings) -> LLMProvider:
    """Return the provider selected by settings; raises ``LLMNotConfigured`` if unusable."""
    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise LLMNotConfigured("OPENAI_API_KEY not set")
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            fast_model=settings.openai_model_fast,
            base_url=settings.openai_base_url,
        )
    # Only the OpenAI adapter is implemented (OPENAI_BASE_URL covers OpenAI-compatible gateways).
    raise LLMNotConfigured(f"LLM provider '{settings.llm_provider}' adapter not installed yet")
