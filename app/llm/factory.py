"""Language model provider selection."""

from app.core.config import Settings
from app.llm.base import LanguageModel
from app.llm.ollama_chat import OllamaChatLanguageModel
from app.llm.openai_responses import OpenAIResponsesLanguageModel


class LLMConfigurationError(RuntimeError):
    """Raised when LLM settings cannot construct the selected provider."""


def create_llm_service(settings: Settings) -> LanguageModel:
    """Create the configured provider behind the shared LLM interface."""

    if not settings.llm_model.strip():
        raise LLMConfigurationError("LLM_MODEL must be configured before asking questions.")

    if settings.llm_provider == "openai":
        if settings.llm_api_key is None:
            raise LLMConfigurationError(
                "LLM_API_KEY must be configured for the OpenAI provider."
            )
        return OpenAIResponsesLanguageModel(
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    if settings.llm_provider == "ollama":
        return OllamaChatLanguageModel(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'."
    )
