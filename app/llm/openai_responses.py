"""OpenAI Responses API adapter."""

from typing import Any


class LLMServiceError(RuntimeError):
    """Raised when a provider cannot return a usable response."""


class OpenAIResponsesLanguageModel:
    """Small adapter over the official OpenAI Responses API client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 800,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key cannot be empty.")
        if not model.strip():
            raise ValueError("OpenAI model name cannot be empty.")
        if timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be positive.")
        if max_output_tokens < 1:
            raise ValueError("Maximum output tokens must be at least 1.")

        self._model_name = model.strip()
        self.max_output_tokens = max_output_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "The 'openai' package is required for LLM_PROVIDER=openai."
                ) from exc
            client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.client = client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, *, instructions: str, input_text: str) -> str:
        if not instructions.strip():
            raise ValueError("LLM instructions cannot be empty.")
        if not input_text.strip():
            raise ValueError("LLM input cannot be empty.")

        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=input_text,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except Exception as exc:
            raise LLMServiceError(f"OpenAI response generation failed: {exc}") from exc

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise LLMServiceError("OpenAI returned an empty text response.")
        return output_text.strip()

