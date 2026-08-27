"""Ollama adapter through its OpenAI-compatible chat endpoint."""

from typing import Any

from app.llm.openai_responses import LLMServiceError


class OllamaChatLanguageModel:
    """Generate deterministic local responses through Ollama."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 800,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model name cannot be empty.")
        if not base_url.strip():
            raise ValueError("Ollama base URL cannot be empty.")
        if timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive.")
        if max_output_tokens < 1:
            raise ValueError("Maximum output tokens must be at least 1.")

        self._model_name = model.strip()
        self.max_output_tokens = max_output_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "The 'openai' package is required for the Ollama-compatible client."
                ) from exc
            client = OpenAI(
                base_url=f"{base_url.rstrip('/')}/v1",
                api_key="ollama",
                timeout=timeout_seconds,
            )
        self.client = client

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, *, instructions: str, input_text: str) -> str:
        if not instructions.strip():
            raise ValueError("LLM instructions cannot be empty.")
        if not input_text.strip():
            raise ValueError("LLM input cannot be empty.")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
                max_tokens=self.max_output_tokens,
                temperature=0,
                reasoning_effort="none",
            )
        except Exception as exc:
            raise LLMServiceError(f"Ollama response generation failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        output_text = (
            getattr(getattr(choices[0], "message", None), "content", None)
            if choices
            else None
        )
        if not isinstance(output_text, str) or not output_text.strip():
            raise LLMServiceError("Ollama returned an empty text response.")
        return output_text.strip()
