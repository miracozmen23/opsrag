"""Ollama adapter behavior without a running local model."""

from types import SimpleNamespace

import pytest

from app.llm.ollama_chat import OllamaChatLanguageModel
from app.llm.openai_responses import LLMServiceError


class FakeCompletions:
    def __init__(self, output_text: str = "Grounded answer [S1].") -> None:
        self.output_text = output_text
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.output_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_ollama_adapter_uses_openai_compatible_chat_contract() -> None:
    completions = FakeCompletions()
    model = OllamaChatLanguageModel(
        model="local-model",
        max_output_tokens=321,
        client=FakeClient(completions),
    )

    answer = model.generate(instructions="Use context.", input_text="Question")

    assert answer == "Grounded answer [S1]."
    assert completions.kwargs == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "Use context."},
            {"role": "user", "content": "Question"},
        ],
        "max_tokens": 321,
        "temperature": 0,
        "reasoning_effort": "none",
    }


def test_ollama_adapter_rejects_empty_provider_output() -> None:
    model = OllamaChatLanguageModel(
        model="local-model",
        client=FakeClient(FakeCompletions("  ")),
    )
    with pytest.raises(LLMServiceError, match="empty"):
        model.generate(instructions="Use context.", input_text="Question")


def test_ollama_adapter_wraps_provider_errors() -> None:
    class FailingCompletions:
        def create(self, **kwargs: object) -> None:
            raise TimeoutError("timed out")

    model = OllamaChatLanguageModel(
        model="local-model",
        client=FakeClient(FailingCompletions()),  # type: ignore[arg-type]
    )
    with pytest.raises(LLMServiceError, match="generation failed"):
        model.generate(instructions="Use context.", input_text="Question")
