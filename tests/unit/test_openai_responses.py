"""OpenAI Responses API adapter behavior without network calls."""

from types import SimpleNamespace

import pytest

from app.llm.openai_responses import LLMServiceError, OpenAIResponsesLanguageModel


class FakeResponses:
    def __init__(self, output_text: str = "Grounded answer [S1].") -> None:
        self.output_text = output_text
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def test_openai_adapter_uses_responses_api_contract() -> None:
    responses = FakeResponses()
    model = OpenAIResponsesLanguageModel(
        api_key="test-key",
        model="test-model",
        max_output_tokens=321,
        client=FakeClient(responses),
    )

    answer = model.generate(instructions="Use context.", input_text="Question")

    assert answer == "Grounded answer [S1]."
    assert responses.kwargs == {
        "model": "test-model",
        "instructions": "Use context.",
        "input": "Question",
        "max_output_tokens": 321,
        "store": False,
    }


def test_openai_adapter_rejects_empty_provider_output() -> None:
    model = OpenAIResponsesLanguageModel(
        api_key="test-key",
        model="test-model",
        client=FakeClient(FakeResponses("  ")),
    )
    with pytest.raises(LLMServiceError, match="empty"):
        model.generate(instructions="Use context.", input_text="Question")


def test_openai_adapter_wraps_provider_errors() -> None:
    class FailingResponses:
        def create(self, **kwargs: object) -> None:
            raise TimeoutError("timed out")

    model = OpenAIResponsesLanguageModel(
        api_key="test-key",
        model="test-model",
        client=FakeClient(FailingResponses()),  # type: ignore[arg-type]
    )
    with pytest.raises(LLMServiceError, match="generation failed"):
        model.generate(instructions="Use context.", input_text="Question")

