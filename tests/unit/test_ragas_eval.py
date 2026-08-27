"""RAGAS adapter behavior without external model calls."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from app.evaluation import ragas_eval
from app.evaluation.ragas_eval import RagasScorer


class FakeMetric:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or SimpleNamespace(value=0.75, reason="evidence")
        self.error = error
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def test_ragas_scorer_preserves_metric_failures_and_undefined_values() -> None:
    faithfulness = FakeMetric()
    answer_relevance = FakeMetric(result=SimpleNamespace(value=float("nan"), reason=None))
    context_precision = FakeMetric(error=TimeoutError("judge timeout"))
    context_recall = FakeMetric(result=SimpleNamespace(value=0.5, reason=None))
    scorer = RagasScorer(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
        ragas_version="0.4.3",
        judge_model="judge",
        embedding_model="embedder",
    )

    outcomes = asyncio.run(
        scorer.score(
            question="Question?",
            response="Answer.",
            reference="Reference.",
            retrieved_contexts=["Context."],
        )
    )

    assert outcomes["faithfulness"].value == 0.75
    assert outcomes["answer_relevance"].status == "undefined"
    assert outcomes["context_precision"].status == "failed"
    assert outcomes["context_precision"].value is None
    assert "judge timeout" in outcomes["context_precision"].error
    assert outcomes["context_recall"].value == 0.5
    assert faithfulness.calls[0]["retrieved_contexts"] == ["Context."]
    assert answer_relevance.calls[0] == {
        "user_input": "Question?",
        "response": "Answer.",
    }


def test_vertexai_compatibility_only_stubs_the_missing_optional_module(
    monkeypatch,
) -> None:
    module_name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def missing_module(name: str):
        assert name == module_name
        error = ModuleNotFoundError(name)
        error.name = module_name
        raise error

    monkeypatch.setattr(ragas_eval.importlib, "import_module", missing_module)
    ragas_eval._install_optional_vertexai_compatibility()

    assert module_name in sys.modules
    assert hasattr(sys.modules[module_name], "ChatVertexAI")


def test_vertexai_compatibility_does_not_hide_nested_import_failures(
    monkeypatch,
) -> None:
    module_name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def missing_dependency(name: str):
        error = ModuleNotFoundError("google dependency")
        error.name = "google.cloud"
        raise error

    monkeypatch.setattr(ragas_eval.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        ragas_eval._install_optional_vertexai_compatibility()
    assert exc_info.value.name == "google.cloud"
