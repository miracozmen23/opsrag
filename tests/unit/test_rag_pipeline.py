"""Basic RAG orchestration behavior."""

import pytest

from app.rag.pipeline import RAGPipeline, RAGPipelineError
from app.rag.prompts import INSUFFICIENT_CONTEXT_ANSWER
from app.retrieval.models import RetrievedChunk
from tests.helpers import make_retrieved_chunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results


class FakeLanguageModel:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, answer: str = "Check the port [S1].") -> None:
        self.answer_text = answer
        self.calls: list[dict[str, str]] = []

    def generate(self, *, instructions: str, input_text: str) -> str:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self.answer_text


def test_pipeline_retrieves_generates_and_returns_sources() -> None:
    chunks = [
        make_retrieved_chunk(score=0.84),
        make_retrieved_chunk(chunk_id="chunk_2", source="other.md", score=0.51),
    ]
    retriever = FakeRetriever(chunks)
    llm = FakeLanguageModel()
    pipeline = RAGPipeline(retriever, llm, top_k=5)

    result = pipeline.answer(" Why is the port unavailable? ")

    assert retriever.calls == [("Why is the port unavailable?", 5)]
    assert result.answer == "Check the port [S1]."
    assert [source.source_id for source in result.sources] == ["S1"]
    assert result.sources[0].document == "guide.md"
    assert result.sources[0].title == "Guide"
    assert result.sources[0].chunk_ids == ("chunk_1",)
    assert result.retrieval_confidence == 0.84
    assert result.metadata.retrieved_chunks == 2
    assert result.metadata.cited_sources == 1
    assert '"source_id": "S1"' in llm.calls[0]["input_text"]
    assert '"excerpts"' in llm.calls[0]["input_text"]


def test_pipeline_trace_preserves_ranked_context_chunks() -> None:
    chunks = [
        make_retrieved_chunk(chunk_id="first", text="First context."),
        make_retrieved_chunk(chunk_id="second", text="Second context."),
    ]

    execution = RAGPipeline(
        FakeRetriever(chunks),
        FakeLanguageModel(),
    ).answer_with_trace("question")

    assert execution.result.answer == "Check the port [S1]."
    assert [chunk.text for chunk in execution.retrieved_chunks] == [
        "First context.",
        "Second context.",
    ]


def test_pipeline_returns_safe_answer_without_context_and_skips_llm() -> None:
    llm = FakeLanguageModel()
    result = RAGPipeline(FakeRetriever([]), llm).answer("Unknown topic")
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.sources == []
    assert result.retrieval_confidence == 0.0
    assert result.metadata.cited_sources == 0
    assert llm.calls == []


@pytest.mark.parametrize(("score", "expected"), [(1.4, 1.0), (-0.2, 0.0)])
def test_retrieval_confidence_is_clamped(score: float, expected: float) -> None:
    result = RAGPipeline(
        FakeRetriever([make_retrieved_chunk(score=score)]),
        FakeLanguageModel(),
    ).answer("question")
    assert result.retrieval_confidence == expected


def test_pipeline_uses_rerank_score_and_reports_hybrid_reranking() -> None:
    chunk = make_retrieved_chunk(
        score=0.03,
        rerank_score=2.0,
        retrieval_method="hybrid",
    )
    pipeline = RAGPipeline(
        FakeRetriever([chunk]),
        FakeLanguageModel(),
        retrieval_method="hybrid_reranked",
    )

    result = pipeline.answer("question")

    assert result.sources[0].score == 0.8808
    assert result.retrieval_confidence == 0.8808
    assert result.metadata.retrieval_method == "hybrid_reranked"


@pytest.mark.parametrize(
    ("logit", "expected"),
    [(1000.0, 1.0), (-1000.0, 0.0)],
)
def test_rerank_confidence_sigmoid_is_numerically_stable(
    logit: float,
    expected: float,
) -> None:
    result = RAGPipeline(
        FakeRetriever([make_retrieved_chunk(rerank_score=logit)]),
        FakeLanguageModel(),
        retrieval_method="hybrid_reranked",
    ).answer("question")
    assert result.retrieval_confidence == expected


def test_empty_reranked_result_preserves_retrieval_method() -> None:
    result = RAGPipeline(
        FakeRetriever([]),
        FakeLanguageModel(),
        retrieval_method="hybrid_reranked",
    ).answer("question")
    assert result.metadata.retrieval_method == "hybrid_reranked"


def test_hybrid_result_preserves_retrieval_method() -> None:
    result = RAGPipeline(
        FakeRetriever([make_retrieved_chunk(retrieval_method="hybrid")]),
        FakeLanguageModel(),
        retrieval_method="hybrid",
    ).answer("question")
    assert result.metadata.retrieval_method == "hybrid"


def test_pipeline_returns_only_sources_cited_by_the_answer() -> None:
    chunks = [
        make_retrieved_chunk(chunk_id="a", source="first.md", score=0.9),
        make_retrieved_chunk(chunk_id="b", source="second.md", score=0.4),
    ]
    result = RAGPipeline(
        FakeRetriever(chunks),
        FakeLanguageModel("Use the second procedure [S2]."),
    ).answer("question")
    assert [source.document for source in result.sources] == ["second.md"]
    assert result.retrieval_confidence == 0.4
    assert result.metadata.retrieved_chunks == 2
    assert result.metadata.cited_sources == 1


@pytest.mark.parametrize(
    "answer",
    ["No citation here.", "Invented source [S9].", "Bad citation [S1, S2]."],
)
def test_pipeline_rejects_unverifiable_answer_attribution(answer: str) -> None:
    with pytest.raises(RAGPipelineError, match="source attribution"):
        RAGPipeline(
            FakeRetriever([make_retrieved_chunk()]),
            FakeLanguageModel(answer),
        ).answer("question")


def test_retrieval_failures_become_pipeline_errors() -> None:
    class FailingRetriever:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            raise ConnectionError("qdrant unavailable")

    with pytest.raises(RAGPipelineError, match="retrieval failed"):
        RAGPipeline(FailingRetriever(), FakeLanguageModel()).answer("question")


def test_generation_failures_become_pipeline_errors() -> None:
    class FailingLLM(FakeLanguageModel):
        def generate(self, *, instructions: str, input_text: str) -> str:
            raise TimeoutError("provider unavailable")

    with pytest.raises(RAGPipelineError, match="generation failed"):
        RAGPipeline(
            FakeRetriever([make_retrieved_chunk()]),
            FailingLLM(),
        ).answer("question")
