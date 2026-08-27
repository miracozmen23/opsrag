"""Cross-encoder reranker and retrieval wrapper behavior."""

from typing import Any

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from tests.helpers import make_retrieved_chunk


class FakeCrossEncoderModel:
    def __init__(self, scores: Any) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        **kwargs: object,
    ) -> Any:
        self.calls.append((pairs, kwargs))
        return self.scores


def test_cross_encoder_reranks_and_records_scores_without_losing_rrf_score() -> None:
    model = FakeCrossEncoderModel([0.15, 0.91, 0.44])
    candidates = [
        make_retrieved_chunk(
            chunk_id="a", score=0.032, retrieval_method="hybrid", text="first"
        ),
        make_retrieved_chunk(
            chunk_id="b", score=0.025, retrieval_method="hybrid", text="second"
        ),
        make_retrieved_chunk(
            chunk_id="c", score=0.021, retrieval_method="hybrid", text="third"
        ),
    ]
    reranker = CrossEncoderReranker(
        "fake-reranker",
        batch_size=7,
        model=model,
    )

    results = reranker.rerank(" database error ", candidates, top_k=2)

    assert [result.metadata.chunk_id for result in results] == ["b", "c"]
    assert [result.rerank_score for result in results] == [0.91, 0.44]
    assert results[0].score == 0.025
    assert results[0].retrieval_method == "hybrid"
    pairs, kwargs = model.calls[0]
    assert pairs == [
        ("database error", "first"),
        ("database error", "second"),
        ("database error", "third"),
    ]
    assert kwargs == {
        "batch_size": 7,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }


def test_cross_encoder_keeps_input_order_when_scores_tie() -> None:
    candidates = [
        make_retrieved_chunk(chunk_id="first", retrieval_method="hybrid"),
        make_retrieved_chunk(chunk_id="second", retrieval_method="hybrid"),
    ]
    results = CrossEncoderReranker(
        "fake",
        model=FakeCrossEncoderModel([[0.5], [0.5]]),
    ).rerank("query", candidates, top_k=2)
    assert [result.metadata.chunk_id for result in results] == ["first", "second"]


def test_cross_encoder_skips_model_loading_for_empty_candidates() -> None:
    reranker = CrossEncoderReranker("not-installed")
    assert reranker.rerank("query", [], top_k=3) == []
    assert reranker._model is None


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.2], [[0.1, 0.2], [0.3]]])
def test_cross_encoder_rejects_incompatible_scores(scores: Any) -> None:
    candidates = [
        make_retrieved_chunk(chunk_id="a"),
        make_retrieved_chunk(chunk_id="b"),
    ]
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoderModel(scores))
    with pytest.raises(RuntimeError, match="Reranking"):
        reranker.rerank("query", candidates, top_k=2)


class FakeCandidateRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            (query, [candidate.metadata.chunk_id for candidate in candidates], top_k)
        )
        return candidates[:top_k]


def test_reranking_retriever_requests_candidate_pool_then_final_top_k() -> None:
    candidates = [make_retrieved_chunk(chunk_id=str(index)) for index in range(12)]
    candidate_retriever = FakeCandidateRetriever(candidates)
    reranker = FakeReranker()
    retriever = RerankingRetriever(
        candidate_retriever,
        reranker,
        candidate_top_k=10,
    )

    results = retriever.search(" query ", top_k=3)

    assert len(results) == 3
    assert candidate_retriever.calls == [("query", 10)]
    assert reranker.calls == [("query", [str(index) for index in range(10)], 3)]


def test_reranking_retriever_never_requests_fewer_candidates_than_final_top_k() -> None:
    candidates = [make_retrieved_chunk(chunk_id=str(index)) for index in range(6)]
    candidate_retriever = FakeCandidateRetriever(candidates)
    retriever = RerankingRetriever(
        candidate_retriever,
        FakeReranker(),
        candidate_top_k=2,
    )
    retriever.search("query", top_k=5)
    assert candidate_retriever.calls == [("query", 5)]


@pytest.mark.parametrize(
    ("query", "top_k"),
    [(" ", 1), ("query", 0)],
)
def test_reranking_retriever_validates_input(query: str, top_k: int) -> None:
    retriever = RerankingRetriever(
        FakeCandidateRetriever([]),
        FakeReranker(),
    )
    with pytest.raises(ValueError):
        retriever.search(query, top_k)
