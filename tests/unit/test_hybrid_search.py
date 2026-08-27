"""Reciprocal Rank Fusion and hybrid retrieval behavior."""

import pytest

from app.retrieval.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk
from tests.helpers import make_retrieved_chunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_rrf_combines_ranks_and_deduplicates_chunk_ids() -> None:
    dense = [
        make_retrieved_chunk(chunk_id="a", score=0.95),
        make_retrieved_chunk(chunk_id="b", score=0.85),
    ]
    sparse = [
        make_retrieved_chunk(
            chunk_id="b",
            score=7.0,
            retrieval_method="bm25",
        ),
        make_retrieved_chunk(
            chunk_id="c",
            score=5.0,
            retrieval_method="bm25",
        ),
    ]

    results = reciprocal_rank_fusion([dense, sparse], rrf_k=60)

    assert [result.metadata.chunk_id for result in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert all(result.retrieval_method == "hybrid" for result in results)


def test_rrf_does_not_double_count_duplicates_within_one_ranked_list() -> None:
    duplicate = make_retrieved_chunk(chunk_id="same")

    results = reciprocal_rank_fusion([[duplicate, duplicate]], rrf_k=60)

    assert len(results) == 1
    assert results[0].score == pytest.approx(1 / 61)


def test_rrf_is_deterministic_for_equal_scores() -> None:
    dense = [make_retrieved_chunk(chunk_id="dense")]
    sparse = [
        make_retrieved_chunk(chunk_id="sparse", retrieval_method="bm25")
    ]

    first = reciprocal_rank_fusion([dense, sparse])
    second = reciprocal_rank_fusion([dense, sparse])

    assert first == second
    assert [result.metadata.chunk_id for result in first] == ["dense", "sparse"]


def test_hybrid_retriever_uses_configured_candidate_counts_and_final_limit() -> None:
    dense = FakeRetriever(
        [
            make_retrieved_chunk(chunk_id="a"),
            make_retrieved_chunk(chunk_id="b"),
        ]
    )
    sparse = FakeRetriever(
        [
            make_retrieved_chunk(chunk_id="b", retrieval_method="bm25"),
            make_retrieved_chunk(chunk_id="c", retrieval_method="bm25"),
        ]
    )
    retriever = HybridRetriever(
        dense,
        sparse,
        dense_top_k=2,
        sparse_top_k=2,
        rrf_k=10,
    )

    results = retriever.search("  database connection  ", top_k=2)

    assert dense.calls == [("database connection", 2)]
    assert sparse.calls == [("database connection", 2)]
    assert len(results) == 2
    assert results[0].metadata.chunk_id == "b"


def test_hybrid_retriever_supports_one_empty_result_set() -> None:
    dense = FakeRetriever([make_retrieved_chunk(chunk_id="a")])
    sparse = FakeRetriever([])

    results = HybridRetriever(dense, sparse).search("query", top_k=3)

    assert [result.metadata.chunk_id for result in results] == ["a"]


@pytest.mark.parametrize(
    ("ranked_lists", "rrf_k", "top_k"),
    [([], 0, None), ([], 60, 0)],
)
def test_rrf_rejects_invalid_configuration(
    ranked_lists: list[list[RetrievedChunk]],
    rrf_k: int,
    top_k: int | None,
) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k, top_k=top_k)


def test_hybrid_retriever_rejects_invalid_inputs() -> None:
    dense = FakeRetriever([])
    sparse = FakeRetriever([])

    with pytest.raises(ValueError, match="Dense"):
        HybridRetriever(dense, sparse, dense_top_k=0)
    with pytest.raises(ValueError, match="Sparse"):
        HybridRetriever(dense, sparse, sparse_top_k=0)
    with pytest.raises(ValueError, match="RRF"):
        HybridRetriever(dense, sparse, rrf_k=0)

    retriever = HybridRetriever(dense, sparse)
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.search("  ", top_k=1)
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search("query", top_k=0)
