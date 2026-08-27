"""Hybrid dense and sparse retrieval using Reciprocal Rank Fusion."""

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """Shared contract implemented by dense and sparse retrievers."""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return ranked chunks for a query."""

        ...


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists by chunk ID without comparing raw provider scores."""

    if rrf_k < 1:
        raise ValueError("RRF k must be at least 1.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided.")

    fused_scores: dict[str, float] = {}
    representative_chunks: dict[str, RetrievedChunk] = {}
    best_ranks: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_results in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, result in enumerate(ranked_results, start=1):
            chunk_id = result.metadata.chunk_id
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            best_ranks[chunk_id] = min(rank, best_ranks.get(chunk_id, rank))
            if chunk_id not in representative_chunks:
                representative_chunks[chunk_id] = result
                first_seen[chunk_id] = seen_counter
                seen_counter += 1

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            best_ranks[chunk_id],
            first_seen[chunk_id],
            chunk_id,
        ),
    )
    if top_k is not None:
        ranked_chunk_ids = ranked_chunk_ids[:top_k]

    return [
        representative_chunks[chunk_id].model_copy(
            update={
                "score": fused_scores[chunk_id],
                "retrieval_method": "hybrid",
            }
        )
        for chunk_id in ranked_chunk_ids
    ]


class HybridRetriever:
    """Retrieve dense and sparse candidates and fuse their ranks with RRF."""

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        *,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if dense_top_k < 1:
            raise ValueError("Dense candidate count must be at least 1.")
        if sparse_top_k < 1:
            raise ValueError("Sparse candidate count must be at least 1.")
        if rrf_k < 1:
            raise ValueError("RRF k must be at least 1.")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return one fused candidate list for a non-empty query."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        started_at = perf_counter()
        dense_results = self.dense_retriever.search(
            normalized_query,
            self.dense_top_k,
        )
        sparse_results = self.sparse_retriever.search(
            normalized_query,
            self.sparse_top_k,
        )
        fused_results = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            rrf_k=self.rrf_k,
            top_k=top_k,
        )
        logger.info(
            "hybrid_search_completed dense_candidates=%d sparse_candidates=%d "
            "hybrid_candidates=%d rrf_k=%d latency_ms=%.2f",
            len(dense_results),
            len(sparse_results),
            len(fused_results),
            self.rrf_k,
            (perf_counter() - started_at) * 1000,
        )
        return fused_results
