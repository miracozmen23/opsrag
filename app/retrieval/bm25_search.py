"""Deterministic lexical retrieval backed by BM25 Okapi."""

import logging
import re
from collections.abc import Sequence
from time import perf_counter

from rank_bm25 import BM25Okapi

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievedChunk, RetrievedChunkMetadata

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[\w]+(?:[./:@-][\w]+)*", re.UNICODE)


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize case-insensitively while preserving common technical terms."""

    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    """Build an in-memory BM25 index and return normalized retrieval results."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 index requires at least one chunk.")

        self._chunks = tuple(chunks)
        tokenized_corpus = [
            tokenize_bm25(_format_chunk_for_index(chunk)) for chunk in self._chunks
        ]
        self._corpus_token_sets = tuple(frozenset(tokens) for tokens in tokenized_corpus)
        self._index = BM25Okapi(tokenized_corpus)
        logger.info("bm25_index_built chunks=%d", len(self._chunks))

    @property
    def corpus_size(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return token-matching chunks ordered by BM25 relevance."""

        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return []

        started_at = perf_counter()
        scores = self._index.get_scores(query_tokens)
        query_token_set = frozenset(query_tokens)
        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), index),
        )
        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            score = float(scores[index])
            if not query_token_set.intersection(self._corpus_token_sets[index]):
                continue
            results.append(_to_retrieved_chunk(self._chunks[index], score))
            if len(results) == top_k:
                break
        logger.info(
            "bm25_search_completed corpus_chunks=%d matched_chunks=%d top_k=%d latency_ms=%.2f",
            len(self._chunks),
            len(results),
            top_k,
            (perf_counter() - started_at) * 1000,
        )
        return results


def _format_chunk_for_index(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.section}\n{chunk.text}"


def _to_retrieved_chunk(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=chunk.text,
        metadata=RetrievedChunkMetadata.model_validate(chunk.model_dump()),
        score=score,
        retrieval_method="bm25",
    )
