"""Cross-encoder reranking for hybrid retrieval candidates."""

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Contract for query-aware candidate reranking."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Score candidates jointly with a query and return the best matches."""

        ...


class CandidateRetriever(Protocol):
    """Retrieval contract consumed by the reranking wrapper."""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return a ranked candidate list."""

        ...


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder reranker."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        cache_folder: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Reranker model name cannot be empty.")
        if batch_size < 1:
            raise ValueError("Reranker batch size must be at least 1.")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_folder = Path(cache_folder) if cache_folder is not None else None
        self._model = model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Assign query-aware scores and return a stable descending ranking."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Reranking query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not candidates:
            return []

        started_at = perf_counter()
        try:
            raw_scores = self._get_model().predict(
                [(normalized_query, candidate.text) for candidate in candidates],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = _coerce_scores(raw_scores, len(candidates))
        except Exception as exc:
            raise RuntimeError(
                f"Reranking with '{self.model_name}' failed: {exc}"
            ) from exc

        ranked = sorted(
            enumerate(zip(candidates, scores, strict=True)),
            key=lambda item: (-item[1][1], item[0]),
        )
        results = [
            candidate.model_copy(update={"rerank_score": score})
            for _, (candidate, score) in ranked[:top_k]
        ]
        logger.info(
            "reranking_completed model=%s candidates=%d selected=%d latency_ms=%.2f",
            self.model_name,
            len(candidates),
            len(results),
            (perf_counter() - started_at) * 1000,
        )
        return results

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            if self.cache_folder is not None:
                self.cache_folder.mkdir(parents=True, exist_ok=True)
            logger.info(
                "reranker_model_loading model=%s device=%s cache_folder=%s",
                self.model_name,
                self.device,
                self.cache_folder,
            )
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                cache_folder=(
                    str(self.cache_folder.resolve())
                    if self.cache_folder is not None
                    else None
                ),
            )
        return self._model


class RerankingRetriever:
    """Retrieve a broad candidate set and select final context by reranking."""

    def __init__(
        self,
        candidate_retriever: CandidateRetriever,
        reranker: Reranker,
        *,
        candidate_top_k: int = 10,
    ) -> None:
        if candidate_top_k < 1:
            raise ValueError("Reranking candidate count must be at least 1.")
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return final query-aware context from hybrid candidates."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        started_at = perf_counter()
        requested_candidates = max(self.candidate_top_k, top_k)
        candidates = self.candidate_retriever.search(
            normalized_query,
            requested_candidates,
        )
        results = self.reranker.rerank(normalized_query, candidates, top_k)
        logger.info(
            "reranked_search_completed candidates=%d selected=%d latency_ms=%.2f",
            len(candidates),
            len(results),
            (perf_counter() - started_at) * 1000,
        )
        return results


def _coerce_scores(raw_scores: Any, expected_count: int) -> list[float]:
    values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if expected_count == 1 and isinstance(values, (int, float)):
        values = [values]
    if not isinstance(values, (list, tuple)) or len(values) != expected_count:
        raise ValueError(
            f"Expected {expected_count} reranker scores, received an incompatible result."
        )

    scores: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("Reranker result contains a non-scalar score.")
            value = value[0]
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("Reranker result contains a non-finite score.")
        scores.append(score)
    return scores
