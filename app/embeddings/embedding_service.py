"""Configurable sentence-transformer embedding service."""

import logging
import math
from collections.abc import Sequence
from time import perf_counter
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    """Interface consumed by indexing and dense retrieval."""

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


class SentenceTransformerEmbeddingService:
    """Lazy sentence-transformer wrapper with normalized output vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Embedding model name cannot be empty.")
        if batch_size < 1:
            raise ValueError("Embedding batch size must be at least 1.")
        self._model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        model = self._get_model()
        dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model,
            "get_sentence_embedding_dimension",
        )
        dimension = dimension_getter()
        if not isinstance(dimension, int) or dimension < 1:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' returned an invalid dimension."
            )
        return dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Document text cannot be empty.")

        started_at = perf_counter()
        try:
            model = self._get_model()
            encode = getattr(model, "encode_document", None) or getattr(model, "encode")
            raw_vectors = encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = _coerce_vectors(raw_vectors, len(texts), self.dimension)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding documents with '{self.model_name}' failed: {exc}"
            ) from exc

        logger.info(
            "documents_embedded model=%s count=%d dimension=%d latency_ms=%.2f",
            self.model_name,
            len(vectors),
            self.dimension,
            (perf_counter() - started_at) * 1000,
        )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        started_at = perf_counter()
        try:
            model = self._get_model()
            encode = getattr(model, "encode_query", None) or getattr(model, "encode")
            raw_vector = encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vector = _coerce_vectors(raw_vector, 1, self.dimension)[0]
        except Exception as exc:
            raise RuntimeError(
                f"Embedding query with '{self.model_name}' failed: {exc}"
            ) from exc

        logger.info(
            "query_embedded model=%s dimension=%d latency_ms=%.2f",
            self.model_name,
            self.dimension,
            (perf_counter() - started_at) * 1000,
        )
        return vector

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "embedding_model_loading model=%s device=%s",
                self.model_name,
                self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


def _coerce_vectors(
    raw_vectors: Any,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    value = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else raw_vectors
    if expected_count == 1 and value and isinstance(value[0], (int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)) or len(value) != expected_count:
        raise ValueError(
            f"Expected {expected_count} embedding vectors, received an incompatible result."
        )

    vectors: list[list[float]] = []
    for raw_vector in value:
        if not isinstance(raw_vector, (list, tuple)):
            raise ValueError("Embedding result contains a non-vector value.")
        vector = [float(component) for component in raw_vector]
        if len(vector) != expected_dimension:
            raise ValueError(
                f"Expected embedding dimension {expected_dimension}, got {len(vector)}."
            )
        if not all(math.isfinite(component) for component in vector):
            raise ValueError("Embedding result contains a non-finite value.")
        vectors.append(vector)
    return vectors
