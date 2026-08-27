"""Embedding wrapper behavior."""

import pytest

from app.embeddings.embedding_service import SentenceTransformerEmbeddingService


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        assert kwargs["normalize_embeddings"] is True
        return [[1.0, 0.0, 0.0] for _ in texts]

    def encode_query(self, query: str, **kwargs: object) -> list[float]:
        assert query
        assert kwargs["normalize_embeddings"] is True
        return [0.0, 1.0, 0.0]


def test_embedding_service_preserves_input_order_and_dimension() -> None:
    service = SentenceTransformerEmbeddingService("fake", model=FakeEmbeddingModel())
    assert service.embed_documents(["a", "b"]) == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert service.dimension == 3


def test_embedding_service_uses_query_encoder() -> None:
    service = SentenceTransformerEmbeddingService("fake", model=FakeEmbeddingModel())
    assert service.embed_query("question") == [0.0, 1.0, 0.0]


def test_embedding_service_rejects_blank_input() -> None:
    service = SentenceTransformerEmbeddingService("fake", model=FakeEmbeddingModel())
    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_query("  ")
    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_documents(["ok", ""])


def test_embedding_service_rejects_wrong_dimension() -> None:
    class WrongDimensionModel(FakeEmbeddingModel):
        def encode_query(self, query: str, **kwargs: object) -> list[float]:
            return [1.0]

    service = SentenceTransformerEmbeddingService("fake", model=WrongDimensionModel())
    with pytest.raises(RuntimeError, match="dimension"):
        service.embed_query("question")

