"""Vector indexing orchestration."""

from qdrant_client import QdrantClient

from app.ingestion.vector_indexer import VectorIndexer
from app.retrieval.vector_search import QdrantVectorStore
from tests.helpers import make_chunk


class FakeEmbeddingService:
    model_name = "fake-embedding"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts[0].startswith("Title: Guide")
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def test_vector_indexer_returns_summary() -> None:
    store = QdrantVectorStore(QdrantClient(location=":memory:"), "chunks")
    result = VectorIndexer(FakeEmbeddingService(), store).index([make_chunk()])
    assert result.chunks == 1
    assert result.documents == 1
    assert result.vector_size == 2
    assert result.collection_created is True

