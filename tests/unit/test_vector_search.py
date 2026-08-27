"""Qdrant storage and dense retrieval behavior."""

import pytest
from qdrant_client import QdrantClient, models

from app.retrieval.vector_search import (
    CollectionConfigurationError,
    DenseRetriever,
    QdrantVectorStore,
)
from tests.helpers import make_chunk


class FakeEmbeddingService:
    model_name = "fake"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(location=":memory:"), "chunks")


def test_collection_is_created_and_reused() -> None:
    store = _store()
    assert store.ensure_collection(2) is True
    assert store.ensure_collection(2) is False


def test_incompatible_collection_requires_explicit_recreation() -> None:
    store = _store()
    store.ensure_collection(3)
    with pytest.raises(CollectionConfigurationError, match="--recreate"):
        store.ensure_collection(2)
    assert store.ensure_collection(2, recreate=True) is True


def test_replace_documents_and_search_returns_payload() -> None:
    store = _store()
    store.ensure_collection(2)
    first = make_chunk(chunk_id="one", text="first")
    second = make_chunk(chunk_id="two", document_id="doc_2", text="second")
    assert store.replace_documents([first, second], [[1.0, 0.0], [0.0, 1.0]]) == 2

    results = store.search([1.0, 0.0], 2)

    assert results[0].metadata.chunk_id == "one"
    assert results[0].text == "first"
    assert results[0].retrieval_method == "dense"


def test_reindex_replaces_points_for_same_document() -> None:
    store = _store()
    store.ensure_collection(2)
    old = make_chunk(chunk_id="old")
    new = make_chunk(chunk_id="new")
    store.replace_documents([old], [[1.0, 0.0]])
    store.replace_documents([new], [[1.0, 0.0]])

    results = store.search([1.0, 0.0], 10)

    assert [result.metadata.chunk_id for result in results] == ["new"]


def test_dense_retriever_embeds_query() -> None:
    store = _store()
    store.ensure_collection(2)
    store.replace_documents([make_chunk()], [[1.0, 0.0]])
    retriever = DenseRetriever(FakeEmbeddingService(), store)
    assert retriever.search("logs", 1)[0].metadata.source == "guide.md"


def test_invalid_search_inputs_are_rejected() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.search([], 1)
    with pytest.raises(ValueError):
        store.search([1.0], 0)

