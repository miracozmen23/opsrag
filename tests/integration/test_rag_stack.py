"""End-to-end retrieval coverage without external model or network calls."""

from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import QdrantClient

from app.api.dependencies import get_rag_pipeline
from app.ingestion.models import Chunk
from app.ingestion.vector_indexer import VectorIndexer, VectorIndexResult
from app.main import app
from app.rag.graph import QueryRoutingGraph
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.vector_search import DenseRetriever, QdrantVectorStore
from tests.helpers import make_chunk


class DeterministicEmbeddingService:
    """Map the small test corpus to stable, human-readable vectors."""

    model_name = "deterministic-test-embedding"
    dimension = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        if "postgresql" in normalized or "database_url" in normalized:
            return [1.0, 0.0, 0.0]
        if "503" in normalized or "gateway" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class DeterministicReranker:
    """Rank the PostgreSQL runbook first and retain production score semantics."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        assert "postgresql" in query.casefold()
        scored = [
            candidate.model_copy(
                update={
                    "rerank_score": (
                        3.0
                        if candidate.metadata.source
                        == "postgresql_troubleshooting.md"
                        else -1.0
                    )
                }
            )
            for candidate in candidates
        ]
        return sorted(
            scored,
            key=lambda candidate: -float(candidate.rerank_score),
        )[:top_k]


class DeterministicLanguageModel:
    """Return one grounded citation after validating the generated prompt."""

    provider_name = "test"
    model_name = "deterministic-test-llm"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def generate(self, *, instructions: str, input_text: str) -> str:
        assert "only with facts supported by the retrieved context" in instructions
        assert "postgresql_troubleshooting.md" in input_text
        assert "DATABASE_URL" in input_text
        self.inputs.append(input_text)
        return "Use the PostgreSQL Compose service name and port 5432 [S1]."


def _chunks() -> list[Chunk]:
    return [
        make_chunk(
            chunk_id="postgres_connection",
            document_id="postgres_doc",
            source="postgresql_troubleshooting.md",
            title="PostgreSQL Troubleshooting",
            section="Connection refused",
            text=(
                "In Docker Compose, connect to PostgreSQL with the database "
                "service name and port 5432. Verify DATABASE_URL and health."
            ),
        ),
        make_chunk(
            chunk_id="http_503",
            document_id="http_doc",
            source="http_troubleshooting.md",
            title="HTTP Troubleshooting",
            section="HTTP 503",
            text="An HTTP 503 can indicate an unhealthy upstream gateway.",
        ),
        make_chunk(
            chunk_id="secrets",
            document_id="security_doc",
            source="security.md",
            title="Security Guide",
            section="Secrets",
            text="Store production secrets outside source control.",
        ),
    ]


def _build_stack() -> tuple[
    QdrantClient,
    VectorIndexResult,
    RerankingRetriever,
    QueryRoutingGraph,
    DeterministicLanguageModel,
]:
    qdrant = QdrantClient(location=":memory:")
    embeddings = DeterministicEmbeddingService()
    chunks = _chunks()
    vector_store = QdrantVectorStore(qdrant, "integration_chunks")
    index_result = VectorIndexer(embeddings, vector_store).index(chunks)
    dense = DenseRetriever(embeddings, vector_store)
    hybrid = HybridRetriever(
        dense,
        BM25Retriever(chunks),
        dense_top_k=3,
        sparse_top_k=3,
        rrf_k=10,
    )
    retriever = RerankingRetriever(
        hybrid,
        DeterministicReranker(),
        candidate_top_k=3,
    )
    llm = DeterministicLanguageModel()
    pipeline = RAGPipeline(
        retriever,
        llm,
        top_k=2,
        retrieval_method="hybrid_reranked",
    )
    return qdrant, index_result, retriever, QueryRoutingGraph(pipeline, llm), llm


def test_qdrant_round_trip_feeds_hybrid_reranking() -> None:
    qdrant, index_result, retriever, _, _ = _build_stack()
    try:
        results = retriever.search(
            "Why does PostgreSQL return connection refused in Docker Compose?",
            top_k=2,
        )
    finally:
        qdrant.close()

    assert index_result.documents == 3
    assert index_result.chunks == 3
    assert index_result.vector_size == 3
    assert results[0].metadata.chunk_id == "postgres_connection"
    assert results[0].retrieval_method == "hybrid"
    assert results[0].rerank_score == 3.0


@pytest.mark.anyio
async def test_indexed_qdrant_context_reaches_grounded_ask_response() -> None:
    qdrant, _, _, graph, llm = _build_stack()
    app.dependency_overrides[get_rag_pipeline] = lambda: graph
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health_response = await client.get("/health")
            ask_response = await client.post(
                "/api/v1/ask",
                json={
                    "question": (
                        "Why does PostgreSQL return connection refused "
                        "in Docker Compose?"
                    )
                },
            )
    finally:
        app.dependency_overrides.clear()
        qdrant.close()

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["answer"].endswith("[S1].")
    assert body["sources"][0]["document"] == "postgresql_troubleshooting.md"
    assert body["sources"][0]["section"] == "Connection refused"
    assert body["sources"][0]["chunk_ids"] == ["postgres_connection"]
    assert body["retrieval_confidence"] == pytest.approx(0.9526)
    assert body["metadata"] == {
        "retrieved_chunks": 2,
        "cited_sources": 1,
        "retrieval_method": "hybrid_reranked",
        "route": "knowledge",
    }
    assert len(llm.inputs) == 1
