"""FastAPI contract tests with an injected RAG service."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_rag_pipeline
from app.main import app
from app.rag.models import RAGMetadata, RAGResult, RAGSource
from app.rag.pipeline import RAGPipelineError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class SuccessfulPipeline:
    def answer(self, question: str) -> RAGResult:
        assert question == "Why is PostgreSQL unavailable?"
        return RAGResult(
            answer="Use the Compose service name [S1].",
            sources=[
                RAGSource(
                    source_id="S1",
                    document="postgresql_troubleshooting.md",
                    title="PostgreSQL Troubleshooting",
                    section="Connection refused",
                    score=0.8765,
                    chunk_id="chunk_1",
                    chunk_ids=("chunk_1",),
                )
            ],
            retrieval_confidence=0.8765,
            metadata=RAGMetadata(retrieved_chunks=1, cited_sources=1),
        )


@pytest.mark.anyio
async def test_health_returns_ok_without_external_dependencies() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_ask_returns_grounded_response_contract() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: SuccessfulPipeline()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/ask",
                json={"question": " Why is PostgreSQL unavailable? "},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[S1].")
    assert body["sources"][0]["document"] == "postgresql_troubleshooting.md"
    assert body["sources"][0]["title"] == "PostgreSQL Troubleshooting"
    assert body["sources"][0]["chunk_ids"] == ["chunk_1"]
    assert body["retrieval_confidence"] == 0.8765
    assert body["metadata"] == {
        "retrieved_chunks": 1,
        "cited_sources": 1,
        "retrieval_method": "dense",
        "route": "knowledge",
    }


@pytest.mark.anyio
async def test_ask_exposes_general_route_without_retrieval_metadata() -> None:
    class SuccessfulGeneralPipeline:
        def answer(self, question: str) -> RAGResult:
            assert question == "Hello"
            return RAGResult(
                answer="Hello! How can I help?",
                sources=[],
                retrieval_confidence=0.0,
                metadata=RAGMetadata(
                    retrieved_chunks=0,
                    cited_sources=0,
                    retrieval_method="not_used",
                    route="general",
                ),
            )

    app.dependency_overrides[get_rag_pipeline] = lambda: SuccessfulGeneralPipeline()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/v1/ask", json={"question": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["retrieval_confidence"] == 0.0
    assert body["metadata"] == {
        "retrieved_chunks": 0,
        "cited_sources": 0,
        "retrieval_method": "not_used",
        "route": "general",
    }


@pytest.mark.anyio
async def test_ask_rejects_blank_question() -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: SuccessfulPipeline()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/v1/ask", json={"question": "   "})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


@pytest.mark.anyio
async def test_ask_hides_internal_pipeline_failure() -> None:
    class FailingPipeline:
        def answer(self, question: str) -> RAGResult:
            raise RAGPipelineError("Answer generation failed.")

    app.dependency_overrides[get_rag_pipeline] = lambda: FailingPipeline()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/v1/ask", json={"question": "question"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json() == {"detail": "Answer generation failed."}
