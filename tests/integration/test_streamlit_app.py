"""Streamlit interaction tests for the minimal demo."""

import socket
import time
from pathlib import Path
from threading import Thread

import uvicorn
from streamlit.testing.v1 import AppTest

from app.api.dependencies import get_rag_pipeline
from app.api.schemas import AskResponse
from app.main import app as fastapi_app
from app.rag.models import RAGMetadata, RAGResult
from frontend.client import OpsRAGAPIClient

APP_PATH = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"
EXPECTED_EXAMPLE_QUESTIONS = [
    "Why does PostgreSQL return connection refused in Docker Compose?",
    "How should I troubleshoot an HTTP 503 error?",
    "What should I check when a FastAPI application fails to start?",
    "How should environment variables and secrets be handled in production?",
]


class DemoPipeline:
    """Predictable service used by the real HTTP boundary test."""

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


def test_streamlit_app_renders_question_form() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert any("Resolve incidents" in item.value for item in app.markdown)
    assert app.text_area[0].label == "Technical question"
    assert app.button[0].label == "Generate grounded answer  →"
    assert [button.label for button in app.button[1:]] == EXPECTED_EXAMPLE_QUESTIONS


def test_streamlit_app_rejects_blank_submission() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    app.text_area[0].set_value("   ")
    app.button[0].click().run()

    assert not app.exception
    assert app.warning[0].value == "Please enter a question."


def test_streamlit_example_button_populates_question() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    app.button[1].click().run()

    assert not app.exception
    assert app.text_area[0].value == EXPECTED_EXAMPLE_QUESTIONS[0]


def test_streamlit_app_renders_answer_metadata_and_sources(monkeypatch) -> None:
    result = AskResponse.model_validate(
        {
            "answer": "Use the Compose service name [S1].",
            "sources": [
                {
                    "source_id": "S1",
                    "document": "postgresql_troubleshooting.md",
                    "title": "PostgreSQL Troubleshooting",
                    "section": "Connection refused",
                    "page_number": None,
                    "score": 0.8765,
                    "chunk_id": "chunk_1",
                    "chunk_ids": ["chunk_1"],
                }
            ],
            "retrieval_confidence": 0.8765,
            "metadata": {
                "retrieved_chunks": 1,
                "cited_sources": 1,
                "retrieval_method": "hybrid_reranked",
                "route": "knowledge",
            },
        }
    )

    def fake_ask(self: OpsRAGAPIClient, question: str) -> AskResponse:
        assert question == "Why is PostgreSQL unavailable?"
        return result

    monkeypatch.setattr(OpsRAGAPIClient, "ask", fake_ask)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    app.text_area[0].set_value("Why is PostgreSQL unavailable?")
    app.button[0].click().run()

    assert not app.exception
    assert any("Evidence-backed answer" in item.value for item in app.markdown)
    assert any(item.value == result.answer for item in app.markdown)
    assert [metric.value for metric in app.metric] == ["87.6%", "Knowledge", "1"]
    assert app.expander[0].label == "[S1] PostgreSQL Troubleshooting"


def test_frontend_client_communicates_with_fastapi_over_http() -> None:
    """Exercise the production client against the real FastAPI application."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(fastapi_app, log_level="warning", lifespan="off")
    )
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    fastapi_app.dependency_overrides[get_rag_pipeline] = lambda: DemoPipeline()

    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "FastAPI test server did not start"

        client = OpsRAGAPIClient(f"http://{host}:{port}", timeout_seconds=2)
        try:
            result = client.ask("Hello")
        finally:
            client.close()

        assert result.answer == "Hello! How can I help?"
        assert result.metadata.route == "general"
        assert result.sources == []
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        fastapi_app.dependency_overrides.clear()
