"""Contract and failure tests for the Streamlit HTTP client."""

from collections.abc import Callable

import httpx
import pytest

from frontend.client import APIClientError, OpsRAGAPIClient


SUCCESS_PAYLOAD = {
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


def build_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpsRAGAPIClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=transport,
        base_url="http://api.test",
    )
    return OpsRAGAPIClient("http://api.test", client=http_client)


def test_client_posts_question_and_validates_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/ask"
        assert request.content == b'{"question":"Why is PostgreSQL unavailable?"}'
        return httpx.Response(200, json=SUCCESS_PAYLOAD)

    result = build_client(handler).ask("  Why is PostgreSQL unavailable?  ")

    assert result.answer.endswith("[S1].")
    assert result.sources[0].document == "postgresql_troubleshooting.md"
    assert result.metadata.retrieval_method == "hybrid_reranked"


def test_client_exposes_safe_api_error_detail() -> None:
    client = build_client(
        lambda request: httpx.Response(
            503,
            json={"detail": "Answer generation failed."},
        )
    )

    with pytest.raises(APIClientError, match="Answer generation failed"):
        client.ask("question")


def test_client_falls_back_to_status_for_unstructured_error() -> None:
    client = build_client(lambda request: httpx.Response(502, text="proxy failure"))

    with pytest.raises(APIClientError, match="HTTP 502"):
        client.ask("question")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx.ReadTimeout("slow"), "too long"),
        (httpx.ConnectError("offline"), "Could not reach"),
    ],
)
def test_client_translates_transport_failures(
    failure: httpx.RequestError,
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        failure.request = request
        raise failure

    with pytest.raises(APIClientError, match=message):
        build_client(handler).ask("question")


def test_client_rejects_malformed_success_response() -> None:
    client = build_client(lambda request: httpx.Response(200, json={"answer": "x"}))

    with pytest.raises(APIClientError, match="invalid response"):
        client.ask("question")


def test_client_rejects_blank_question_without_http_call() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP should not be called for a blank question")

    with pytest.raises(APIClientError, match="enter a question"):
        build_client(unexpected_request).ask("   ")
