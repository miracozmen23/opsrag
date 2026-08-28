"""Typed HTTP client for the public OpsRAG API."""

from typing import Any

import httpx
from pydantic import ValidationError

from app.api.schemas import AskResponse


class APIClientError(RuntimeError):
    """A safe, user-facing FastAPI communication failure."""


class OpsRAGAPIClient:
    """Small synchronous client used by the Streamlit application."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("API base URL cannot be empty.")
        if timeout_seconds <= 0:
            raise ValueError("API timeout must be greater than zero.")

        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=normalized_url,
            timeout=timeout_seconds,
        )

    def ask(self, question: str) -> AskResponse:
        """Submit one question and validate the public response contract."""

        normalized_question = question.strip()
        if not normalized_question:
            raise APIClientError("Please enter a question.")

        try:
            response = self._client.post(
                "/api/v1/ask",
                json={"question": normalized_question},
            )
        except httpx.TimeoutException as exc:
            raise APIClientError(
                "The API took too long to respond. Please try again."
            ) from exc
        except httpx.RequestError as exc:
            raise APIClientError(
                "Could not reach the OpsRAG API. Make sure the backend is running."
            ) from exc

        if not response.is_success:
            raise APIClientError(self._response_error(response))

        try:
            payload = response.json()
            return AskResponse.model_validate(payload)
        except (ValueError, ValidationError, TypeError) as exc:
            raise APIClientError("The API returned an invalid response.") from exc

    def close(self) -> None:
        """Close the internally created connection pool."""

        if self._owns_client:
            self._client.close()

    @staticmethod
    def _response_error(response: httpx.Response) -> str:
        detail: Any = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail")
        except ValueError:
            pass

        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        return f"The OpsRAG API returned HTTP {response.status_code}."
