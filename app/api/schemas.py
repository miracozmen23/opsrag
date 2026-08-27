"""Public API request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.models import RAGMetadata, RAGResult, RAGSource


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"] = "ok"


class AskRequest(BaseModel):
    """Validated RAG question payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value.strip()


class SourceResponse(RAGSource):
    """Source metadata exposed by the API."""


class AskMetadata(RAGMetadata):
    """Execution metadata exposed by the API."""


class AskResponse(RAGResult):
    """Grounded answer and its retrieval evidence."""

    sources: list[SourceResponse]
    metadata: AskMetadata

