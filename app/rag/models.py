"""Domain models returned by the RAG pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.models import RetrievedChunk


class RAGSource(BaseModel):
    """One retrieved source made visible with an answer."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    document: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    score: float = Field(ge=0.0, le=1.0)
    chunk_id: str = Field(min_length=1)
    chunk_ids: tuple[str, ...] = Field(min_length=1)


class RAGMetadata(BaseModel):
    """Basic retrieval execution metadata."""

    model_config = ConfigDict(frozen=True)

    retrieved_chunks: int = Field(ge=0)
    cited_sources: int = Field(default=0, ge=0)
    retrieval_method: Literal[
        "dense", "hybrid", "hybrid_reranked", "not_used"
    ] = "dense"
    route: Literal["knowledge", "general"] = "knowledge"


class RAGResult(BaseModel):
    """Grounded answer result shared by service and API layers."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: list[RAGSource]
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    metadata: RAGMetadata


class RAGExecution(BaseModel):
    """Internal RAG result plus the exact ranked contexts used for generation."""

    model_config = ConfigDict(frozen=True)

    result: RAGResult
    retrieved_chunks: tuple[RetrievedChunk, ...]
