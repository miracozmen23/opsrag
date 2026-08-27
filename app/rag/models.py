"""Domain models returned by the RAG pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RAGSource(BaseModel):
    """One retrieved source made visible with an answer."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    document: str = Field(min_length=1)
    section: str = Field(min_length=1)
    score: float
    chunk_id: str = Field(min_length=1)


class RAGMetadata(BaseModel):
    """Basic retrieval execution metadata."""

    model_config = ConfigDict(frozen=True)

    retrieved_chunks: int = Field(ge=0)
    retrieval_method: Literal["dense"] = "dense"


class RAGResult(BaseModel):
    """Grounded answer result shared by service and API layers."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: list[RAGSource]
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    metadata: RAGMetadata

