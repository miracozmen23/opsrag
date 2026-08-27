"""Normalized result models shared by retrieval implementations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.models import DocumentType


class RetrievedChunkMetadata(BaseModel):
    """Source metadata returned with a retrieved chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    section_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=1)
    document_type: DocumentType | None = None
    page_number: int | None = Field(default=None, ge=1)


class RetrievedChunk(BaseModel):
    """Provider-neutral retrieval result."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    metadata: RetrievedChunkMetadata
    score: float
    retrieval_method: Literal["dense", "bm25", "hybrid"] = "dense"
