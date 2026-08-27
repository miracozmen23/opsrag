"""Validated document and chunk models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentType = Literal["markdown", "text", "pdf"]


class ParsedSection(BaseModel):
    """One title-preserving section extracted from a source document."""

    model_config = ConfigDict(frozen=True)

    section_index: int = Field(ge=0)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class Document(BaseModel):
    """Normalized source document before chunking."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    document_type: DocumentType
    sections: list[ParsedSection] = Field(min_length=1)


class Chunk(BaseModel):
    """Retrieval-ready text chunk with source attribution metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    section_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)
    document_type: DocumentType
    page_number: int | None = Field(default=None, ge=1)

