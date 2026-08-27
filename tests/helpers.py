"""Shared factories used by unit and integration tests."""

from typing import Literal

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievedChunk, RetrievedChunkMetadata


def make_chunk(
    *,
    chunk_id: str = "chunk_1",
    document_id: str = "doc_1",
    source: str = "guide.md",
    title: str = "Guide",
    section: str = "Troubleshooting",
    chunk_index: int = 0,
    text: str = "Check the service logs and verify the configured port.",
    token_count: int = 10,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source=source,
        title=title,
        section=section,
        section_index=0,
        chunk_index=chunk_index,
        text=text,
        token_count=token_count,
        document_type="markdown",
    )


def make_retrieved_chunk(
    *,
    chunk_id: str = "chunk_1",
    source: str = "guide.md",
    title: str = "Guide",
    section: str = "Troubleshooting",
    page_number: int | None = None,
    text: str = "Check the service logs and verify the configured port.",
    score: float = 0.82,
    rerank_score: float | None = None,
    retrieval_method: Literal["dense", "bm25", "hybrid"] = "dense",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=RetrievedChunkMetadata(
            chunk_id=chunk_id,
            document_id=f"doc_{chunk_id}",
            source=source,
            title=title,
            section=section,
            section_index=0,
            chunk_index=0,
            token_count=10,
            document_type="markdown",
            page_number=page_number,
        ),
        score=score,
        rerank_score=rerank_score,
        retrieval_method=retrieval_method,
    )
