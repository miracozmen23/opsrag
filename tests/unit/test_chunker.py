"""Deterministic chunking behavior."""

import pytest

from app.ingestion.chunker import RecursiveTextChunker, chunk_documents, count_tokens
from app.ingestion.models import Document, ParsedSection


def _document(text: str) -> Document:
    return Document(
        document_id="doc_a",
        source="a.md",
        title="A",
        document_type="markdown",
        sections=[ParsedSection(section_index=0, title="Section", text=text)],
    )


def test_token_count_includes_punctuation_deterministically() -> None:
    assert count_tokens("hello, world!") == 4


def test_short_section_stays_in_one_chunk() -> None:
    chunks = RecursiveTextChunker(chunk_size_tokens=10, overlap_tokens=2).split_text(
        "one two three"
    )
    assert chunks == ["one two three"]


def test_long_section_uses_overlap() -> None:
    text = " ".join(f"token{index}" for index in range(20))
    chunks = RecursiveTextChunker(chunk_size_tokens=10, overlap_tokens=2).split_text(text)
    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_chunking_is_deterministic() -> None:
    text = "First sentence. " * 20
    chunker = RecursiveTextChunker(chunk_size_tokens=12, overlap_tokens=3)
    assert chunk_documents([_document(text)], chunker) == chunk_documents(
        [_document(text)], chunker
    )


def test_chunk_metadata_and_ids_are_preserved() -> None:
    chunks = chunk_documents(
        [_document("one two three four five six")],
        RecursiveTextChunker(chunk_size_tokens=4, overlap_tokens=1),
    )
    assert chunks[0].document_id == "doc_a"
    assert chunks[0].section == "Section"
    assert chunks[0].chunk_id.startswith("chunk_")
    assert chunks[0].token_count <= 4


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(1, 0), (10, -1), (10, 10)],
)
def test_invalid_chunk_configuration_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size_tokens=size, overlap_tokens=overlap)

