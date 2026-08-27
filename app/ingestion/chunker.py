"""Deterministic boundary-aware token chunking."""

import hashlib
import re
from collections.abc import Sequence

from app.ingestion.models import Chunk, Document

_TOKEN_PATTERN = re.compile(r"\w+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Count deterministic regex tokens without a model download."""

    return len(_TOKEN_PATTERN.findall(text))


class RecursiveTextChunker:
    """Split long sections with paragraph, line, and sentence boundary preference."""

    def __init__(self, *, chunk_size_tokens: int = 600, overlap_tokens: int = 75) -> None:
        if chunk_size_tokens < 2:
            raise ValueError("Chunk size must be at least 2 tokens.")
        if overlap_tokens < 0:
            raise ValueError("Chunk overlap cannot be negative.")
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def split_text(self, text: str) -> list[str]:
        """Return deterministic chunks while preferring natural text boundaries."""

        normalized = text.strip()
        matches = list(_TOKEN_PATTERN.finditer(normalized))
        if not matches:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(matches):
            hard_end = min(start + self.chunk_size_tokens, len(matches))
            end = self._preferred_end(normalized, matches, start, hard_end)
            char_start = matches[start].start()
            char_end = matches[end - 1].end()
            chunks.append(normalized[char_start:char_end].strip())
            if end >= len(matches):
                break
            start = max(start + 1, end - self.overlap_tokens)
        return chunks

    def _preferred_end(
        self,
        text: str,
        matches: list[re.Match[str]],
        start: int,
        hard_end: int,
    ) -> int:
        if hard_end >= len(matches):
            return hard_end
        minimum = start + max(1, int(self.chunk_size_tokens * 0.6))
        for end in range(hard_end, minimum - 1, -1):
            left_end = matches[end - 1].end()
            right_start = matches[end].start()
            gap = text[left_end:right_start]
            left_text = text[:left_end].rstrip()
            if "\n\n" in gap:
                return end
            if "\n" in gap:
                return end
            if left_text.endswith((".", "!", "?", ":", ";")):
                return end
        return hard_end


def chunk_documents(
    documents: Sequence[Document],
    chunker: RecursiveTextChunker,
) -> list[Chunk]:
    """Chunk documents in order and preserve all attribution metadata."""

    chunks: list[Chunk] = []
    for document in documents:
        document_chunk_index = 0
        for section in document.sections:
            for text in chunker.split_text(section.text):
                chunk_id = _chunk_id(
                    document.document_id,
                    section.section_index,
                    document_chunk_index,
                    text,
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        source=document.source,
                        title=document.title,
                        section=section.title,
                        section_index=section.section_index,
                        chunk_index=document_chunk_index,
                        text=text,
                        token_count=count_tokens(text),
                        document_type=document.document_type,
                        page_number=section.page_number,
                    )
                )
                document_chunk_index += 1
    return chunks


def _chunk_id(
    document_id: str,
    section_index: int,
    chunk_index: int,
    text: str,
) -> str:
    raw = f"{document_id}|{section_index}|{chunk_index}|{text}".encode("utf-8")
    return f"chunk_{hashlib.sha256(raw).hexdigest()[:20]}"

