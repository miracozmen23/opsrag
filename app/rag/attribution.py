"""Deterministic source grouping and answer citation validation."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.rag.models import RAGSource
from app.retrieval.models import RetrievedChunk

_EXACT_CITATION_PATTERN = re.compile(r"\[S([1-9][0-9]*)\]")
_SOURCE_LIKE_PATTERN = re.compile(r"\[S[^\]\r\n]*\]")


class SourceAttributionError(ValueError):
    """Raised when a generated answer has missing or invalid source citations."""


@dataclass(frozen=True)
class SourceContext:
    """One human-readable source and its retrieved excerpts."""

    source: RAGSource
    chunks: tuple[RetrievedChunk, ...]


def build_source_contexts(chunks: Sequence[RetrievedChunk]) -> list[SourceContext]:
    """Group duplicate document sections while preserving retrieval order."""

    grouped_chunks: dict[
        tuple[str, str, str, str, int | None],
        list[RetrievedChunk],
    ] = {}
    for chunk in chunks:
        document = _human_readable_document(chunk.metadata.source)
        title = chunk.metadata.title.strip() or document
        section = chunk.metadata.section.strip() or title
        key = (
            chunk.metadata.source.strip(),
            document,
            title,
            section,
            chunk.metadata.page_number,
        )
        group = grouped_chunks.setdefault(key, [])
        if all(
            existing.metadata.chunk_id != chunk.metadata.chunk_id
            for existing in group
        ):
            group.append(chunk)

    contexts: list[SourceContext] = []
    for index, (key, group) in enumerate(grouped_chunks.items(), start=1):
        _, document, title, section, page_number = key
        chunk_ids = tuple(chunk.metadata.chunk_id for chunk in group)
        source = RAGSource(
            source_id=f"S{index}",
            document=document,
            title=title,
            section=section,
            page_number=page_number,
            score=round(max(public_relevance_score(chunk) for chunk in group), 4),
            chunk_id=chunk_ids[0],
            chunk_ids=chunk_ids,
        )
        contexts.append(SourceContext(source=source, chunks=tuple(group)))
    return contexts


def select_cited_sources(
    answer: str,
    source_contexts: Sequence[SourceContext],
) -> list[RAGSource]:
    """Return valid cited sources once, ordered by first answer reference."""

    if not answer.strip():
        raise SourceAttributionError("Generated answer cannot be empty.")

    source_by_id = {
        source_context.source.source_id: source_context.source
        for source_context in source_contexts
    }
    citation_ids: list[str] = []
    for token in _SOURCE_LIKE_PATTERN.findall(answer):
        match = _EXACT_CITATION_PATTERN.fullmatch(token)
        if match is None:
            raise SourceAttributionError(
                f"Generated answer contains a malformed source citation: {token}"
            )
        source_id = f"S{match.group(1)}"
        if source_id not in source_by_id:
            raise SourceAttributionError(
                f"Generated answer cites an unknown source identifier: {source_id}"
            )
        if source_id not in citation_ids:
            citation_ids.append(source_id)

    if not citation_ids:
        raise SourceAttributionError(
            "Generated answer must cite at least one retrieved source."
        )
    return [source_by_id[source_id] for source_id in citation_ids]


def public_relevance_score(chunk: RetrievedChunk) -> float:
    """Return a stable public 0-1 relevance score for one retrieved chunk."""

    if chunk.rerank_score is None:
        return max(0.0, min(1.0, float(chunk.score)))
    return _sigmoid(float(chunk.rerank_score))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _human_readable_document(source: str) -> str:
    normalized = source.strip().replace("\\", "/")
    return PurePosixPath(normalized).name or normalized
