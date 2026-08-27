"""Source grouping, human-readable metadata, and citation validation."""

import pytest

from app.rag.attribution import (
    SourceAttributionError,
    build_source_contexts,
    select_cited_sources,
)
from tests.helpers import make_retrieved_chunk


def test_duplicate_document_sections_are_grouped_with_all_chunk_ids() -> None:
    contexts = build_source_contexts(
        [
            make_retrieved_chunk(
                chunk_id="a",
                source=r"private\runbooks\docker.md",
                title="Docker Runbook",
                section="Networking",
                rerank_score=0.0,
            ),
            make_retrieved_chunk(
                chunk_id="b",
                source=r"private\runbooks\docker.md",
                title="Docker Runbook",
                section="Networking",
                rerank_score=2.0,
            ),
        ]
    )

    assert len(contexts) == 1
    assert contexts[0].source.source_id == "S1"
    assert contexts[0].source.document == "docker.md"
    assert contexts[0].source.title == "Docker Runbook"
    assert contexts[0].source.section == "Networking"
    assert contexts[0].source.chunk_id == "a"
    assert contexts[0].source.chunk_ids == ("a", "b")
    assert contexts[0].source.score == 0.8808
    assert [chunk.metadata.chunk_id for chunk in contexts[0].chunks] == ["a", "b"]


def test_duplicate_chunk_id_is_not_repeated_within_source() -> None:
    chunk = make_retrieved_chunk(chunk_id="same")
    context = build_source_contexts([chunk, chunk])[0]
    assert context.source.chunk_ids == ("same",)
    assert len(context.chunks) == 1


def test_different_sections_and_pages_remain_distinct_sources() -> None:
    contexts = build_source_contexts(
        [
            make_retrieved_chunk(section="Install", page_number=1),
            make_retrieved_chunk(chunk_id="b", section="Operate", page_number=2),
        ]
    )
    assert [context.source.source_id for context in contexts] == ["S1", "S2"]
    assert [context.source.page_number for context in contexts] == [1, 2]


def test_same_filename_from_different_source_paths_remains_distinct() -> None:
    contexts = build_source_contexts(
        [
            make_retrieved_chunk(chunk_id="a", source=r"team-a\guide.md"),
            make_retrieved_chunk(chunk_id="b", source=r"team-b\guide.md"),
        ]
    )
    assert len(contexts) == 2
    assert [context.source.document for context in contexts] == ["guide.md", "guide.md"]


def test_only_cited_sources_are_returned_in_first_citation_order() -> None:
    contexts = build_source_contexts(
        [
            make_retrieved_chunk(chunk_id="a", source="a.md"),
            make_retrieved_chunk(chunk_id="b", source="b.md"),
        ]
    )
    sources = select_cited_sources(
        "Second source [S2], then first [S1], and second again [S2].",
        contexts,
    )
    assert [source.source_id for source in sources] == ["S2", "S1"]


@pytest.mark.parametrize(
    "answer",
    [
        "Unsupported citation [S99].",
        "Combined citation [S1, S2].",
        "Malformed citation [S0].",
        "Answer without any citation.",
    ],
)
def test_invalid_or_missing_citations_are_rejected(answer: str) -> None:
    contexts = build_source_contexts([make_retrieved_chunk()])
    with pytest.raises(SourceAttributionError):
        select_cited_sources(answer, contexts)
