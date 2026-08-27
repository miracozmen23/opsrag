"""BM25 lexical retrieval behavior."""

import pytest

from app.ingestion.models import Chunk
from app.retrieval.bm25_search import BM25Retriever, tokenize_bm25
from tests.helpers import make_chunk


def _chunks() -> list[Chunk]:
    return [
        make_chunk(
            chunk_id="postgres",
            source="postgresql.md",
            text="Set DATABASE_URL to postgresql://db:5432/app when ECONNREFUSED occurs.",
        ),
        make_chunk(
            chunk_id="docker",
            document_id="doc_2",
            source="docker.md",
            text="Inspect the Docker Compose network and service health checks.",
        ),
        make_chunk(
            chunk_id="logging",
            document_id="doc_3",
            source="logging.md",
            text="Configure structured application logs and request identifiers.",
        ),
    ]


def test_tokenizer_preserves_technical_terms_case_insensitively() -> None:
    assert tokenize_bm25("DATABASE_URL ECONNREFUSED api/v1 port:5432") == [
        "database_url",
        "econnrefused",
        "api/v1",
        "port:5432",
    ]


def test_exact_technical_terms_rank_the_expected_chunk_first() -> None:
    retriever = BM25Retriever(_chunks())

    results = retriever.search("ECONNREFUSED DATABASE_URL", top_k=3)

    assert results[0].metadata.source == "postgresql.md"
    assert results[0].score > 0
    assert results[0].retrieval_method == "bm25"
    assert results[0].text.startswith("Set DATABASE_URL")


def test_search_respects_top_k_and_is_deterministic() -> None:
    retriever = BM25Retriever(_chunks())
    first = retriever.search("service application", top_k=1)
    second = retriever.search("service application", top_k=1)

    assert len(first) == 1
    assert first == second


def test_unmatched_query_returns_no_results() -> None:
    assert BM25Retriever(_chunks()).search("ZXCVBNM_NOT_PRESENT", top_k=3) == []


def test_exact_match_survives_zero_idf_in_a_small_corpus() -> None:
    chunks = [
        make_chunk(chunk_id="one", text="rareterm first"),
        make_chunk(chunk_id="two", document_id="doc_2", text="other second"),
    ]

    results = BM25Retriever(chunks).search("rareterm", top_k=1)

    assert [result.metadata.chunk_id for result in results] == ["one"]


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Retriever([])

    retriever = BM25Retriever(_chunks())
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.search("  ", top_k=1)
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search("logs", top_k=0)
