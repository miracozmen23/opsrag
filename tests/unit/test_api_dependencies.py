"""API service construction for the final retrieval path."""

from pathlib import Path

from app.api import dependencies
from app.core.config import Settings
from app.ingestion.pipeline import write_chunks_jsonl
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import RerankingRetriever
from tests.helpers import make_chunk


class FakeLanguageModel:
    provider_name = "fake"
    model_name = "fake-model"

    def generate(self, *, instructions: str, input_text: str) -> str:
        return "answer"


def test_api_builds_hybrid_reranked_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl([make_chunk()], chunks_path)
    settings = Settings(
        _env_file=None,
        processed_chunks_path=chunks_path,
        model_cache_dir=tmp_path / "model-cache",
        top_k_dense=8,
        top_k_sparse=7,
        top_k_hybrid=6,
        top_k_rerank=4,
        rrf_k=55,
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(
        dependencies,
        "create_llm_service",
        lambda configured_settings: FakeLanguageModel(),
    )

    graph = dependencies._build_rag_pipeline.__wrapped__()
    pipeline = dependencies._build_knowledge_pipeline.__wrapped__()

    assert graph.llm.model_name == "fake-model"
    assert pipeline.top_k == 4
    assert pipeline.retrieval_method == "hybrid_reranked"
    assert isinstance(pipeline.retriever, RerankingRetriever)
    assert pipeline.retriever.candidate_top_k == 6
    hybrid = pipeline.retriever.candidate_retriever
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.dense_top_k == 8
    assert hybrid.sparse_top_k == 7
    assert hybrid.rrf_k == 55
    assert hybrid.sparse_retriever.corpus_size == 1


def test_general_graph_setup_does_not_read_chunks(
    monkeypatch,
) -> None:
    settings = Settings(_env_file=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(
        dependencies,
        "create_llm_service",
        lambda configured_settings: FakeLanguageModel(),
    )
    monkeypatch.setattr(
        dependencies,
        "create_retriever_suite",
        lambda settings: (_ for _ in ()).throw(
            AssertionError("retrievers were constructed")
        ),
    )

    graph = dependencies._build_rag_pipeline.__wrapped__()
    result = graph.answer("Hello")

    assert result.metadata.route == "general"
    assert result.metadata.retrieval_method == "not_used"
