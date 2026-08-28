"""Dependency construction for API routes."""

from functools import lru_cache

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.llm.factory import LLMConfigurationError, create_llm_service
from app.observability import ObservabilityClient, create_observability
from app.rag.graph import QueryRoutingGraph
from app.rag.models import RAGResult
from app.rag.pipeline import RAGPipeline
from app.retrieval.factory import create_retriever_suite


@lru_cache(maxsize=1)
def _build_observability() -> ObservabilityClient:
    return create_observability(get_settings())


@lru_cache(maxsize=1)
def _build_knowledge_pipeline() -> RAGPipeline:
    settings = get_settings()
    retriever = create_retriever_suite(settings).hybrid_reranked
    return RAGPipeline(
        retriever,
        create_llm_service(settings),
        top_k=settings.top_k_rerank,
        retrieval_method="hybrid_reranked",
        observability=_build_observability(),
    )


class _LazyKnowledgePipeline:
    """Delay retrieval setup until the graph actually selects the RAG branch."""

    def answer(self, question: str) -> RAGResult:
        return _build_knowledge_pipeline().answer(question)


@lru_cache(maxsize=1)
def _build_rag_pipeline() -> QueryRoutingGraph:
    settings = get_settings()
    llm = create_llm_service(settings)
    return QueryRoutingGraph(
        _LazyKnowledgePipeline(),
        llm,
        observability=_build_observability(),
    )


def get_rag_pipeline() -> QueryRoutingGraph:
    """Resolve the routed answer service while translating setup errors to HTTP 503."""

    try:
        return _build_rag_pipeline()
    except (LLMConfigurationError, ModuleNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def clear_dependency_cache() -> None:
    """Clear cached services, primarily for isolated tests."""

    _build_rag_pipeline.cache_clear()
    _build_knowledge_pipeline.cache_clear()
    if _build_observability.cache_info().currsize:
        _build_observability().shutdown()
    _build_observability.cache_clear()
