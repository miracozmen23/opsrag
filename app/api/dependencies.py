"""Dependency construction for API routes."""

from functools import lru_cache

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.service_factory import create_embedding_service, create_qdrant_client
from app.llm.factory import LLMConfigurationError, create_llm_service
from app.rag.pipeline import RAGPipeline
from app.retrieval.vector_search import DenseRetriever, QdrantVectorStore


@lru_cache(maxsize=1)
def _build_rag_pipeline() -> RAGPipeline:
    settings = get_settings()
    embedding_service = create_embedding_service(settings)
    vector_store = QdrantVectorStore(
        create_qdrant_client(settings),
        settings.qdrant_collection,
    )
    retriever = DenseRetriever(embedding_service, vector_store)
    llm = create_llm_service(settings)
    return RAGPipeline(
        retriever,
        llm,
        top_k=settings.top_k_dense,
    )


def get_rag_pipeline() -> RAGPipeline:
    """Resolve the RAG service while translating setup errors to HTTP 503."""

    try:
        return _build_rag_pipeline()
    except (LLMConfigurationError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def clear_dependency_cache() -> None:
    """Clear cached services, primarily for isolated tests."""

    _build_rag_pipeline.cache_clear()

