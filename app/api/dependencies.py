"""Dependency construction for API routes."""

from functools import lru_cache

from fastapi import HTTPException, status

from app.core.config import get_settings, resolve_project_path
from app.core.service_factory import (
    create_embedding_service,
    create_qdrant_client,
    create_reranker_service,
)
from app.ingestion.pipeline import read_chunks_jsonl
from app.llm.factory import LLMConfigurationError, create_llm_service
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.vector_search import DenseRetriever, QdrantVectorStore


@lru_cache(maxsize=1)
def _build_rag_pipeline() -> RAGPipeline:
    settings = get_settings()
    llm = create_llm_service(settings)
    chunks = read_chunks_jsonl(resolve_project_path(settings.processed_chunks_path))
    embedding_service = create_embedding_service(settings)
    vector_store = QdrantVectorStore(
        create_qdrant_client(settings),
        settings.qdrant_collection,
    )
    dense_retriever = DenseRetriever(embedding_service, vector_store)
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        BM25Retriever(chunks),
        dense_top_k=settings.top_k_dense,
        sparse_top_k=settings.top_k_sparse,
        rrf_k=settings.rrf_k,
    )
    retriever = RerankingRetriever(
        hybrid_retriever,
        create_reranker_service(settings),
        candidate_top_k=settings.top_k_hybrid,
    )
    return RAGPipeline(
        retriever,
        llm,
        top_k=settings.top_k_rerank,
        retrieval_method="hybrid_reranked",
    )


def get_rag_pipeline() -> RAGPipeline:
    """Resolve the RAG service while translating setup errors to HTTP 503."""

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
