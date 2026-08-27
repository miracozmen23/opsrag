"""Factories for external retrieval services."""

from qdrant_client import QdrantClient

from app.core.config import Settings, resolve_project_path
from app.embeddings.embedding_service import SentenceTransformerEmbeddingService
from app.retrieval.reranker import CrossEncoderReranker


def create_embedding_service(settings: Settings) -> SentenceTransformerEmbeddingService:
    """Create the configured lazy embedding service."""

    return SentenceTransformerEmbeddingService(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        cache_folder=resolve_project_path(settings.model_cache_dir),
    )


def create_reranker_service(settings: Settings) -> CrossEncoderReranker:
    """Create the configured lazy cross-encoder reranker."""

    return CrossEncoderReranker(
        settings.reranker_model,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
        cache_folder=resolve_project_path(settings.model_cache_dir),
    )


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant HTTP client without connecting eagerly."""

    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
