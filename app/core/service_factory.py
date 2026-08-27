"""Factories for external retrieval services."""

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.embeddings.embedding_service import SentenceTransformerEmbeddingService


def create_embedding_service(settings: Settings) -> SentenceTransformerEmbeddingService:
    """Create the configured lazy embedding service."""

    return SentenceTransformerEmbeddingService(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
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

