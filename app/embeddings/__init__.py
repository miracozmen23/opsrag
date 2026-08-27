"""Embedding service abstractions."""

from app.embeddings.embedding_service import (
    EmbeddingService,
    SentenceTransformerEmbeddingService,
)

__all__ = ["EmbeddingService", "SentenceTransformerEmbeddingService"]

