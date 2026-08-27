"""Qdrant vector storage and dense retrieval."""

import logging
import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.models import Chunk
from app.retrieval.models import RetrievedChunk, RetrievedChunkMetadata

logger = logging.getLogger(__name__)
_POINT_NAMESPACE = uuid.UUID("512135c8-20e1-4bba-a1cf-a3a76d22420a")


class CollectionConfigurationError(RuntimeError):
    """Raised when an existing collection is incompatible with the model."""


class RetrievalPayloadError(RuntimeError):
    """Raised when a stored point lacks required chunk metadata."""


class QdrantVectorStore:
    """Small Qdrant wrapper that owns collection and payload conventions."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        if not collection_name.strip():
            raise ValueError("Qdrant collection name cannot be empty.")
        self.client = client
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int, *, recreate: bool = False) -> bool:
        """Create a cosine collection or validate the existing one."""

        if vector_size < 1:
            raise ValueError("Vector size must be at least 1.")
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return True

        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise CollectionConfigurationError(
                "OpsRAG requires one unnamed vector configuration."
            )
        if vectors.size != vector_size or vectors.distance != models.Distance.COSINE:
            raise CollectionConfigurationError(
                "Existing Qdrant collection is incompatible with the configured "
                f"embedding model (expected size={vector_size}, cosine). "
                "Run the index command with --recreate to replace it explicitly."
            )
        return False

    def replace_documents(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        batch_size: int = 64,
    ) -> int:
        """Replace points belonging to supplied documents with deterministic IDs."""

        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match.")
        if batch_size < 1:
            raise ValueError("Qdrant batch size must be at least 1.")
        if not chunks:
            return 0

        document_ids = sorted({chunk.document_id for chunk in chunks})
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchAny(any=document_ids),
                        )
                    ]
                )
            ),
            wait=True,
        )

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, chunk.chunk_id)),
                vector=[float(value) for value in vector],
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )
        logger.info(
            "qdrant_documents_replaced collection=%s documents=%d chunks=%d",
            self.collection_name,
            len(document_ids),
            len(points),
        )
        return len(points)

    def search(self, query_vector: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Query Qdrant and normalize stored payloads."""

        if not query_vector:
            raise ValueError("Query vector cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=[float(value) for value in query_vector],
            limit=top_k,
            with_payload=True,
        )
        return [_point_to_retrieved_chunk(point) for point in response.points]


class DenseRetriever:
    """Embed a query and execute vector search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        vector = self.embedding_service.embed_query(query)
        return self.vector_store.search(vector, top_k)


def _point_to_retrieved_chunk(point: models.ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    try:
        text = str(payload["text"])
        metadata = RetrievedChunkMetadata.model_validate(payload)
    except Exception as exc:
        raise RetrievalPayloadError(
            f"Qdrant point '{point.id}' has invalid chunk payload."
        ) from exc
    return RetrievedChunk(
        text=text,
        metadata=metadata,
        score=float(point.score),
    )

