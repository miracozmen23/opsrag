"""Orchestrate chunk embedding and safe Qdrant replacement."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.models import Chunk
from app.retrieval.vector_search import QdrantVectorStore


class VectorIndexResult(BaseModel):
    """Summary of one completed vector indexing operation."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(min_length=1)
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    embedding_model: str = Field(min_length=1)
    vector_size: int = Field(ge=1)
    collection_created: bool


class VectorIndexer:
    """Create embeddings before replacing matching document points."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        *,
        qdrant_batch_size: int = 64,
    ) -> None:
        if qdrant_batch_size < 1:
            raise ValueError("Qdrant batch size must be at least 1.")
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.qdrant_batch_size = qdrant_batch_size

    def index(self, chunks: Sequence[Chunk], *, recreate: bool = False) -> VectorIndexResult:
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")

        embedding_inputs = [_format_chunk_for_embedding(chunk) for chunk in chunks]
        vectors = self.embedding_service.embed_documents(embedding_inputs)
        vector_size = self.embedding_service.dimension
        collection_created = self.vector_store.ensure_collection(
            vector_size,
            recreate=recreate,
        )
        indexed = self.vector_store.replace_documents(
            chunks,
            vectors,
            batch_size=self.qdrant_batch_size,
        )
        return VectorIndexResult(
            collection=self.vector_store.collection_name,
            documents=len({chunk.document_id for chunk in chunks}),
            chunks=indexed,
            embedding_model=self.embedding_service.model_name,
            vector_size=vector_size,
            collection_created=collection_created,
        )


def _format_chunk_for_embedding(chunk: Chunk) -> str:
    labels = [f"Title: {chunk.title}"]
    if chunk.section != chunk.title:
        labels.append(f"Section: {chunk.section}")
    return "\n".join(labels) + f"\n\n{chunk.text}"

