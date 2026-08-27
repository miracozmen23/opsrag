"""Build the production retrieval configurations from one shared service graph."""

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings, resolve_project_path
from app.core.service_factory import (
    create_embedding_service,
    create_qdrant_client,
    create_reranker_service,
)
from app.ingestion.pipeline import read_chunks_jsonl
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.vector_search import DenseRetriever, QdrantVectorStore

RetrievalConfiguration = Literal["dense", "hybrid", "hybrid_reranked"]


@dataclass(frozen=True)
class RetrieverSuite:
    """The three comparable retrieval paths used by production and evaluation."""

    dense: DenseRetriever
    hybrid: HybridRetriever
    hybrid_reranked: RerankingRetriever

    def get(
        self,
        configuration: RetrievalConfiguration,
    ) -> DenseRetriever | HybridRetriever | RerankingRetriever:
        """Resolve one named retriever without rebuilding shared services."""

        return getattr(self, configuration)


def create_retriever_suite(settings: Settings) -> RetrieverSuite:
    """Create dense, hybrid, and reranked retrievers with production settings."""

    chunks = read_chunks_jsonl(resolve_project_path(settings.processed_chunks_path))
    dense = DenseRetriever(
        create_embedding_service(settings),
        QdrantVectorStore(
            create_qdrant_client(settings),
            settings.qdrant_collection,
        ),
    )
    hybrid = HybridRetriever(
        dense,
        BM25Retriever(chunks),
        dense_top_k=settings.top_k_dense,
        sparse_top_k=settings.top_k_sparse,
        rrf_k=settings.rrf_k,
    )
    reranked = RerankingRetriever(
        hybrid,
        create_reranker_service(settings),
        candidate_top_k=settings.top_k_hybrid,
    )
    return RetrieverSuite(
        dense=dense,
        hybrid=hybrid,
        hybrid_reranked=reranked,
    )
