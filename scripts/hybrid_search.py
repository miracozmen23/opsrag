"""Run dense plus BM25 retrieval with Reciprocal Rank Fusion."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.service_factory import create_embedding_service, create_qdrant_client  # noqa: E402
from app.ingestion.pipeline import read_chunks_jsonl  # noqa: E402
from app.retrieval.bm25_search import BM25Retriever  # noqa: E402
from app.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from app.retrieval.vector_search import DenseRetriever, QdrantVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Technical query to retrieve hybrid matches for.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    chunks = read_chunks_jsonl(args.input)
    dense_retriever = DenseRetriever(
        create_embedding_service(settings),
        QdrantVectorStore(
            create_qdrant_client(settings),
            settings.qdrant_collection,
        ),
    )
    retriever = HybridRetriever(
        dense_retriever,
        BM25Retriever(chunks),
        dense_top_k=settings.top_k_dense,
        sparse_top_k=settings.top_k_sparse,
        rrf_k=settings.rrf_k,
    )
    results = retriever.search(args.query, args.top_k or settings.top_k_hybrid)
    print(
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
