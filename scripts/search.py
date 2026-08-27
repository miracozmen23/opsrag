"""Run a dense search against the configured Qdrant collection."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.service_factory import create_embedding_service, create_qdrant_client  # noqa: E402
from app.retrieval.vector_search import DenseRetriever, QdrantVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Technical question to retrieve context for.")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    retriever = DenseRetriever(
        create_embedding_service(settings),
        QdrantVectorStore(
            create_qdrant_client(settings),
            settings.qdrant_collection,
        ),
    )
    results = retriever.search(args.query, args.top_k or settings.top_k_dense)
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

