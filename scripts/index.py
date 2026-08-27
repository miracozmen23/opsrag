"""Embed processed chunks and replace their Qdrant points."""

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
from app.ingestion.vector_indexer import VectorIndexer  # noqa: E402
from app.retrieval.vector_search import QdrantVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Explicitly delete and recreate an incompatible/existing collection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    chunks = read_chunks_jsonl(args.input)
    embedding_service = create_embedding_service(settings)
    vector_store = QdrantVectorStore(
        create_qdrant_client(settings),
        settings.qdrant_collection,
    )
    result = VectorIndexer(
        embedding_service,
        vector_store,
        qdrant_batch_size=settings.qdrant_batch_size,
    ).index(chunks, recreate=args.recreate)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

