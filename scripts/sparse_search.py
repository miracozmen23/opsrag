"""Run a BM25 search against the processed local chunk artifact."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.pipeline import read_chunks_jsonl  # noqa: E402
from app.retrieval.bm25_search import BM25Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Technical query to retrieve lexical matches for.")
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
    retriever = BM25Retriever(read_chunks_jsonl(args.input))
    results = retriever.search(args.query, args.top_k or settings.top_k_sparse)
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
