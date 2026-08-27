"""Load raw documents and write deterministic retrieval chunks."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.chunker import RecursiveTextChunker  # noqa: E402
from app.ingestion.pipeline import ingest_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    chunker = RecursiveTextChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    report, load_result = ingest_directory(args.input, args.output, chunker)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    if load_result.failures:
        print(
            json.dumps(
                {"failure_details": [item.model_dump() for item in load_result.failures]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

