"""Validate and summarize the version-controlled evaluation benchmark."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.dataset import validate_evaluation_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.jsonl",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_evaluation_dataset(args.input, args.sources)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
