"""Fresh-interpreter checks for package import boundaries."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_service_factory_imports_without_retrieval_cycle() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.core.service_factory import create_embedding_service",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
