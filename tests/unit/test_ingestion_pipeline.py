"""Ingestion artifact behavior."""

from pathlib import Path

from app.ingestion.chunker import RecursiveTextChunker
from app.ingestion.pipeline import ingest_directory, read_chunks_jsonl


def test_ingestion_writes_valid_deterministic_jsonl(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "guide.md").write_text("# Guide\n\nCheck logs and ports.", encoding="utf-8")
    output = tmp_path / "processed" / "chunks.jsonl"
    chunker = RecursiveTextChunker(chunk_size_tokens=10, overlap_tokens=2)

    first_report, _ = ingest_directory(raw, output, chunker)
    first_content = output.read_text(encoding="utf-8")
    first_bytes = output.read_bytes()
    second_report, _ = ingest_directory(raw, output, chunker)

    assert output.read_text(encoding="utf-8") == first_content
    assert output.read_bytes() == first_bytes
    assert first_bytes.endswith(b"\n")
    assert b"\r\n" not in first_bytes
    assert read_chunks_jsonl(output)[0].source == "guide.md"
    assert first_report.chunks == second_report.chunks == 1
