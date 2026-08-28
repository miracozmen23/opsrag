"""Serialization helpers for the reproducible ingestion pipeline."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.chunker import RecursiveTextChunker, chunk_documents
from app.ingestion.loader import DocumentLoadResult, load_documents
from app.ingestion.models import Chunk


class IngestionReport(BaseModel):
    """Observable summary of a completed ingestion run."""

    model_config = ConfigDict(frozen=True)

    documents: int = Field(ge=0)
    sections: int = Field(ge=0)
    chunks: int = Field(ge=0)
    failures: int = Field(ge=0)
    average_tokens_per_chunk: float = Field(ge=0)
    min_tokens: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    output_path: str = Field(min_length=1)


def ingest_directory(
    input_dir: Path,
    output_path: Path,
    chunker: RecursiveTextChunker,
) -> tuple[IngestionReport, DocumentLoadResult]:
    """Load documents, create chunks, and write deterministic JSONL."""

    load_result = load_documents(input_dir)
    chunks = chunk_documents(load_result.documents, chunker)
    write_chunks_jsonl(chunks, output_path)
    token_counts = [chunk.token_count for chunk in chunks]
    report = IngestionReport(
        documents=len(load_result.documents),
        sections=load_result.section_count,
        chunks=len(chunks),
        failures=len(load_result.failures),
        average_tokens_per_chunk=(sum(token_counts) / len(token_counts) if token_counts else 0),
        min_tokens=min(token_counts, default=0),
        max_tokens=max(token_counts, default=0),
        output_path=str(output_path),
    )
    return report, load_result


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Write stable UTF-8 JSONL and atomically replace the previous output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    lines = [
        json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for chunk in chunks
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    temporary_path.write_bytes(payload.encode("utf-8"))
    temporary_path.replace(output_path)


def read_chunks_jsonl(input_path: Path) -> list[Chunk]:
    """Read and validate chunks from the ingestion artifact."""

    if not input_path.is_file():
        raise ValueError(f"Chunk file does not exist: {input_path}")
    chunks: list[Chunk] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            chunks.append(Chunk.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid chunk JSONL at line {line_number}: {exc}") from exc
    return chunks
