"""Markdown, text, and PDF document loading."""

import hashlib
import logging
import re
from pathlib import Path

from charset_normalizer import from_bytes
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from app.ingestion.cleaner import clean_text
from app.ingestion.models import Document, DocumentType, ParsedSection

logger = logging.getLogger(__name__)
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SUPPORTED_TYPES: dict[str, DocumentType] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
}


class IngestionFailure(BaseModel):
    """One source that could not be parsed."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=1)
    error: str = Field(min_length=1)


class DocumentLoadResult(BaseModel):
    """Successful documents and isolated failures from one directory scan."""

    model_config = ConfigDict(frozen=True)

    documents: list[Document]
    failures: list[IngestionFailure]

    @property
    def section_count(self) -> int:
        return sum(len(document.sections) for document in self.documents)


def load_documents(input_dir: Path) -> DocumentLoadResult:
    """Load supported files recursively in deterministic path order."""

    root = input_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    documents: list[Document] = []
    failures: list[IngestionFailure] = []
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_TYPES
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    for path in paths:
        source = path.relative_to(root).as_posix()
        try:
            documents.append(load_document(path, source=source))
        except Exception as exc:
            logger.warning("document_load_failed source=%s error=%s", source, exc)
            failures.append(IngestionFailure(source=source, error=str(exc)))

    return DocumentLoadResult(documents=documents, failures=failures)


def load_document(path: Path, *, source: str | None = None) -> Document:
    """Load and normalize one supported source file."""

    suffix = path.suffix.lower()
    document_type = _SUPPORTED_TYPES.get(suffix)
    if document_type is None:
        raise ValueError(f"Unsupported document type: {suffix or '<none>'}")
    source_name = source or path.name

    if document_type == "pdf":
        title, sections = _load_pdf(path)
    else:
        text = clean_text(_decode_text(path.read_bytes()))
        if not text:
            raise ValueError("Document contains no readable text.")
        if document_type == "markdown":
            title, sections = _parse_markdown(text, path.stem)
        else:
            title = _humanize_stem(path.stem)
            sections = [ParsedSection(section_index=0, title=title, text=text)]

    return Document(
        document_id=_document_id(source_name),
        source=source_name,
        title=title,
        document_type=document_type,
        sections=sections,
    )


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        match = from_bytes(data).best()
        if match is None:
            raise ValueError("Unable to determine document encoding.")
        return str(match)


def _parse_markdown(text: str, fallback_title: str) -> tuple[str, list[ParsedSection]]:
    title = _humanize_stem(fallback_title)
    current_title = title
    current_lines: list[str] = []
    sections: list[ParsedSection] = []
    in_fence = False

    def flush() -> None:
        section_text = clean_text("\n".join(current_lines))
        if section_text:
            sections.append(
                ParsedSection(
                    section_index=len(sections),
                    title=current_title,
                    text=section_text,
                )
            )

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _MARKDOWN_HEADING.match(line)
        if match:
            flush()
            current_lines = []
            heading = clean_text(match.group(2))
            if len(match.group(1)) == 1:
                title = heading
            current_title = heading
            continue
        current_lines.append(line)
    flush()

    if not sections:
        raise ValueError("Markdown document contains no section text.")
    return title, sections


def _load_pdf(path: Path) -> tuple[str, list[ParsedSection]]:
    reader = PdfReader(str(path))
    metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    title = clean_text(str(metadata_title)) if metadata_title else _humanize_stem(path.stem)
    sections: list[ParsedSection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append(
                ParsedSection(
                    section_index=len(sections),
                    title=f"Page {page_number}",
                    text=text,
                    page_number=page_number,
                )
            )
    if not sections:
        raise ValueError("PDF contains no extractable text.")
    return title, sections


def _document_id(source: str) -> str:
    digest = hashlib.sha256(source.replace("\\", "/").casefold().encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"


def _humanize_stem(stem: str) -> str:
    return re.sub(r"[-_]+", " ", stem).strip().title() or "Untitled"

