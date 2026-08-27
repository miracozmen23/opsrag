"""Document loader behavior."""

from pathlib import Path

import pytest

from app.ingestion.loader import load_document, load_documents


def test_markdown_loader_preserves_title_and_sections(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# Service Guide\n\nIntro text.\n\n## Connection Refused\n\nCheck the port.",
        encoding="utf-8",
    )

    document = load_document(path)

    assert document.title == "Service Guide"
    assert [section.title for section in document.sections] == [
        "Service Guide",
        "Connection Refused",
    ]
    assert document.source == "guide.md"


def test_markdown_ignores_heading_syntax_inside_code_fence(tmp_path: Path) -> None:
    path = tmp_path / "code.md"
    path.write_text("# Title\n\n```text\n## not a heading\n```", encoding="utf-8")

    document = load_document(path)

    assert len(document.sections) == 1
    assert "## not a heading" in document.sections[0].text


def test_text_loader_detects_non_utf8_encoding(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes("café service notes".encode("cp1252"))

    document = load_document(path)

    assert "café" in document.sections[0].text
    assert document.document_type == "text"


def test_document_id_is_stable_for_same_source(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("first", encoding="utf-8")
    first = load_document(path)
    path.write_text("changed", encoding="utf-8")
    second = load_document(path)
    assert first.document_id == second.document_id


def test_unsupported_document_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(path)


def test_directory_loader_is_deterministic_and_ignores_unsupported(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\nA text", encoding="utf-8")
    (tmp_path / "skip.json").write_text("{}", encoding="utf-8")

    result = load_documents(tmp_path)

    assert [document.source for document in result.documents] == ["a.md", "b.txt"]
    assert result.section_count == 2
    assert result.failures == []
