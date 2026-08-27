"""Text cleaner behavior."""

from app.ingestion.cleaner import clean_text


def test_clean_text_normalizes_line_endings_and_blank_lines() -> None:
    assert clean_text("one\r\n\r\n\r\n\r\ntwo\r") == "one\n\ntwo"


def test_clean_text_removes_trailing_whitespace_and_nul() -> None:
    assert clean_text(" value  \x00\nnext\t") == "value\nnext"


def test_clean_text_preserves_indentation() -> None:
    assert clean_text("command:\n    uvicorn app.main:app") == (
        "command:\n    uvicorn app.main:app"
    )

