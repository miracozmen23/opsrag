"""Conservative text normalization for technical documents."""

import re
import unicodedata

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(text: str) -> str:
    """Normalize encoding artifacts while preserving technical layout."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    normalized = _TRAILING_WHITESPACE.sub("", normalized)
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()

