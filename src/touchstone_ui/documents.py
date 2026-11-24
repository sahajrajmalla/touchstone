"""Turn what the user typed or uploaded into documents for the indexer."""

import re
from collections.abc import Iterable

_BLANK_LINES = re.compile(r"\n\s*\n")


def split_text(text: str, *, split_paragraphs: bool) -> list[str]:
    """Return the non-empty documents in ``text``.

    With ``split_paragraphs`` each blank-line separated block becomes a document;
    otherwise the whole text is one document.
    """
    text = text.strip()
    if not text:
        return []
    if not split_paragraphs:
        return [text]
    return [block.strip() for block in _BLANK_LINES.split(text) if block.strip()]


def parse_urls(text: str) -> tuple[list[str], list[str]]:
    """Split one-URL-per-line input into ``(valid_urls, rejected_lines)``."""
    valid: list[str] = []
    rejected: list[str] = []
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        is_url = line.startswith(("http://", "https://")) and " " not in line
        (valid if is_url else rejected).append(line)
    return valid, rejected


def decode_uploads(files: Iterable[tuple[str, bytes]]) -> tuple[list[str], list[str]]:
    """Decode uploaded text files into ``(documents, unreadable_file_names)``."""
    documents: list[str] = []
    unreadable: list[str] = []
    for name, data in files:
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            unreadable.append(name)
            continue
        if text:
            documents.append(text)
    return documents, unreadable
