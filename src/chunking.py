"""Text chunking utilities.

Sentence splitting is Arabic-aware (handles both "?" and the Arabic
question mark "؟") and chunking is recursive: paragraphs first, then
sentences, so chunks never cut a sentence in half unless a single
sentence is longer than the chunk size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+|(?<=\n)\s*")
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """A retrievable piece of text with provenance."""

    text: str
    doc_id: str
    index: int
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping Arabic punctuation intact."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 80,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Recursively chunk *text* into ~chunk_size characters with overlap.

    Packing order: paragraphs -> sentences -> hard character split, so
    semantic boundaries are preserved whenever possible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(split_sentences(para))

    chunks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > chunk_size:  # a single unit longer than the window
            head, unit = unit[:chunk_size], unit[chunk_size - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # seed the next chunk with an overlap tail of the previous one
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]  # start on a word boundary
            current = f"{tail}\n{unit}".strip() if tail else unit
    if current:
        chunks.append(current)

    base_meta = dict(metadata or {})
    return [
        Chunk(text=c, doc_id=doc_id, index=i, metadata={**base_meta, "doc_id": doc_id})
        for i, c in enumerate(chunks)
    ]
