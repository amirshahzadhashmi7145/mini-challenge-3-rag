"""Recursive character splitter.

Breaks on a paragraph, then a line, then a sentence, then a word.
A piece bigger than ``chunk_size`` is broken again with the next separator.
``overlap`` keeps the tail of the previous piece so a fact on the cut survives.
"""

from __future__ import annotations

SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def recursive_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    return _split(text.strip(), chunk_size, overlap, SEPARATORS)


def _split(text: str, chunk_size: int, overlap: int, separators: tuple[str, ...]) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    separator, rest = _pick(text, separators)
    if separator == "":
        return _windows(text, chunk_size, overlap)
    parts = [part.strip() for part in text.split(separator) if part.strip()]
    good: list[str] = []
    chunks: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            good.append(part)
            continue
        if good:
            chunks.extend(_merge(good, separator, chunk_size, overlap))
            good = []
        chunks.extend(_split(part, chunk_size, overlap, rest))
    if good:
        chunks.extend(_merge(good, separator, chunk_size, overlap))
    return chunks


def _pick(text: str, separators: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    for index, separator in enumerate(separators):
        if separator == "":
            return "", ()
        if separator in text:
            return separator, separators[index + 1 :]
    return "", ()


def _merge(parts: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def joined(pieces: list[str]) -> str:
        return separator.join(pieces).strip()

    def size(pieces: list[str]) -> int:
        if not pieces:
            return 0
        return len(separator.join(pieces))

    for part in parts:
        if current and size(current) + len(separator) + len(part) > chunk_size:
            chunks.append(joined(current))
            while current and (size(current) > overlap or size(current) + len(separator) + len(part) > chunk_size):
                current.pop(0)
        current.append(part)
    if current:
        text = joined(current)
        if text:
            chunks.append(text)
    return chunks


def _windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks
