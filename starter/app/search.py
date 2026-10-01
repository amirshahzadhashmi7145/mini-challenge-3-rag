"""Split saved documents into passages and rank them for one question.

BM25 is keyword search. A passage scores higher when the question's words
are in it, and higher still when those words are rare in the rest of the corpus.
Part numbers and error codes are exact words, which is what this is good at.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = {
    "a", "an", "and", "does", "for", "in", "is", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who",
}


def chunk_document(path: str, kind: str, text: str) -> list[dict]:
    """One passage per row, log line, or paragraph. The path stays on each one."""
    if kind in {"csv", "xlsx"}:
        pieces = _row_chunks(text)
    elif path.endswith(".log"):
        pieces = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        pieces = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return [{"path": path, "text": piece} for piece in pieces if piece]


def search_chunks(chunks: list[dict], query: str, k: int = 5) -> list[dict]:
    """Return the top passages for ``query``. Empty when nothing matches."""
    if not chunks:
        return []
    from rank_bm25 import BM25Okapi

    docs = [tokenize(chunk["text"]) for chunk in chunks]
    scores = BM25Okapi(docs).get_scores(tokenize(query))
    ranked = sorted(
        (
            {"path": chunk["path"], "text": chunk["text"], "score": float(score)}
            for chunk, score in zip(chunks, scores)
            if score > 0
        ),
        key=lambda hit: hit["score"],
        reverse=True,
    )
    return ranked[:k]


def tokenize(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower().replace("_", " "))
    kept = [word for word in words if word not in _STOP]
    return kept or words


def _row_chunks(text: str) -> list[str]:
    """Keep the header on every data row so the column name is searchable."""
    pieces = []
    header = None
    sheet = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--- ") and stripped.endswith(" ---"):
            sheet = stripped
            header = None
            continue
        if "\t" not in stripped:
            pieces.append(stripped)
            continue
        if header is None:
            header = stripped
            continue
        prefix = f"{sheet}\n" if sheet else ""
        pieces.append(f"{prefix}{header}\n{stripped}")
    return pieces
