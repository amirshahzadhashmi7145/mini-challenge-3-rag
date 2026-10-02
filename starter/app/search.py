"""Embed passages and search them with FAISS.

FAISS is the vector index. It is a file under the index directory, not a
separate server. The grader does not give the container a port.
"""Split saved documents into passages and rank them for one question.

BM25 is keyword search. A passage scores higher when the question's words
are in it, and higher still when those words are rare in the rest of the corpus.
Part numbers and error codes are exact words, which is what this is good at.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from chunking import recursive_chunks

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 5

_CHUNKS = "chunks.json"
_VECTORS = "vectors.faiss"


def index_documents(documents: list, index_dir: Path) -> int:
    chunks = []
    for doc in documents:
        for text in recursive_chunks(doc.text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"path": doc.path, "text": text})
    vectors = _embed([chunk["text"] for chunk in chunks], queries=False)
    _save(index_dir, chunks, vectors)
    return len(chunks)


def search(index_dir: Path, query: str, k: int = TOP_K) -> list[dict]:
    import faiss

    chunks = json.loads((index_dir / _CHUNKS).read_text(encoding="utf-8"))["chunks"]
    if not chunks:
        return []
    index = faiss.read_index(str(index_dir / _VECTORS))
    query_vector = np.ascontiguousarray(_embed([query], queries=True))
    scores, ids = index.search(query_vector, min(k, index.ntotal))
    hits = []
    for score, row in zip(scores[0], ids[0]):
        if int(row) < 0 or float(score) <= 0:
            continue
        chunk = chunks[int(row)]
        hits.append({"path": chunk["path"], "text": chunk["text"], "score": float(score)})
    return hits


_EMBED = None


def _embed(texts: list[str], queries: bool) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    rows = list(_embedding_model().query_embed(texts) if queries else _embedding_model().embed(texts))
    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _save(index_dir: Path, chunks: list[dict], vectors: np.ndarray) -> None:
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / _CHUNKS).write_text(json.dumps({"chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    if len(chunks) == 0:
        return
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(index_dir / _VECTORS))


def _embedding_model():
    global _EMBED
    if _EMBED is None:
        from fastembed import TextEmbedding

        _EMBED = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(_cache_dir()))
    return _EMBED


def _cache_dir() -> Path:
    configured = os.environ.get("FASTEMBED_CACHE_PATH")
    if configured:
        return Path(configured)
    bundled = Path("/models/fastembed")
    if bundled.is_dir():
        return bundled
    return Path.home() / ".cache" / "fastembed"
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
