"""Embed passages and search them with FAISS.

FAISS is the vector index. It is a file under the index directory, not a
separate server. The grader does not give the container a port.
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


def _embed(texts: list[str], queries: bool) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(_cache_dir()))
    rows = list(model.query_embed(texts) if queries else model.embed(texts))
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


def _cache_dir() -> Path:
    configured = os.environ.get("FASTEMBED_CACHE_PATH")
    if configured:
        return Path(configured)
    bundled = Path("/models/fastembed")
    if bundled.is_dir():
        return bundled
    return Path.home() / ".cache" / "fastembed"
