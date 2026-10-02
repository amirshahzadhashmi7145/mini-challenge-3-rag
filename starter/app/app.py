#!/usr/bin/env python3
"""Reference submission skeleton for AMD Mini-Challenge 3 (RAG).

Unlike MC-2, the harness invokes this script in TWO different ways, inside your
already-running container.

1. ONCE, before any graded clock starts, to build whatever index you want:

       python3 /app/app.py --index /app/corpus

   This pass is ungraded and charged to your startup budget, not to a query.
   It is where you should pay for parsing PDFs, spreadsheets and images.

2. ONCE PER QUESTION, ten times:

       python3 /app/app.py --corpus /app/corpus \\
                           --query-id query_01 \\
                           --query "What is the maximum junction temperature?"

   and then it reads the file you wrote:

       /app/output/query_01_output.json

   The query id is GIVEN to you, not derived -- a question has no filename to
   take a stem from. Write to exactly that id plus _output.json.

The output is a JSON object with three keys:

    {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}

  * answer     -- the VALUE only. "94", not "The maximum junction temperature
                  is 94 C". Qualifiers that are part of the value stay:
                  "Q3 FY27", not "Q3". Units and degree signs are normalised
                  away for you, so 94, "94 C" and "94°C" are all accepted.
  * citations  -- corpus-relative paths, compared as an EXACT SET. The test is
                  necessity, not relevance: cite a file only if REMOVING it
                  would make your answer impossible. Dumping your whole
                  retrieval fails almost every question even when the answer is
                  right. Some questions need two files; cite both.
  * confidence -- recorded, never scored.

Missing "answer" or missing "citations" is malformed and scores zero, on
purpose: a crashed writer and a considered refusal must not look the same.

If the corpus does not contain the answer, return an empty answer AND an empty
citation list. "I don't know" is prose, and prose is graded as a wrong answer.
Guessing is wrong, and so is answering from what the model already knows --
the products in this corpus are fictional, so anything recalled rather than
retrieved is wrong by construction.

Three things that will cost you points no matter how good your model is:

  * Loading the model inside answer(). The harness starts a NEW PROCESS for
    every question. A naive implementation loads ten times and blows the
    budget. Persist your index in --index, and keep the model resident behind
    a unix socket, or make loading cheap enough not to matter.
  * Letting one bad file kill the walk. The corpus deliberately contains an
    unreadable file, an encrypted PDF, an unknown binary type and an empty
    directory. A walk that raises on the first unreadable file indexes nothing
    after it -- and because walk order follows the directory listing, WHICH
    files you lose depends on filename order. That is how a submission tests
    clean locally and grades badly here.
  * Running on the CPU. VRAM is sampled continuously and a run that never uses
    the GPU is rejected, so place your model on the device explicitly rather
    than letting it fall back.

Replace index() and answer() below. Leave the plumbing alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


def index(corpus: Path) -> None:
    """Build and PERSIST whatever you need to answer questions later.

    Called once, before any question, with the corpus root. Anything you write
    under INDEX_DIR survives into the per-question invocations; anything you
    keep only in memory does not, because each question is a new process.

    The corpus is a directory tree of mixed types -- pdf, docx, xlsx, csv, txt,
    log, py, png, jpg -- and EVERY type holds at least one graded answer, so
    skipping a format silently costs you the questions that depend on it. Two
    of the ten questions are answerable only by looking at an image.

    Walk defensively: catch per-file, keep going, and record what you could not
    read rather than aborting. See the third bullet in the module docstring.
    """
    from extract import walk_corpus

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    report = walk_corpus(corpus)
    payload = {
        "documents": [
            {"path": doc.path, "kind": doc.kind, "text": doc.text}
            for doc in report.documents
        ],
        "skipped": [
            {"path": item.path, "reason": item.reason} for item in report.skipped
        ],
    }
    (INDEX_DIR / "extracted.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    from search import index_documents

    count = index_documents(report.documents, INDEX_DIR)
    _print_walk(report)
    print(f"chunks {count}")


def _print_walk(report) -> None:
    """Show what the walk kept and what it left behind. Not read by the grader."""
    print(f"parsed {len(report.documents)}")
    for doc in report.documents:
        print(f"  kept   {doc.path}  ({doc.kind}, {len(doc.text)} chars)")
    print(f"skipped {len(report.skipped)}")
    for item in report.skipped:
        print(f"  skip   {item.path}  ({item.reason})")


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Return (answer, citations, confidence) for ONE question.

    THIS is what you replace. `citations` are paths relative to `corpus`, using
    forward slashes -- "specs/tq40_datasheet_r2.pdf", not an absolute path and
    not a bare filename.

    Return ("", [], 0.0) to refuse. Refusing correctly scores full marks on the
    question that has no answer in the corpus, and refusing on a question that
    does have one scores the same as a wrong guess -- so refusal is safe to use
    honestly and useless to use defensively.

    The value has to be copied from a retrieved passage. A passage that does
    not contain it is not cited.
    """
    if not (INDEX_DIR / "chunks.json").is_file():
        print("no vector index — run --index first")
        return "", [], 0.0
    remote = _ask_server(query)
    if remote is not None:
        text, citations, confidence = remote
        print(f"via server answer={text!r} citations={citations}")
        return text, citations, confidence
    print("model server is not running; loading the model in this process")
    from search import search

    hits = search(INDEX_DIR, query)
    _print_hits(hits)
    from answer import decide

    return decide(query, hits, INDEX_DIR)


def _ask_server(query: str) -> tuple[str, list[str], float] | None:
    import socket
    import struct

    path = os.environ.get("MC3_SOCKET", "/tmp/mc3-answer.sock")
    if not os.path.exists(path):
        return None
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(120)
    try:
        connection.connect(path)
    except OSError:
        return None
    payload = json.dumps({"query": query, "index": str(INDEX_DIR)}).encode()
    connection.sendall(struct.pack(">I", len(payload)) + payload)
    header = _exact(connection, 4)
    size = struct.unpack(">I", header)[0]
    body = json.loads(_exact(connection, size))
    connection.close()
    if body.get("error"):
        raise RuntimeError(body["error"])
    return body["answer"], list(body["citations"]), float(body["confidence"])


def _exact(connection, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        piece = connection.recv(remaining)
        if not piece:
            raise ConnectionError("socket closed")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _print_hits(hits: list[dict]) -> None:
    print(f"hits {len(hits)}")
    for hit in hits:
        preview = " ".join(hit["text"].split())
        if len(preview) > 140:
            preview = preview[:140] + "..."
        print(f"  {hit['score']:.2f}  {hit['path']}")
        print(f"       {preview}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    # Written whole rather than streamed: a partially-written file that the
    # harness reads mid-flush parses as invalid JSON and scores the query zero.
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
