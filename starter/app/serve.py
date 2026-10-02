#!/usr/bin/env python3
"""Hold the language model and answer questions over a unix socket.

Each graded question is a new process. Loading the model inside that process
uses up the 30 second budget. This process loads it once, at container start.
"""

from __future__ import annotations

import json
import os
import socket
import struct
from pathlib import Path

SOCKET_PATH = os.environ.get("MC3_SOCKET", "/tmp/mc3-answer.sock")


def main() -> None:
    from answer import _load

    print("loading model", flush=True)
    _load()
    path = Path(SOCKET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    print(f"ready {path}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                _handle(connection)
    finally:
        server.close()
        if path.exists():
            path.unlink()


def _handle(connection: socket.socket) -> None:
    from answer import decide
    from search import search

    try:
        request = json.loads(_read(connection))
        index = Path(request["index"])
        query = request["query"]
        hits = search(index, query)
        text, citations, confidence = decide(query, hits, index)
        payload = {"answer": text, "citations": citations, "confidence": confidence}
    except Exception as error:
        payload = {"answer": "", "citations": [], "confidence": 0.0, "error": str(error)}
    _write(connection, json.dumps(payload).encode())


def _read(connection: socket.socket) -> bytes:
    header = _exact(connection, 4)
    size = struct.unpack(">I", header)[0]
    return _exact(connection, size)


def _write(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(struct.pack(">I", len(payload)) + payload)


def _exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        piece = connection.recv(remaining)
        if not piece:
            raise ConnectionError("socket closed")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


if __name__ == "__main__":
    main()
