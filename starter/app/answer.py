"""Turn the retrieved passages into a value and the files it came from.

The model may only copy a value that is written in a passage. If it names a
passage that does not contain that value, the citation is dropped. If nothing
remains, the answer is empty.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

LLM_NAME = os.environ.get("MC3_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
_LINK = re.compile(r"\b(?:ORR-\d+|E\d{3,})\b", re.I)
_MODEL = None
_TOKENIZER = None
_FORCE_CPU = False


def decide(query: str, hits: list[dict], index_dir: Path) -> tuple[str, list[str], float]:
    usable = [hit for hit in hits if not _obsolete(hit)] or list(hits)
    text, citations, score = _from_hits(query, usable)
    if text:
        return text, citations, score
    extra = _linked_hits(index_dir, usable)
    if extra:
        text, citations, score = _from_hits(query, _dedupe(usable + extra))
        if text:
            return text, citations, score
    return "", [], 0.0


def _from_hits(query: str, hits: list[dict]) -> tuple[str, list[str], float]:
    if not hits:
        return "", [], 0.0
    raw = _generate(_prompt(query, hits))
    answer, numbers = _parse(raw)
    if not answer:
        answer, numbers = _salvage(raw, hits)
    if not answer:
        return "", [], 0.0
    chosen = []
    for number in numbers:
        if 1 <= number <= len(hits):
            chosen.append(hits[number - 1])
    if not chosen:
        chosen = [hit for hit in hits if _contains(hit["text"], answer)]
    carriers = [hit for hit in chosen if _contains(hit["text"], answer)]
    if not carriers:
        return "", [], 0.0
    linked = _keep_links(query, carriers, hits)
    if not _supports(query, linked):
        return "", [], 0.0
    paths = []
    for hit in linked:
        if hit["path"] not in paths:
            paths.append(hit["path"])
    score = max(hit["score"] for hit in linked)
    return answer, paths, score


def _prompt(query: str, hits: list[dict]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {hit['path']}\n{hit['text']}")
    passages = "\n\n".join(blocks)
    return (
        "Answer using only the passages. Do not use outside knowledge.\n"
        "Reply in exactly this shape:\n"
        "ANSWER: <the value only>\n"
        "SOURCES: <passage numbers, comma-separated>\n"
        "If the passages do not state the answer, leave ANSWER empty and SOURCES empty.\n"
        "The value only: no sentence. Copy it from a passage.\n"
        "A passage that says it is withdrawn is not the source when another passage gives the value.\n"
        "If one passage gives an id and another gives the value for that id, list both numbers.\n"
        "Do not list a passage you did not use.\n"
        "A nearby number is not the answer. If the passages do not state the asked fact, leave ANSWER empty.\n\n"
        f"{passages}\n\nQuestion: {query}"
    )


def _generate(prompt: str) -> str:
    try:
        return _generate_once(prompt)
    except Exception:
        # This laptop's CUDA build needs Python.h to compile a kernel.
        # Drop to CPU rather than failing the question. The eval GPU does not hit this.
        _reset(cpu=True)
        return _generate_once(prompt)


def _generate_once(prompt: str) -> str:
    import torch

    tokenizer, model = _load()
    messages = [
        {"role": "system", "content": "You extract a value from the passages you are given."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _reset(cpu: bool) -> None:
    global _MODEL, _TOKENIZER, _FORCE_CPU
    _MODEL = None
    _TOKENIZER = None
    _FORCE_CPU = cpu


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(LLM_NAME)
    use_cuda = torch.cuda.is_available() and not _FORCE_CPU
    dtype = torch.float16 if use_cuda else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype=dtype)
    if use_cuda:
        _MODEL = _MODEL.to("cuda")
    _MODEL.eval()
    return _TOKENIZER, _MODEL


def _parse(raw: str) -> tuple[str, list[int]]:
    answer = ""
    numbers: list[int] = []
    for line in raw.splitlines():
        upper = line.strip().upper()
        if upper.startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif upper.startswith("SOURCES:"):
            numbers = [int(item) for item in re.findall(r"\d+", line)]
    if answer.lower() in {"none", "n/a", "unknown", "empty"}:
        answer = ""
    return answer, numbers


def _salvage(raw: str, hits: list[dict]) -> tuple[str, list[int]]:
    """The model sometimes pastes the whole line. Pull the single code out of that line."""
    if "ANSWER:" in raw.upper():
        return "", []
    patterns = (
        r"\bE\d{3,}\b",
        r"\bORR-[A-Z0-9]+(?:-[A-Z0-9]+)+\b",
        r"\bQ[1-4]\s*FY\d{2}\b",
        r"\bREV-[A-Z0-9]+\b",
        r"\bB\d+\b",
        r"\b\d+\.\d+\.\d+\b",
    )
    found = []
    for pattern in patterns:
        for match in re.findall(pattern, raw, flags=re.I):
            if match not in found:
                found.append(match)
    if len(found) != 1:
        return "", []
    answer = found[0]
    numbers = [index for index, hit in enumerate(hits, start=1) if answer.lower() in hit["text"].lower()]
    return (answer, numbers) if numbers else ("", [])


def _contains(passage: str, answer: str) -> bool:
    hay = " ".join(passage.lower().split())
    needle = " ".join(answer.lower().split())
    return bool(needle) and needle in hay


def _keep_links(query: str, carriers: list[dict], hits: list[dict]) -> list[dict]:
    """Keep the passage that holds the value, and a second passage that supplied the id."""
    question = query.upper()
    kept = list(carriers)
    for carrier in carriers:
        for identifier in _LINK.findall(carrier["text"]):
            if identifier.upper() in question:
                continue
            for hit in hits:
                if hit in kept:
                    continue
                if identifier.upper() in hit["text"].upper():
                    kept.append(hit)
    return kept


def _supports(query: str, carriers: list[dict]) -> bool:
    """The cited passage has to share a real word with the question, not only a nearby number."""
    words = [word for word in re.findall(r"[a-z0-9]{5,}", query.lower()) if word not in {"which", "where", "there"}]
    if not words:
        return True
    blob = " ".join(hit["text"].lower() for hit in carriers)
    squashed = re.sub(r"[^a-z0-9]", "", blob)
    return any(word in blob or word in squashed for word in words)


def _linked_hits(index_dir: Path, hits: list[dict]) -> list[dict]:
    from search import search

    found = []
    seen = {hit["text"] for hit in hits}
    for hit in hits[:3]:
        for identifier in _LINK.findall(hit["text"]):
            for extra in search(index_dir, identifier, k=3):
                if extra["text"] not in seen:
                    seen.add(extra["text"])
                    found.append(extra)
    return found


def _dedupe(hits: list[dict]) -> list[dict]:
    kept = []
    seen = set()
    for hit in hits:
        if hit["text"] in seen:
            continue
        seen.add(hit["text"])
        kept.append(hit)
    return kept


def _obsolete(hit: dict) -> bool:
    opening = f"{hit['path']}\n{hit['text'][:240]}".lower()
    return "withdrawn" in opening
