"""Section/heading-aware chunking for the HR policy markdown document.

Each `## ` heading in the source document is treated as one policy clause.
Most clauses in data/hr_policy.md are short enough to stay as a single
chunk, which keeps retrieval precise (a hit maps to exactly one policy
section). Any section that exceeds the configured token budget is split
into overlapping word-windows so no single chunk blows the context budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src import config


@dataclass
class Chunk:
    chunk_id: str
    section_title: str
    text: str
    token_estimate: int


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split on level-2 headings ('## Title'). Returns (title, body) pairs."""
    pattern = re.compile(r"^##\s+(.*)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown_text))
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        sections.append((title, body))
    return sections


def _split_long_section(
    title: str, body: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    words = body.split()
    if len(words) <= max_tokens:
        return [body]

    step = max(1, max_tokens - overlap_tokens)
    windows = []
    start = 0
    while start < len(words):
        window = words[start : start + max_tokens]
        windows.append(" ".join(window))
        if start + max_tokens >= len(words):
            break
        start += step
    return windows


def chunk_document(
    markdown_text: str,
    max_tokens: int = config.RAG_CHUNK_MAX_TOKENS,
    overlap_tokens: int = config.RAG_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for title, body in split_into_sections(markdown_text):
        if not body:
            continue
        slug = _slugify(title)
        parts = _split_long_section(title, body, max_tokens, overlap_tokens)
        for idx, part in enumerate(parts):
            text = f"{title}\n{part}"
            chunks.append(
                Chunk(
                    chunk_id=f"{slug}-{idx}",
                    section_title=title,
                    text=text,
                    token_estimate=_estimate_tokens(text),
                )
            )
    return chunks
