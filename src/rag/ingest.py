"""One-time (re-runnable) script: chunk data/hr_policy.md, embed each chunk
with Gemini, and persist the vectors into the local Chroma collection.

Usage:
    python -m src.rag.ingest
"""
from __future__ import annotations

from src import config
from src.llm.gemini_client import get_client
from src.rag.chunker import chunk_document
from src.rag.vector_store import VectorStore


def ingest(policy_path=None, reset: bool = True) -> int:
    policy_path = policy_path or (config.DATA_DIR / "hr_policy.md")
    text = policy_path.read_text(encoding="utf-8")

    chunks = chunk_document(text)
    if not chunks:
        raise RuntimeError(f"No chunks produced from {policy_path} — check the '## ' headings.")

    client = get_client()
    embedding_result = client.embed([c.text for c in chunks])

    store = VectorStore()
    if reset:
        store.reset()

    store.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embedding_result.vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"section_title": c.section_title} for c in chunks],
    )

    print(
        f"Ingested {len(chunks)} chunks from {policy_path.name} "
        f"({embedding_result.tokens_in} tokens, ${embedding_result.cost_usd:.6f}, "
        f"{embedding_result.latency_ms:.0f}ms)."
    )
    return len(chunks)


if __name__ == "__main__":
    ingest()
