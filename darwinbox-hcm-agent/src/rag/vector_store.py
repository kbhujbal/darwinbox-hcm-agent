"""Generic persistent-Chroma wrapper, keyed by collection name.

Part 1 only uses one collection (the HR policy doc). Kept generic so a
future Part 2 episodic-memory collection (past incident resolutions) can
reuse this same class rather than a new vector-store integration.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src import config


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    section_title: str
    distance: float


class VectorStore:
    def __init__(
        self,
        collection_name: str = config.RAG_COLLECTION_NAME,
        persist_dir: str | None = None,
    ):
        persist_dir = persist_dir or str(config.CHROMA_DIR)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(
        self, query_embedding: list[float], top_k: int = config.RAG_TOP_K
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        out = []
        for i, chunk_id in enumerate(result["ids"][0]):
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=result["documents"][0][i],
                    section_title=result["metadatas"][0][i].get("section_title", ""),
                    distance=result["distances"][0][i],
                )
            )
        return out

    def count(self) -> int:
        return self._collection.count()
