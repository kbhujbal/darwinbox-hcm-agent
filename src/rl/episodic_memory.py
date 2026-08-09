"""Episodic memory: past incident resolutions stored in a second Chroma
collection, reusing Part 1's generic VectorStore wrapper unchanged (see
src/rag/vector_store.py) — this is the "Chroma vector store is reused, not
replaced" design decision.

A new anomaly retrieves semantically similar past incidents; a close match
both nudges its reported confidence up and feeds an "average past reward
per action" signal into the bandit's context vector (src/rl/features.py),
which is what makes a repeated anomaly type get handled faster and more
confidently the second time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src import config
from src.anomaly.models import Anomaly
from src.llm.gemini_client import get_client
from src.rag.vector_store import VectorStore

INCIDENT_COLLECTION_NAME = "incident_memory"
SIMILARITY_DISTANCE_FLOOR = 0.5  # cosine distance below which two incidents count as "similar"

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(collection_name=INCIDENT_COLLECTION_NAME)
    return _store


def reset_memory() -> None:
    """Clears all recorded incidents — used by demo/eval scripts that want
    a clean-slate run not influenced by a previous invocation's history."""
    _get_store().reset()


@dataclass
class SimilarIncident:
    incident_id: str
    description: str
    anomaly_type: str
    action_taken: str
    reward: float
    distance: float


def record_incident(anomaly: Anomaly, action_taken: str, reward: float) -> str:
    """Embeds and stores a resolved incident. Call once the reward is known
    (after a HITL decision or an auto-executed action's outcome)."""
    client = get_client()
    text = anomaly.description()
    embed_result = client.embed([text])

    incident_id = f"{anomaly.employee_id}-{anomaly.anomaly_type}-{int(time.time() * 1000)}"
    store = _get_store()
    store.upsert(
        ids=[incident_id],
        embeddings=embed_result.vectors,
        documents=[text],
        metadatas=[
            {
                "employee_id": anomaly.employee_id,
                "anomaly_type": anomaly.anomaly_type,
                "action_taken": action_taken,
                "reward": reward,
            }
        ],
    )
    return incident_id


def retrieve_similar(anomaly: Anomaly, top_k: int = 3) -> list[SimilarIncident]:
    client = get_client()
    embed_result = client.embed([anomaly.description()])
    store = _get_store()
    chunks = store.query(embed_result.vectors[0], top_k=top_k)

    out = []
    for c in chunks:
        if c.distance > SIMILARITY_DISTANCE_FLOOR:
            continue
        out.append(
            SimilarIncident(
                incident_id=c.chunk_id,
                description=c.text,
                anomaly_type=c.metadata.get("anomaly_type", ""),
                action_taken=c.metadata.get("action_taken", ""),
                reward=float(c.metadata.get("reward", 0.0)),
                distance=c.distance,
            )
        )
    return out


def similarity_bias(anomaly: Anomaly, top_k: int = 3) -> tuple[float, bool]:
    """Returns (past_similar_avg_reward, has_prior_incident) ready to feed
    straight into src.rl.features.encode_context()."""
    similar = retrieve_similar(anomaly, top_k=top_k)
    if not similar:
        return 0.0, False
    avg_reward = sum(s.reward for s in similar) / len(similar)
    return avg_reward, True


def confidence_boost(anomaly: Anomaly, similar: list[SimilarIncident] | None = None) -> float:
    """How much to bump the anomaly's raw statistical confidence given
    precedent — a close, positively-resolved past match makes the system
    more confident this time, which is the 'faster & higher confidence on
    the 2nd occurrence' behavior the assignment asks to demonstrate."""
    similar = similar if similar is not None else retrieve_similar(anomaly)
    same_type = [s for s in similar if s.anomaly_type == anomaly.anomaly_type]
    if not same_type:
        return 0.0
    best = min(same_type, key=lambda s: s.distance)
    closeness = max(0.0, 1.0 - best.distance / SIMILARITY_DISTANCE_FLOOR)
    return round(0.15 * closeness * max(0.0, best.reward), 3)
