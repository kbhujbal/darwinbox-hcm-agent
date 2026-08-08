"""Policy Agent: retrieve → ground → answer. Refuses rather than
hallucinates when retrieval doesn't produce a confident match.
"""
from __future__ import annotations

import time

from src import config
from src.graph.state import HCMState
from src.llm.gemini_client import get_client
from src.observability.tracer import Tracer
from src.rag.vector_store import VectorStore

POLICY_SYSTEM_PROMPT = (
    "You are the HR Policy Agent for Meridian Industries. Answer the employee's question "
    "using ONLY the policy excerpts provided below. Cite the section title(s) you used. "
    "If the excerpts do not contain enough information to answer confidently, say so "
    "plainly and suggest the employee contact HR — do NOT invent policy details."
)

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def answer_policy_question(state: HCMState, tracer: Tracer) -> dict:
    start = time.perf_counter()
    query = state["user_input"]

    client = get_client()
    embed_result = client.embed([query])
    query_vector = embed_result.vectors[0]

    store = _get_store()
    chunks = store.query(query_vector, top_k=config.RAG_TOP_K)
    grounded_chunks = [c for c in chunks if c.distance <= config.RAG_DISTANCE_FLOOR]

    tokens_in = embed_result.tokens_in
    tokens_out = 0
    cost_usd = embed_result.cost_usd
    model = embed_result.model

    if not grounded_chunks:
        response_text = (
            "I couldn't find a policy section that confidently answers this. "
            "Please rephrase, or reach out to HR directly for an authoritative answer."
        )
        sources: list[dict] = []
    else:
        context = "\n\n---\n\n".join(c.text for c in grounded_chunks)
        prompt = f"Policy excerpts:\n\n{context}\n\nEmployee question: {query}"
        gen_result = client.generate(
            model=config.FLASH_MODEL,
            system_instruction=POLICY_SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.1,
        )
        response_text = gen_result.text
        tokens_in += gen_result.tokens_in
        tokens_out += gen_result.tokens_out
        cost_usd += gen_result.cost_usd
        model = gen_result.model
        sources = [
            {"section": c.section_title, "distance": round(c.distance, 4)}
            for c in grounded_chunks
        ]

    latency_ms = (time.perf_counter() - start) * 1000
    tracer.log_step(
        turn_id=state["turn_id"],
        agent_name="policy_agent",
        input={"query": query},
        output={"response": response_text, "sources": sources},
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model=model,
    )

    return {
        "final_response": response_text,
        "history": [{"role": "assistant", "content": response_text, "agent": "policy_agent"}],
    }
