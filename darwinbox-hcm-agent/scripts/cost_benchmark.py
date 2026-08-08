"""Runs a fixed set of representative HR requests through two pipelines and
reports real, measured Gemini token/cost totals for each:

  naive      — every step forced through the larger Pro model, no regex
               routing shortcut, and the full policy document stuffed into
               context instead of top-k retrieved chunks.
  optimized  — this project's actual default pipeline: regex fast-path
               routing (skips an LLM call entirely for clear requests),
               Flash model, top-k grounded RAG context, and a single
               extraction call for action requests (templated response,
               no second generation call).

Usage:
    python scripts/cost_benchmark.py

Requires GEMINI_API_KEY in .env — this makes real (cheap) API calls.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.graph.action_agent import EXTRACTION_SCHEMA, EXTRACTION_SYSTEM_PROMPT
from src.graph.orchestrator import ROUTER_SCHEMA, ROUTER_SYSTEM_PROMPT, _regex_route
from src.graph.policy_agent import POLICY_SYSTEM_PROMPT
from src.llm.gemini_client import get_client
from src.observability.cost import compare
from src.rag.chunker import chunk_document
from src.rag.vector_store import VectorStore

SAMPLE_REQUESTS = [
    ("policy", "What is our maternity leave policy?"),
    ("policy", "How many days of earned leave am I entitled to per year?"),
    ("policy", "What is the notice period for resignation?"),
    ("action", "Check my leave balance for earned leave"),
    ("action", "Apply for 3 days of earned leave starting 2026-06-15"),
    ("action", "Can I get my payslip for June 2026?"),
]

# PRO_MODEL's free-tier quota in this project is tightly rate-limited
# (observed: 5 requests/minute). Pace naive-pipeline calls proactively
# rather than relying purely on reactive 429 retries.
NAIVE_CALL_SPACING_SECONDS = 13

NAIVE_RESPONSE_PROMPT = (
    "You are an HR assistant. Given the extracted tool call below, write a short, "
    "friendly natural-language response to the employee confirming or reporting the "
    "result.\n\nExtracted tool call: {payload}"
)


def _step(agent_name, result, extra=None):
    return {
        "agent_name": agent_name,
        "tokens_in": result.tokens_in,
        "tokens_out": getattr(result, "tokens_out", 0),
        "cost_usd": result.cost_usd,
        "model": result.model,
        **(extra or {}),
    }


def run_optimized(policy_text: str, store: VectorStore) -> list[dict]:
    client = get_client()
    steps = []

    for kind, text in SAMPLE_REQUESTS:
        route = _regex_route(text)  # zero-cost for every sample request here
        steps.append({"agent_name": "orchestrator_regex", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})

        if route == "policy":
            embed_result = client.embed([text])
            steps.append(_step("policy_embed", embed_result))
            chunks = store.query(embed_result.vectors[0], top_k=config.RAG_TOP_K)
            context = "\n\n---\n\n".join(c.text for c in chunks)
            gen = client.generate(
                model=config.FLASH_MODEL,
                system_instruction=POLICY_SYSTEM_PROMPT,
                prompt=f"Policy excerpts:\n\n{context}\n\nEmployee question: {text}",
                temperature=0.1,
            )
            steps.append(_step("policy_generate", gen))
        else:
            gen = client.generate(
                model=config.FLASH_MODEL,
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                prompt=f"Conversation so far:\n\nLatest message: {text}",
                response_schema=EXTRACTION_SCHEMA,
                temperature=0.0,
            )
            steps.append(_step("action_extract", gen))
            # Final response is templated in Python — zero additional LLM cost.

    return steps


def run_naive(policy_text: str) -> list[dict]:
    client = get_client()
    steps = []
    first_call = True

    def _paced_generate(**kwargs):
        nonlocal first_call
        if not first_call:
            time.sleep(NAIVE_CALL_SPACING_SECONDS)
        first_call = False
        return client.generate(**kwargs)

    for kind, text in SAMPLE_REQUESTS:
        route_gen = _paced_generate(
            model=config.PRO_MODEL,
            system_instruction=ROUTER_SYSTEM_PROMPT,
            prompt=text,
            response_schema=ROUTER_SCHEMA,
        )
        steps.append(_step("orchestrator_llm", route_gen))

        if kind == "policy":
            gen = _paced_generate(
                model=config.PRO_MODEL,
                system_instruction=POLICY_SYSTEM_PROMPT,
                prompt=f"Full policy document:\n\n{policy_text}\n\nEmployee question: {text}",
                temperature=0.1,
            )
            steps.append(_step("policy_generate_full_doc", gen))
        else:
            extract_gen = _paced_generate(
                model=config.PRO_MODEL,
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                prompt=f"Conversation so far:\n\nLatest message: {text}",
                response_schema=EXTRACTION_SCHEMA,
                temperature=0.0,
            )
            steps.append(_step("action_extract", extract_gen))

            respond_gen = _paced_generate(
                model=config.PRO_MODEL,
                system_instruction="You are a helpful HR assistant.",
                prompt=NAIVE_RESPONSE_PROMPT.format(payload=extract_gen.text),
            )
            steps.append(_step("action_respond_second_call", respond_gen))

    return steps


def main() -> None:
    policy_text = (config.DATA_DIR / "hr_policy.md").read_text(encoding="utf-8")
    store = VectorStore()
    if store.count() == 0:
        print("Vector index is empty — run `python -m src.rag.ingest` first.")
        sys.exit(1)

    print(f"Running {len(SAMPLE_REQUESTS)} sample requests through both pipelines...\n")

    optimized_steps = run_optimized(policy_text, store)
    naive_steps = run_naive(policy_text)

    result = compare(naive_steps, optimized_steps)

    print("=" * 60)
    print(f"{'':20}{'naive':>18}{'optimized':>20}")
    print("-" * 60)
    print(f"{'LLM calls':20}{result['naive'].llm_calls:>18}{result['optimized'].llm_calls:>20}")
    print(
        f"{'tokens in':20}{result['naive'].total_tokens_in:>18}"
        f"{result['optimized'].total_tokens_in:>20}"
    )
    print(
        f"{'tokens out':20}{result['naive'].total_tokens_out:>18}"
        f"{result['optimized'].total_tokens_out:>20}"
    )
    print(
        f"{'cost (USD)':20}{result['naive'].total_cost_usd:>18.6f}"
        f"{result['optimized'].total_cost_usd:>20.6f}"
    )
    print("=" * 60)
    print(f"Savings: {result['savings_pct']:.1f}% vs naive baseline")
    if result["savings_pct"] >= 20:
        print("PASS — meets the >=20% cost reduction target.")
    else:
        print("Below the 20% target — investigate before reporting in README.")

    out_path = config.ROOT_DIR / "traces" / "cost_benchmark_result.json"
    out_path.write_text(
        json.dumps(
            {
                "naive": vars(result["naive"]),
                "optimized": vars(result["optimized"]),
                "savings_pct": result["savings_pct"],
            },
            indent=2,
        )
    )
    print(f"\nSaved raw numbers to {out_path}")


if __name__ == "__main__":
    main()
