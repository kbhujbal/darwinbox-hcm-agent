"""Cost aggregation over trace steps, and the naive-vs-optimized comparator
used by scripts/cost_benchmark.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostSummary:
    total_calls: int
    llm_calls: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float


def summarize(steps: list[dict]) -> CostSummary:
    llm_calls = [s for s in steps if s.get("tokens_in", 0) or s.get("tokens_out", 0)]
    return CostSummary(
        total_calls=len(steps),
        llm_calls=len(llm_calls),
        total_tokens_in=sum(s.get("tokens_in", 0) for s in steps),
        total_tokens_out=sum(s.get("tokens_out", 0) for s in steps),
        total_cost_usd=sum(s.get("cost_usd", 0.0) for s in steps),
    )


def compare(naive_steps: list[dict], optimized_steps: list[dict]) -> dict:
    naive = summarize(naive_steps)
    optimized = summarize(optimized_steps)
    savings_pct = 0.0
    if naive.total_cost_usd > 0:
        savings_pct = (
            (naive.total_cost_usd - optimized.total_cost_usd) / naive.total_cost_usd * 100
        )
    return {"naive": naive, "optimized": optimized, "savings_pct": savings_pct}
