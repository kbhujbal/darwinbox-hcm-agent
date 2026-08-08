"""Orchestrator node: routes the user's message to the Policy Agent or the
Action Agent.

This is the main cost-optimization lever in the system. Clearly-worded
requests are routed by a zero-cost regex classifier. Only ambiguous
requests fall back to a single Gemini Flash classification call — the
naive baseline (see scripts/cost_benchmark.py) instead routes every
request through an LLM call with no shortcut.
"""
from __future__ import annotations

import json
import re
import time

from src import config
from src.graph.state import HCMState
from src.llm.gemini_client import get_client
from src.observability.tracer import Tracer

ACTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bapply\b.*\bleave\b",
        r"\bleave\s+balance\b",
        r"\bcheck\b.*\bbalance\b",
        r"\bpayslip\b",
        r"\bpay\s*slip\b",
        r"\bsalary\s+slip\b",
        r"\bhow\s+many\s+.*\bleave\s+days?\s+do\s+i\s+have\b",
    ]
]

POLICY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpolicy\b",
        r"\bmaternity\b",
        r"\bpaternity\b",
        r"\bbereavement\b",
        r"\bnotice\s+period\b",
        r"\bprobation\b",
        r"\bovertime\b",
        r"\bentitled\b",
        r"\bhow many days\b",
        r"\bwhat is\b",
        r"\bwhat's\b",
        r"\bhow do i\b",
    ]
]

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["policy", "action", "clarify"]},
        "confidence": {"type": "number"},
    },
    "required": ["route", "confidence"],
}

ROUTER_SYSTEM_PROMPT = (
    "You are the routing classifier for an HR assistant. Classify the user's message "
    "into exactly one of: 'policy' (a question about HR policy, entitlements, or rules), "
    "'action' (a request to check leave balance, apply for leave, or fetch a payslip), or "
    "'clarify' (too ambiguous to route safely). "
    'Respond with ONLY a JSON object of the form {"route": "...", "confidence": 0.0-1.0}, '
    "no other text."
)


def _regex_route(text: str) -> str | None:
    if any(p.search(text) for p in ACTION_PATTERNS):
        return "action"
    if any(p.search(text) for p in POLICY_PATTERNS):
        return "policy"
    return None


def route_request(state: HCMState, tracer: Tracer) -> dict:
    start = time.perf_counter()
    user_input = state["user_input"]

    tokens_in = tokens_out = 0
    cost_usd = 0.0
    model = None

    if state.get("pending_action"):
        # Mid slot-filling from a previous turn — go straight back to the
        # Action Agent, no re-classification needed.
        route = "action"
        method = "pending_action_continuation"
    else:
        regex_route = _regex_route(user_input)
        if regex_route:
            route = regex_route
            method = "regex"
        else:
            client = get_client()
            result = client.generate(
                model=config.FLASH_MODEL,
                system_instruction=ROUTER_SYSTEM_PROMPT,
                prompt=user_input,
                response_schema=ROUTER_SCHEMA,
            )
            tokens_in, tokens_out, cost_usd, model = (
                result.tokens_in,
                result.tokens_out,
                result.cost_usd,
                result.model,
            )
            try:
                parsed = result.parsed or json.loads(result.text)
                route = parsed["route"]
                confidence = float(parsed["confidence"])
            except (ValueError, KeyError, TypeError):
                route, confidence = "clarify", 0.0
            if confidence < config.ROUTING_CONFIDENCE_THRESHOLD:
                route = "clarify"
            method = "llm_fallback"

    latency_ms = (time.perf_counter() - start) * 1000
    tracer.log_step(
        turn_id=state["turn_id"],
        agent_name="orchestrator",
        input={"user_input": user_input, "method": method},
        output={"route": route},
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model=model,
    )

    return {"route": route}
