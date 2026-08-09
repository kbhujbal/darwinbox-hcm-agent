"""Handles reactive NL anomaly-flagging requests, e.g. "Flag anyone in
Engineering who has taken more than 15 days leave in Q1".

Deliberately a *reporting* path, not an acting one: it runs detection only
(src.anomaly.scoring.scan) and summarizes results in the chat — it does not
run the RL/compliance/auto-execute pipeline. That pipeline (src.graph.
anomaly_pipeline.process_anomaly) is reserved for scheduled scans and
system-generated alerts, where autonomous correction is actually
appropriate; a chat question shouldn't have side effects on its own.
"""
from __future__ import annotations

import time

from src import config
from src.anomaly import scoring
from src.graph.state import HCMState
from src.llm.gemini_client import get_client
from src.observability.tracer import Tracer

_VALID_DEPARTMENTS = {
    "Engineering", "Sales", "Finance", "HR", "Operations", "Marketing", "Customer Support",
}

QUERY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {"type": "string"},
        "leave_days_threshold": {"type": "integer"},
    },
    "required": ["department", "leave_days_threshold"],
}

QUERY_EXTRACTION_SYSTEM_PROMPT = (
    "Extract filter parameters from this HR anomaly-flagging request. Valid departments: "
    "Engineering, Sales, Finance, HR, Operations, Marketing, Customer Support (empty string "
    "if none is mentioned). leave_days_threshold is a number of leave days mentioned as a "
    'cutoff (0 if none is mentioned). Respond with ONLY a JSON object of the form '
    '{"department": "...", "leave_days_threshold": N}, no other text.'
)


def _format_response(anomalies, department_filter, leave_days_threshold) -> str:
    scope = f" in {department_filter}" if department_filter else ""
    if not anomalies:
        return f"No anomalies found{scope}."

    plural = "anomaly" if len(anomalies) == 1 else "anomalies"
    lines = [f"Found {len(anomalies)} {plural}{scope}:"]
    for a in anomalies[:15]:
        evidence = ", ".join(f"{k}={v}" for k, v in a.evidence.items())
        lines.append(
            f"- {a.employee_id} ({a.anomaly_type.replace('_', ' ')}, "
            f"confidence {a.confidence:.2f}): {evidence}"
        )
    if len(anomalies) > 15:
        lines.append(f"...and {len(anomalies) - 15} more.")
    return "\n".join(lines)


def answer_anomaly_query(state: HCMState, tracer: Tracer) -> dict:
    start = time.perf_counter()
    query = state["user_input"]

    client = get_client()
    gen_result = client.generate(
        model=config.FLASH_MODEL,
        system_instruction=QUERY_EXTRACTION_SYSTEM_PROMPT,
        prompt=query,
        response_schema=QUERY_EXTRACTION_SCHEMA,
        temperature=0.0,
    )

    department_filter = None
    leave_days_threshold = None
    try:
        import json

        parsed = gen_result.parsed or json.loads(gen_result.text)
        dept = parsed.get("department", "")
        if dept in _VALID_DEPARTMENTS:
            department_filter = dept
        threshold = parsed.get("leave_days_threshold", 0)
        if threshold:
            leave_days_threshold = int(threshold)
    except (ValueError, TypeError, AttributeError):
        pass

    anomalies = scoring.scan(department_filter=department_filter, leave_days_threshold=leave_days_threshold)
    response_text = _format_response(anomalies, department_filter, leave_days_threshold)

    latency_ms = (time.perf_counter() - start) * 1000
    tracer.log_step(
        turn_id=state["turn_id"],
        agent_name="anomaly_query_agent",
        input={"query": query, "department_filter": department_filter, "leave_days_threshold": leave_days_threshold},
        output={"response": response_text, "anomaly_count": len(anomalies)},
        latency_ms=latency_ms,
        tokens_in=gen_result.tokens_in,
        tokens_out=gen_result.tokens_out,
        cost_usd=gen_result.cost_usd,
        model=gen_result.model,
        signal_type="reactive_nl",
    )

    return {
        "final_response": response_text,
        "history": [{"role": "assistant", "content": response_text, "agent": "anomaly_query_agent"}],
    }
