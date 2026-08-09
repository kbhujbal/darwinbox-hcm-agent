"""Anomaly Detection Agent graph node: wraps src/anomaly/scoring.scan()
into a traced graph step. Pure stats/rules under the hood — zero LLM cost,
even when scanning the full 600-employee dataset.
"""
from __future__ import annotations

import time

from src.anomaly import scoring
from src.anomaly.models import Anomaly
from src.observability.tracer import Tracer


def _count_by_type(anomalies: list[Anomaly]) -> dict:
    counts: dict[str, int] = {}
    for a in anomalies:
        counts[a.anomaly_type] = counts.get(a.anomaly_type, 0) + 1
    return counts


def run_scan(
    turn_id: int,
    tracer: Tracer,
    department_filter: str | None = None,
    leave_days_threshold: int | None = None,
    signal_type: str = "scheduled_scan",
) -> list[Anomaly]:
    start = time.perf_counter()
    anomalies = scoring.scan(
        department_filter=department_filter, leave_days_threshold=leave_days_threshold
    )
    latency_ms = (time.perf_counter() - start) * 1000

    tracer.log_step(
        turn_id=turn_id,
        agent_name="anomaly_detection_agent",
        input={"department_filter": department_filter, "leave_days_threshold": leave_days_threshold},
        output={"anomaly_count": len(anomalies), "by_type": _count_by_type(anomalies)},
        latency_ms=latency_ms,
        signal_type=signal_type,
    )
    return anomalies
