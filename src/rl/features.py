"""Turns an Anomaly into the fixed-length numeric context vector the LinUCB
bandit scores actions against. Deliberately hand-designed and explainable
(not an embedding) — separate from the semantic embedding used for
episodic-memory retrieval in src/rag/vector_store.py.
"""
from __future__ import annotations

import numpy as np

from src.anomaly.models import Anomaly
from src.rl.bandit import CONTEXT_DIM

ANOMALY_TYPES = ["payroll_outlier", "leave_abuse", "compliance_violation"]


def _severity(anomaly: Anomaly) -> float:
    ev = anomaly.evidence
    if anomaly.anomaly_type == "payroll_outlier":
        return min(1.0, abs(ev.get("z_score", 0.0)) / 8.0)
    if anomaly.anomaly_type == "leave_abuse":
        return min(1.0, ev.get("q1_total_leave_days", 0) / 30.0)
    if anomaly.anomaly_type == "compliance_violation":
        if ev.get("reason") == "training_overdue":
            return min(1.0, ev.get("overdue_days", 0) / 90.0)
        return min(1.0, ev.get("max_hours", 0) / 40.0)
    return 0.0


def encode_context(
    anomaly: Anomaly,
    past_similar_avg_reward: float = 0.0,
    has_prior_incident: bool = False,
) -> np.ndarray:
    tenure_days = anomaly.context.get("tenure_days", 0)
    vec = np.array(
        [
            1.0,  # bias
            anomaly.confidence,
            1.0 if anomaly.anomaly_type == "payroll_outlier" else 0.0,
            1.0 if anomaly.anomaly_type == "leave_abuse" else 0.0,
            1.0 if anomaly.anomaly_type == "compliance_violation" else 0.0,
            _severity(anomaly),
            min(1.0, tenure_days / 3650.0),
            past_similar_avg_reward,
            1.0 if has_prior_incident else 0.0,
        ],
        dtype=float,
    )
    assert vec.shape[0] == CONTEXT_DIM
    return vec
