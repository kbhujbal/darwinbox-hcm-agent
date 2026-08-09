"""Heuristic batch reviewer standing in for a real HR manager — used to
generate enough HITL decision volume across feedback cycles for the RL
bandit to show a real learning trend (see scripts/run_feedback_cycles.py).
The real Streamlit Approvals tab exists alongside this for genuine manual
review during the demo; this is only for bulk training signal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.anomaly.models import Anomaly
from src.rl.bandit import ACTIONS
from src.rl.reward import action_distance

CLOSE_ENOUGH_DISTANCE = 0.25


@dataclass
class SimulatedDecision:
    decision: str  # approve | reject | modify
    modified_action: str | None
    reason: str | None


def preferred_action(anomaly: Anomaly) -> str:
    """The heuristic 'ideal' action a careful reviewer would pick — the
    hidden pattern the bandit needs to learn to approximate across cycles."""
    if anomaly.anomaly_type == "payroll_outlier":
        if anomaly.confidence >= 0.8:
            return "auto-correct"
        if anomaly.confidence >= 0.5:
            return "escalate-to-manager"
        return "flag-for-audit"

    if anomaly.anomaly_type == "leave_abuse":
        return "flag-for-audit" if anomaly.confidence >= 0.9 else "escalate-to-manager"

    if anomaly.anomaly_type == "compliance_violation":
        if anomaly.evidence.get("reason") == "training_overdue":
            if anomaly.confidence < 0.5:
                return "auto-correct"
            if anomaly.confidence < 0.85:
                return "escalate-to-manager"
            return "escalate-to-HR"
        return "flag-for-audit" if anomaly.evidence.get("max_hours", 0) >= 20 else "escalate-to-manager"

    return "flag-for-audit"


def _nearby_action(action: str, rng: random.Random) -> str:
    idx = ACTIONS.index(action)
    candidates = [a for i, a in enumerate(ACTIONS) if abs(i - idx) == 1]
    return rng.choice(candidates) if candidates else action


def simulate_decision(
    anomaly: Anomaly, proposed_action: str, rng: random.Random | None = None
) -> SimulatedDecision:
    rng = rng or random.Random()
    ideal = preferred_action(anomaly)
    distance = action_distance(proposed_action, ideal)

    if proposed_action == ideal:
        if rng.random() < 0.08:  # a little reviewer noise even on a good call
            return SimulatedDecision(
                "modify", _nearby_action(ideal, rng), "reviewer preference, minor adjustment"
            )
        return SimulatedDecision("approve", None, None)

    if distance <= CLOSE_ENOUGH_DISTANCE:
        return SimulatedDecision(
            "modify", ideal, f"closer fit for a {anomaly.anomaly_type} case at this confidence"
        )

    return SimulatedDecision(
        "reject", None, f"proposed action too far from what this {anomaly.anomaly_type} case warrants"
    )
