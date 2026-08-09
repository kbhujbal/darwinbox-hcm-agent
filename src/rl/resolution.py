"""Resolves a HITL decision end-to-end: updates the queue, computes the
reward, updates + persists the bandit, and records the outcome to episodic
memory. Shared by the Streamlit Approvals tab and the batch feedback-cycle
simulator so both paths go through identical logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.anomaly.models import Anomaly
from src.hitl import queue as hitl_queue
from src.rl import episodic_memory
from src.rl import reward as reward_mod
from src.rl.bandit import DEFAULT_STATE_PATH, LinUCBBandit
from src.rl.features import encode_context


@dataclass
class ResolutionResult:
    item_id: str
    decision: str
    resolved_action: str
    reward: float


def resolve_decision(
    item_id: str,
    decision: str,
    bandit: LinUCBBandit,
    modified_action: str | None = None,
    rejection_reason: str | None = None,
    compliance_vetoed: bool = False,
    record_memory: bool = True,
    queue_path=None,
    bandit_path=None,
) -> ResolutionResult:
    item = hitl_queue.decide(
        item_id,
        decision,
        modified_action=modified_action,
        rejection_reason=rejection_reason,
        path=queue_path,
    )

    base_reward = reward_mod.compute_hitl_reward(decision, item["proposed_action"], modified_action)
    final_reward = reward_mod.apply_outcome_adjustments(base_reward, compliance_vetoed=compliance_vetoed)

    anomaly = Anomaly(
        employee_id=item["employee_id"],
        anomaly_type=item["anomaly_type"],
        confidence=item["confidence"],
        evidence=item["evidence"],
        context=item["context"],
    )
    context_vec = encode_context(anomaly)
    bandit.update(item["proposed_action"], context_vec, final_reward)
    bandit.save(bandit_path or DEFAULT_STATE_PATH)

    resolved_action = item["decision"] or "no-action"
    if record_memory:
        episodic_memory.record_incident(anomaly, action_taken=resolved_action, reward=final_reward)

    return ResolutionResult(
        item_id=item_id, decision=decision, resolved_action=resolved_action, reward=final_reward
    )
