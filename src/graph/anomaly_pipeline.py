"""Ties Anomaly Detection -> Episodic Memory -> RL Bandit -> Compliance ->
(auto-execute | HITL queue) together. This is the core of the Supervisor's
non-conversational routing — used for scheduled scans, system-generated
alerts, and reactive "flag anyone who..." NL requests alike.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.anomaly.models import Anomaly
from src.graph import compliance_agent
from src.hitl import queue as hitl_queue
from src.observability.tracer import Tracer
from src.rl import episodic_memory
from src.rl.bandit import DEFAULT_STATE_PATH, LinUCBBandit
from src.rl.features import encode_context
from src.tools.executor import execute_tool

HIGH_CONFIDENCE_THRESHOLD = 0.75

# Only "auto-correct" needs a real tool call — escalate-to-manager,
# escalate-to-HR, and flag-for-audit are internal categorizations (routing
# to a queue), not actions with an external system to call; no-action is a
# no-op. Only payroll_outlier and the training_overdue compliance_violation
# have a concrete corrective tool; other (anomaly_type, auto-correct)
# combinations fall through with no tool call.
_AUTO_CORRECT_DISPATCH = {"payroll_outlier", "compliance_violation"}


@dataclass
class ProcessedAnomaly:
    anomaly: Anomaly
    rl_action: str
    outcome: str  # auto_executed | queued_for_review | vetoed_queued
    reasoning: str
    tool_result: dict | None = None
    reward: float | None = None
    hitl_item_id: str | None = None


def _reasoning_text(anomaly: Anomaly, rl_action: str, similar_count: int) -> str:
    text = (
        f"{anomaly.anomaly_type.replace('_', ' ')} detected for {anomaly.employee_id} "
        f"(confidence {anomaly.confidence:.2f})."
    )
    if similar_count:
        text += f" {similar_count} similar past incident(s) found in memory."
    text += f" Recommended action: {rl_action}."
    return text


def _dispatch_auto_correct(anomaly: Anomaly) -> dict | None:
    if anomaly.anomaly_type == "payroll_outlier":
        result = execute_tool(
            "correct_payroll_discrepancy",
            {
                "employee_id": anomaly.employee_id,
                "month": anomaly.evidence.get("month", ""),
                "adjustment_amount": round(
                    anomaly.evidence.get("cohort_median", 0) - anomaly.evidence.get("gross_pay", 0), 2
                ),
                "reason": f"z-score {anomaly.evidence.get('z_score')} vs peer cohort",
            },
        )
        return result.output

    if anomaly.anomaly_type == "compliance_violation" and anomaly.evidence.get("reason") == "training_overdue":
        result = execute_tool(
            "remind_compliance_training", {"employee_id": anomaly.employee_id}
        )
        return result.output

    return None


def process_anomaly(
    anomaly: Anomaly,
    bandit: LinUCBBandit,
    turn_id: int,
    tracer: Tracer,
    signal_type: str = "scheduled_scan",
    use_memory: bool = True,
    bandit_path=None,
) -> ProcessedAnomaly:
    start = time.perf_counter()

    past_reward, has_prior, similar_count = 0.0, False, 0
    if use_memory:
        try:
            similar = episodic_memory.retrieve_similar(anomaly)
            similar_count = len(similar)
            if similar:
                past_reward, has_prior = episodic_memory.similarity_bias(anomaly, top_k=3)
                anomaly.confidence = min(
                    1.0, anomaly.confidence + episodic_memory.confidence_boost(anomaly, similar=similar)
                )
        except Exception:
            pass  # memory is a bias signal, never a hard dependency

    context_vec = encode_context(
        anomaly, past_similar_avg_reward=past_reward, has_prior_incident=has_prior
    )
    rl_action, _scores = bandit.select(context_vec)

    compliance_context = dict(anomaly.context)
    compliance_context["confidence"] = anomaly.confidence
    compliance_result = compliance_agent.evaluate(compliance_context, proposed_action=rl_action)

    reasoning = _reasoning_text(anomaly, rl_action, similar_count)
    reward = None
    tool_result = None
    hitl_item_id = None

    if compliance_result.vetoed:
        violated = "; ".join(v.description for v in compliance_result.violations)
        item = hitl_queue.enqueue(
            employee_id=anomaly.employee_id,
            anomaly_type=anomaly.anomaly_type,
            evidence=anomaly.evidence,
            proposed_action=rl_action,
            confidence=anomaly.confidence,
            reasoning=f"{reasoning} VETOED by compliance: {violated}",
            context=anomaly.context,
        )
        hitl_item_id = item.item_id
        reward = -1.5  # compliance-veto penalty, applied immediately regardless of later human review
        bandit.update(rl_action, context_vec, reward=reward)
        bandit.save(bandit_path or DEFAULT_STATE_PATH)
        outcome = "vetoed_queued"

    elif anomaly.confidence >= HIGH_CONFIDENCE_THRESHOLD:
        if rl_action == "auto-correct" and anomaly.anomaly_type in _AUTO_CORRECT_DISPATCH:
            tool_result = _dispatch_auto_correct(anomaly)
        outcome = "auto_executed"

    else:
        item = hitl_queue.enqueue(
            employee_id=anomaly.employee_id,
            anomaly_type=anomaly.anomaly_type,
            evidence=anomaly.evidence,
            proposed_action=rl_action,
            confidence=anomaly.confidence,
            reasoning=reasoning,
            context=anomaly.context,
        )
        hitl_item_id = item.item_id
        outcome = "queued_for_review"

    latency_ms = (time.perf_counter() - start) * 1000
    tracer.log_step(
        turn_id=turn_id,
        agent_name="rl_policy",
        input={
            "anomaly_type": anomaly.anomaly_type,
            "employee_id": anomaly.employee_id,
            "confidence": anomaly.confidence,
        },
        output={"outcome": outcome, "reasoning": reasoning, "tool_result": tool_result},
        latency_ms=latency_ms,
        signal_type=signal_type,
        rl_action_selected=rl_action,
        reward=reward,
        compliance_veto=compliance_result.vetoed,
    )

    return ProcessedAnomaly(
        anomaly=anomaly,
        rl_action=rl_action,
        outcome=outcome,
        reasoning=reasoning,
        tool_result=tool_result,
        reward=reward,
        hitl_item_id=hitl_item_id,
    )
