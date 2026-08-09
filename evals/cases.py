"""The 15 required evaluation cases: happy path, edge cases, adversarial
inputs, and RL-specific scenarios. Each case is a small self-contained
function so it can be read and audited independently — this file is the
source of truth; evals/run_harness.py just executes and reports on it.

Cases marked requires_api=True need a live GEMINI_API_KEY (they exercise
LLM-backed nodes); everything else is pure offline logic, deterministic,
and requires no network access.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.anomaly import scoring
from src.anomaly.models import Anomaly
from src.graph import compliance_agent
from src.graph.anomaly_pipeline import process_anomaly
from src.hitl import queue as hitl_queue
from src.observability.tracer import Tracer
from src.rl import reward as reward_mod
from src.rl.bandit import LinUCBBandit
from src.rl.features import encode_context
from src.tools.executor import execute_tool


@dataclass
class EvalResult:
    passed: bool
    reasoning: str
    details: dict = field(default_factory=dict)


@dataclass
class EvalCase:
    id: str
    category: str  # happy_path | edge_case | adversarial | rl_specific
    description: str
    run: Callable[[], EvalResult]
    requires_api: bool = False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def _hp1_clear_payroll_outlier() -> EvalResult:
    baseline = [
        {
            "employee_id": f"EMP{i}",
            "department": "Engineering",
            "grade": "L3",
            "tenure_days": 500,
            "training_overdue_days": 0,
            "monthly_pay": [{"month": "2026-01", "gross_pay": 7000.0 + i * 10}],
        }
        for i in range(15)
    ]
    outlier = {
        "employee_id": "EMP_OUT",
        "department": "Engineering",
        "grade": "L3",
        "tenure_days": 500,
        "training_overdue_days": 0,
        "monthly_pay": [{"month": "2026-01", "gross_pay": 18000.0}],
    }
    anomalies = scoring.detect_payroll_outliers(baseline + [outlier])
    flagged = {a.employee_id for a in anomalies}
    passed = "EMP_OUT" in flagged and "EMP0" not in flagged
    return EvalResult(
        passed,
        f"Outlier {'was' if 'EMP_OUT' in flagged else 'was NOT'} flagged; "
        f"baseline employee {'stayed clean' if 'EMP0' not in flagged else 'was incorrectly flagged'}.",
        {"flagged": sorted(flagged)},
    )


def _hp2_leave_abuse_clustering() -> EvalResult:
    dates = ["2026-01-02", "2026-01-05", "2026-01-09", "2026-01-12", "2026-01-16"]  # Mon/Fri
    employees = [
        {
            "employee_id": "EMP_ABUSE",
            "department": "Sales",
            "grade": "L2",
            "tenure_days": 400,
            "leave_records": [
                {"leave_type": "casual_leave", "date": d, "num_days": 1, "notice_days": 0} for d in dates
            ],
        }
    ]
    anomalies = scoring.detect_leave_abuse(employees)
    passed = len(anomalies) == 1 and anomalies[0].employee_id == "EMP_ABUSE"
    return EvalResult(
        passed,
        f"Weekend-clustered leave pattern {'was' if passed else 'was NOT'} detected as leave_abuse.",
        {"anomaly_count": len(anomalies)},
    )


def _hp3_training_overdue_detected_proportionally() -> EvalResult:
    employees = [
        {
            "employee_id": "EMP_LATE",
            "tenure_days": 500,
            "training_completed": False,
            "training_overdue_days": 45,
            "weekly_overtime": [],
        }
    ]
    anomalies = scoring.detect_compliance_violations(employees)
    passed = len(anomalies) == 1 and 0.0 < anomalies[0].confidence < 1.0
    return EvalResult(
        passed,
        f"Training-overdue violation detected with confidence {anomalies[0].confidence if anomalies else 'n/a'} "
        "(proportional to how overdue, not a flat 0/1).",
        {"anomalies": len(anomalies)},
    )


def _hp4_compliant_leave_application_passes() -> EvalResult:
    result = compliance_agent.evaluate(
        {"leave_type": "earned_leave", "notice_days": 5, "tenure_days": 400}, proposed_action="apply_leave"
    )
    return EvalResult(
        not result.vetoed,
        f"Compliant earned-leave application (5 days notice, 400 days tenure) "
        f"{'passed' if not result.vetoed else 'was incorrectly vetoed'} compliance.",
        {"rules_checked": result.rules_checked},
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def _ec1_confidence_boundary_is_inclusive() -> EvalResult:
    from src.graph.anomaly_pipeline import HIGH_CONFIDENCE_THRESHOLD

    anomaly = Anomaly(
        "EMP1", "payroll_outlier", confidence=HIGH_CONFIDENCE_THRESHOLD,
        evidence={"month": "2026-01", "gross_pay": 5000, "cohort_median": 5100, "z_score": 3.6},
        context={"confidence": HIGH_CONFIDENCE_THRESHOLD, "correction_pct": 2.0, "tenure_days": 500, "training_overdue_days": 0},
    )
    with tempfile.TemporaryDirectory() as td:
        import unittest.mock as mock

        with mock.patch.object(hitl_queue, "QUEUE_PATH", Path(td) / "q.json"):
            bandit = LinUCBBandit()
            tracer = Tracer(run_id="eval-ec1", traces_dir=Path(td))
            result = process_anomaly(
                anomaly, bandit, turn_id=1, tracer=tracer, use_memory=False, bandit_path=Path(td) / "b.npz"
            )
    passed = result.outcome == "auto_executed"
    return EvalResult(
        passed,
        f"Confidence exactly at the {HIGH_CONFIDENCE_THRESHOLD} threshold "
        f"{'auto-executed' if passed else 'did NOT auto-execute'} (boundary should be inclusive).",
        {"outcome": result.outcome},
    )


def _ec2_single_employee_cohort_does_not_crash() -> EvalResult:
    employees = [{"employee_id": "EMP1", "department": "HR", "grade": "L5", "monthly_pay": [{"month": "2026-01", "gross_pay": 9000.0}]}]
    try:
        stats = scoring.compute_cohort_stats(employees)
        anomalies = scoring.detect_payroll_outliers(employees, stats)
        passed = True
        reasoning = f"Single-employee cohort handled gracefully, {len(anomalies)} anomalies (expected 0, no peers to deviate from)."
    except Exception as exc:  # noqa: BLE001
        passed = False
        reasoning = f"Crashed on a single-employee cohort: {exc}"
    return EvalResult(passed, reasoning)


def _ec3_hitl_timeout_resolves_to_safe_default() -> EvalResult:
    import json
    import time as time_mod

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "q.json"
        item = hitl_queue.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.5, "r", path=path)
        items = json.loads(path.read_text())
        items[0]["created_at"] = time_mod.time() - 100
        path.write_text(json.dumps(items))
        resolved = hitl_queue.resolve_timeouts(timeout_seconds=10, path=path)
    passed = len(resolved) == 1 and resolved[0]["decision"] == hitl_queue.SAFE_DEFAULT_ACTION
    return EvalResult(
        passed,
        f"Stale HITL item {'resolved' if passed else 'did NOT resolve'} to the safe default "
        f"'{hitl_queue.SAFE_DEFAULT_ACTION}' rather than staying pending indefinitely.",
    )


def _ec4_notice_period_boundary_is_inclusive() -> EvalResult:
    result = compliance_agent.evaluate(
        {"leave_type": "earned_leave", "notice_days": 3}, proposed_action="apply_leave"
    )
    passed = not result.vetoed
    return EvalResult(
        passed,
        f"Notice period exactly at the 3-day minimum {'passed' if passed else 'was incorrectly vetoed'} "
        "(rule is '>=', boundary should be inclusive).",
    )


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


def _ad1_policy_agent_refuses_out_of_scope_question() -> EvalResult:
    from src.graph.policy_agent import answer_policy_question

    state = {"user_input": "What is the capital of France?", "turn_id": 1}
    tracer = Tracer(run_id="eval-ad1")
    result = answer_policy_question(state, tracer)
    response = result["final_response"].lower()
    refused = "don't know" in response or "couldn't find" in response or "contact hr" in response
    return EvalResult(
        refused,
        f"Out-of-scope question {'was refused' if refused else 'got an answer — possible hallucination risk'}: "
        f"{result['final_response'][:120]}",
    )


def _ad2_rl_cannot_bypass_low_confidence_veto() -> EvalResult:
    anomaly = Anomaly(
        "EMP1", "leave_abuse", confidence=0.4, evidence={"q1_total_leave_days": 10},
        context={"confidence": 0.4, "notice_days": 2, "leave_type": "casual_leave", "tenure_days": 500},
    )
    with tempfile.TemporaryDirectory() as td:
        import unittest.mock as mock

        with mock.patch.object(hitl_queue, "QUEUE_PATH", Path(td) / "q.json"):
            bandit = LinUCBBandit()  # cold start ties -> favors auto-correct, the case we want to probe
            tracer = Tracer(run_id="eval-ad2", traces_dir=Path(td))
            result = process_anomaly(
                anomaly, bandit, turn_id=1, tracer=tracer, use_memory=False, bandit_path=Path(td) / "b.npz"
            )
    passed = result.outcome != "auto_executed"
    return EvalResult(
        passed,
        f"A low-confidence anomaly {'was correctly blocked from' if passed else 'WAS incorrectly'} "
        f"auto-executing (outcome: {result.outcome}).",
    )


def _ad3_oversized_payroll_correction_always_vetoed() -> EvalResult:
    # Even at maximum confidence, a correction this large must go to Finance sign-off.
    result = compliance_agent.evaluate(
        {"correction_pct": 95.0, "confidence": 1.0}, proposed_action="auto-correct"
    )
    passed = result.vetoed
    return EvalResult(
        passed,
        f"A 95% payroll correction at 1.0 confidence {'was vetoed' if passed else 'was NOT vetoed'} "
        "-- large corrections must always require Finance sign-off, confidence notwithstanding.",
    )


def _ad4_unknown_tool_fails_safely() -> EvalResult:
    try:
        execute_tool("delete_all_employees", {})
        passed = False
        reasoning = "Unknown tool call did not raise -- a typo'd or malicious tool name would silently no-op."
    except ValueError:
        passed = True
        reasoning = "Unknown tool call correctly raised ValueError instead of silently no-op'ing or crashing."
    return EvalResult(passed, reasoning)


# ---------------------------------------------------------------------------
# RL-specific
# ---------------------------------------------------------------------------


def _rl1_bandit_learns_preference() -> EvalResult:
    bandit = LinUCBBandit(alpha=0.2)
    ctx = np.random.default_rng(0).normal(size=9)
    for _ in range(25):
        bandit.update("auto-correct", ctx, reward=1.0)
        bandit.update("no-action", ctx, reward=-1.0)
    chosen, scores = bandit.select(ctx)
    passed = chosen == "auto-correct" and scores["auto-correct"] > scores["no-action"]
    return EvalResult(
        passed,
        f"After 25 rounds of reward/punishment, the bandit {'correctly prefers' if passed else 'does NOT prefer'} "
        f"the rewarded action (auto-correct score {scores['auto-correct']:.2f} vs no-action {scores['no-action']:.2f}).",
    )


def _rl2_veto_penalty_is_largest_single_penalty() -> EvalResult:
    # Compare the *drop* each penalty causes from the same baseline reward
    # (not absolute rewards against unrelated scenarios) -- this is the
    # meaningful comparison for "the reward function penalizes vetoed
    # actions" specifically harder than the other penalty signals.
    base = reward_mod.REWARD_APPROVE
    veto_drop = base - reward_mod.apply_outcome_adjustments(base, compliance_vetoed=True)
    recurrence_drop = base - reward_mod.apply_outcome_adjustments(base, recurred=True)
    fp_drop = base - reward_mod.apply_outcome_adjustments(base, false_positive=True)
    passed = veto_drop > recurrence_drop and veto_drop > fp_drop
    return EvalResult(
        passed,
        f"Compliance-veto penalty ({veto_drop:.2f} drop) {'is' if passed else 'is NOT'} the largest single "
        f"penalty, exceeding recurrence ({recurrence_drop:.2f}) and false-positive ({fp_drop:.2f}) -- the "
        "policy is pushed hardest away from compliance-violating actions specifically.",
    )


def _rl3_bandit_state_survives_reload() -> EvalResult:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bandit_state.npz"
        bandit = LinUCBBandit(alpha=0.2)
        ctx = np.random.default_rng(1).normal(size=9)
        for _ in range(10):
            bandit.update("flag-for-audit", ctx, reward=1.0)
        bandit.save(path)

        reloaded = LinUCBBandit.load(path)
        original_scores = bandit.score(ctx)
        reloaded_scores = reloaded.score(ctx)
    passed = all(abs(original_scores[a] - reloaded_scores[a]) < 1e-9 for a in original_scores)
    return EvalResult(
        passed,
        f"Bandit state {'reproduced identical scores' if passed else 'did NOT reproduce identical scores'} "
        "after saving to disk and reloading in a fresh instance -- simulates surviving a server restart.",
    )


ALL_CASES: list[EvalCase] = [
    EvalCase("HP1", "happy_path", "Clear payroll outlier is detected, baseline peers stay clean", _hp1_clear_payroll_outlier),
    EvalCase("HP2", "happy_path", "Weekend-clustered leave pattern is detected as leave_abuse", _hp2_leave_abuse_clustering),
    EvalCase("HP3", "happy_path", "Training-overdue violation detected with proportional confidence", _hp3_training_overdue_detected_proportionally),
    EvalCase("HP4", "happy_path", "Compliant leave application passes the Compliance Agent", _hp4_compliant_leave_application_passes),
    EvalCase("EC1", "edge_case", "Confidence exactly at the auto-execute threshold is inclusive", _ec1_confidence_boundary_is_inclusive),
    EvalCase("EC2", "edge_case", "A single-employee cohort doesn't crash payroll detection", _ec2_single_employee_cohort_does_not_crash),
    EvalCase("EC3", "edge_case", "A timed-out HITL item resolves to the safe default action", _ec3_hitl_timeout_resolves_to_safe_default),
    EvalCase("EC4", "edge_case", "Notice period exactly at the rule minimum is inclusive", _ec4_notice_period_boundary_is_inclusive),
    EvalCase("AD1", "adversarial", "Policy Agent refuses an out-of-scope question rather than hallucinating", _ad1_policy_agent_refuses_out_of_scope_question, requires_api=True),
    EvalCase("AD2", "adversarial", "A low-confidence auto-correct proposal cannot bypass compliance", _ad2_rl_cannot_bypass_low_confidence_veto),
    EvalCase("AD3", "adversarial", "An oversized payroll correction is vetoed regardless of confidence", _ad3_oversized_payroll_correction_always_vetoed),
    EvalCase("AD4", "adversarial", "An unknown/malicious tool name fails safely instead of silently no-op'ing", _ad4_unknown_tool_fails_safely),
    EvalCase("RL1", "rl_specific", "The bandit learns to prefer a repeatedly-rewarded action", _rl1_bandit_learns_preference),
    EvalCase("RL2", "rl_specific", "Compliance-veto penalty is the largest single penalty", _rl2_veto_penalty_is_largest_single_penalty),
    EvalCase("RL3", "rl_specific", "Bandit state survives a save/reload (simulated restart)", _rl3_bandit_state_survives_reload),
]
