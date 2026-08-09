import random

from src.anomaly.models import Anomaly
from src.rl import simulated_reviewer as sim


def test_preferred_action_high_confidence_payroll_is_auto_correct():
    a = Anomaly("EMP1", "payroll_outlier", confidence=0.9, evidence={}, context={})
    assert sim.preferred_action(a) == "auto-correct"


def test_preferred_action_low_confidence_payroll_is_flag_for_audit():
    a = Anomaly("EMP1", "payroll_outlier", confidence=0.2, evidence={}, context={})
    assert sim.preferred_action(a) == "flag-for-audit"


def test_preferred_action_severe_overtime_is_flag_for_audit():
    a = Anomaly(
        "EMP1", "compliance_violation", confidence=0.9,
        evidence={"reason": "overtime_cap_breach", "max_hours": 22}, context={},
    )
    assert sim.preferred_action(a) == "flag-for-audit"


def test_matching_proposal_is_mostly_approved():
    a = Anomaly("EMP1", "payroll_outlier", confidence=0.9, evidence={}, context={})
    rng = random.Random(0)
    decisions = [sim.simulate_decision(a, "auto-correct", rng).decision for _ in range(200)]
    approve_rate = decisions.count("approve") / len(decisions)
    assert approve_rate > 0.85  # small amount of noise allowed, per _nearby_action


def test_close_mismatch_is_modified_to_ideal():
    a = Anomaly("EMP1", "payroll_outlier", confidence=0.9, evidence={}, context={})
    # ideal is auto-correct; escalate-to-manager is one step away (close)
    result = sim.simulate_decision(a, "escalate-to-manager", random.Random(1))
    assert result.decision == "modify"
    assert result.modified_action == "auto-correct"


def test_far_mismatch_is_rejected():
    a = Anomaly("EMP1", "payroll_outlier", confidence=0.9, evidence={}, context={})
    # ideal is auto-correct; no-action is maximally far.
    result = sim.simulate_decision(a, "no-action", random.Random(2))
    assert result.decision == "reject"
    assert result.reason is not None
