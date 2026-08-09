import pytest

from src.rl import reward as rw


def test_approve_gives_positive_reward():
    assert rw.compute_hitl_reward("approve", "auto-correct") == rw.REWARD_APPROVE


def test_reject_gives_negative_reward():
    assert rw.compute_hitl_reward("reject", "auto-correct") == rw.REWARD_REJECT


def test_modify_to_same_action_scores_like_approve():
    r = rw.compute_hitl_reward("modify", "auto-correct", modified_action="auto-correct")
    assert r == pytest.approx(1.0)


def test_modify_to_nearby_action_scores_mildly_positive():
    # auto-correct -> escalate-to-manager is the smallest possible substitution
    r = rw.compute_hitl_reward("modify", "auto-correct", modified_action="escalate-to-manager")
    assert 0.0 < r < 1.0


def test_modify_to_far_action_scores_negative():
    # auto-correct -> no-action is the largest possible substitution
    r = rw.compute_hitl_reward("modify", "auto-correct", modified_action="no-action")
    assert r == pytest.approx(-1.0)


def test_unknown_decision_raises():
    with pytest.raises(ValueError):
        rw.compute_hitl_reward("shrug", "auto-correct")


def test_recurrence_false_positive_and_veto_all_penalize():
    base = rw.REWARD_APPROVE
    assert rw.apply_outcome_adjustments(base, recurred=True) < base
    assert rw.apply_outcome_adjustments(base, false_positive=True) < base
    assert rw.apply_outcome_adjustments(base, compliance_vetoed=True) < base


def test_compliance_veto_penalty_is_the_largest_single_penalty():
    base = rw.REWARD_APPROVE
    veto_drop = base - rw.apply_outcome_adjustments(base, compliance_vetoed=True)
    recurrence_drop = base - rw.apply_outcome_adjustments(base, recurred=True)
    fp_drop = base - rw.apply_outcome_adjustments(base, false_positive=True)
    assert veto_drop > recurrence_drop
    assert veto_drop > fp_drop


def test_penalties_stack():
    base = rw.REWARD_APPROVE
    stacked = rw.apply_outcome_adjustments(base, recurred=True, false_positive=True, compliance_vetoed=True)
    expected = base + rw.REWARD_RECURRENCE_PENALTY + rw.REWARD_FALSE_POSITIVE_PENALTY + rw.REWARD_COMPLIANCE_VETO_PENALTY
    assert stacked == pytest.approx(expected)
