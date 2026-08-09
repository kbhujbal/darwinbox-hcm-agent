"""Reward computation combining HITL decisions, outcome recurrence, false
positives, and compliance vetoes — the four signals the assignment
specifies.
"""
from __future__ import annotations

from src.rl.bandit import ACTIONS

# Actions ordered by "how autonomous/aggressive" the response is, most to
# least: auto-correct (system acts immediately) -> escalate-to-manager ->
# escalate-to-HR -> flag-for-audit (log, no action yet) -> no-action.
# This ordering is what "edit distance" is adapted from below — a
# categorical action space has no native string-edit-distance, so we use
# normalized position difference along this ordering as our stand-in: a
# "modify" from auto-correct to escalate-to-manager is a small, mild
# correction; a modify from auto-correct to no-action is a large miss.
_ACTION_ORDER = {a: i for i, a in enumerate(ACTIONS)}
_MAX_DISTANCE = len(ACTIONS) - 1

REWARD_APPROVE = 1.0
REWARD_REJECT = -1.0
REWARD_RECURRENCE_PENALTY = -0.5
REWARD_FALSE_POSITIVE_PENALTY = -0.5
REWARD_COMPLIANCE_VETO_PENALTY = -1.5


def action_distance(a: str, b: str) -> float:
    if a not in _ACTION_ORDER or b not in _ACTION_ORDER:
        return 1.0
    return abs(_ACTION_ORDER[a] - _ACTION_ORDER[b]) / _MAX_DISTANCE


def reward_for_modify(proposed_action: str, modified_action: str) -> float:
    """+1 if the human's correction matches what was proposed (shouldn't
    happen — that'd be an approve), scaling down to -1 for a maximally
    distant substitution."""
    distance = action_distance(proposed_action, modified_action)
    return 1.0 - 2.0 * distance


def compute_hitl_reward(
    decision: str, proposed_action: str, modified_action: str | None = None
) -> float:
    if decision == "approve":
        return REWARD_APPROVE
    if decision == "reject":
        return REWARD_REJECT
    if decision == "modify":
        return reward_for_modify(proposed_action, modified_action or proposed_action)
    raise ValueError(f"Unknown HITL decision: {decision!r}")


def apply_outcome_adjustments(
    base_reward: float,
    recurred: bool = False,
    false_positive: bool = False,
    compliance_vetoed: bool = False,
) -> float:
    """Applied on top of the HITL-derived reward once later signals are
    known (an auto-corrected anomaly recurring, a flagged item turning out
    to be a data error, or the proposal having been compliance-vetoed)."""
    reward = base_reward
    if recurred:
        reward += REWARD_RECURRENCE_PENALTY
    if false_positive:
        reward += REWARD_FALSE_POSITIVE_PENALTY
    if compliance_vetoed:
        reward += REWARD_COMPLIANCE_VETO_PENALTY
    return reward
