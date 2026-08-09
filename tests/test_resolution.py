from src.hitl import queue as hitl_queue
from src.rl.bandit import LinUCBBandit
from src.rl.resolution import resolve_decision


def _enqueue(queue_path):
    return hitl_queue.enqueue(
        employee_id="EMP1",
        anomaly_type="payroll_outlier",
        evidence={"z_score": 4.5},
        proposed_action="auto-correct",
        confidence=0.85,
        reasoning="pay 4.5 std devs above cohort median",
        context={"confidence": 0.85, "tenure_days": 900},
        path=queue_path,
    )


def test_resolve_approve_gives_positive_reward_and_persists_bandit(tmp_path):
    queue_path = tmp_path / "hitl_queue.json"
    bandit_path = tmp_path / "bandit_state.npz"
    item = _enqueue(queue_path)
    bandit = LinUCBBandit()

    result = resolve_decision(
        item.item_id, "approve", bandit,
        record_memory=False, queue_path=queue_path, bandit_path=bandit_path,
    )

    assert result.reward == 1.0
    assert result.resolved_action == "auto-correct"
    assert bandit_path.exists()
    resolved = hitl_queue.get(item.item_id, path=queue_path)
    assert resolved["status"] == "approved"


def test_resolve_reject_gives_negative_reward(tmp_path):
    queue_path = tmp_path / "hitl_queue.json"
    bandit_path = tmp_path / "bandit_state.npz"
    item = _enqueue(queue_path)
    bandit = LinUCBBandit()

    result = resolve_decision(
        item.item_id, "reject", bandit,
        rejection_reason="not accurate", record_memory=False,
        queue_path=queue_path, bandit_path=bandit_path,
    )

    assert result.reward == -1.0
    assert result.resolved_action == "no-action"


def test_resolve_modify_gives_partial_reward(tmp_path):
    queue_path = tmp_path / "hitl_queue.json"
    bandit_path = tmp_path / "bandit_state.npz"
    item = _enqueue(queue_path)
    bandit = LinUCBBandit()

    result = resolve_decision(
        item.item_id, "modify", bandit,
        modified_action="escalate-to-manager", record_memory=False,
        queue_path=queue_path, bandit_path=bandit_path,
    )

    assert -1.0 < result.reward < 1.0
    assert result.resolved_action == "escalate-to-manager"


def test_compliance_veto_pulls_reward_down(tmp_path):
    queue_path = tmp_path / "hitl_queue.json"
    bandit_path = tmp_path / "bandit_state.npz"
    item = _enqueue(queue_path)
    bandit = LinUCBBandit()

    result = resolve_decision(
        item.item_id, "approve", bandit,
        compliance_vetoed=True, record_memory=False,
        queue_path=queue_path, bandit_path=bandit_path,
    )

    assert result.reward == 1.0 - 1.5  # REWARD_APPROVE + REWARD_COMPLIANCE_VETO_PENALTY
