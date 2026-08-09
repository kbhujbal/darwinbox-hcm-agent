import json
import time

import pytest

from src.hitl import queue as hitl


@pytest.fixture
def queue_path(tmp_path):
    return tmp_path / "hitl_queue.json"


def test_enqueue_creates_pending_item(queue_path):
    item = hitl.enqueue(
        "EMP1", "payroll_outlier", {"z_score": 4.0}, "auto-correct", 0.6, "reasoning text", path=queue_path
    )
    assert item.status == "pending"
    pending = hitl.list_pending(path=queue_path)
    assert len(pending) == 1
    assert pending[0]["item_id"] == item.item_id


def test_decide_approve_sets_decision_to_proposed_action(queue_path):
    item = hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)
    resolved = hitl.decide(item.item_id, "approve", path=queue_path)
    assert resolved["status"] == "approved"
    assert resolved["decision"] == "auto-correct"


def test_decide_reject_has_no_decision_action(queue_path):
    item = hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)
    resolved = hitl.decide(item.item_id, "reject", rejection_reason="not accurate", path=queue_path)
    assert resolved["status"] == "rejected"
    assert resolved["decision"] is None
    assert resolved["rejection_reason"] == "not accurate"


def test_decide_modify_sets_decision_to_modified_action(queue_path):
    item = hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)
    resolved = hitl.decide(item.item_id, "modify", modified_action="escalate-to-manager", path=queue_path)
    assert resolved["status"] == "modified"
    assert resolved["decision"] == "escalate-to-manager"


def test_cannot_decide_an_already_resolved_item(queue_path):
    item = hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)
    hitl.decide(item.item_id, "approve", path=queue_path)
    with pytest.raises(ValueError):
        hitl.decide(item.item_id, "reject", path=queue_path)


def test_deciding_unknown_item_raises(queue_path):
    with pytest.raises(KeyError):
        hitl.decide("does-not-exist", "approve", path=queue_path)


def test_list_pending_excludes_resolved_items(queue_path):
    a = hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)
    hitl.enqueue("EMP2", "leave_abuse", {}, "escalate-to-manager", 0.5, "r", path=queue_path)
    hitl.decide(a.item_id, "approve", path=queue_path)

    pending = hitl.list_pending(path=queue_path)
    assert len(pending) == 1
    assert pending[0]["employee_id"] == "EMP2"


def test_timeout_resolves_stale_items_to_safe_default(queue_path):
    item = hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)

    # Manually backdate created_at past the timeout window.
    items = json.loads(queue_path.read_text())
    items[0]["created_at"] = time.time() - 100
    queue_path.write_text(json.dumps(items))

    resolved = hitl.resolve_timeouts(timeout_seconds=10, path=queue_path)
    assert len(resolved) == 1
    assert resolved[0]["status"] == "timed_out"
    assert resolved[0]["decision"] == hitl.SAFE_DEFAULT_ACTION
    assert hitl.list_pending(path=queue_path) == []


def test_recent_items_are_not_timed_out(queue_path):
    hitl.enqueue("EMP1", "payroll_outlier", {}, "auto-correct", 0.6, "r", path=queue_path)
    resolved = hitl.resolve_timeouts(timeout_seconds=24 * 3600, path=queue_path)
    assert resolved == []
    assert len(hitl.list_pending(path=queue_path)) == 1
