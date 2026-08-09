from src.anomaly.models import Anomaly
from src.graph.anomaly_pipeline import process_anomaly
from src.hitl import queue as hitl_queue
from src.observability.tracer import Tracer
from src.rl.bandit import LinUCBBandit


def _high_confidence_clean_payroll_anomaly():
    return Anomaly(
        employee_id="EMP1",
        anomaly_type="payroll_outlier",
        confidence=0.9,
        evidence={"month": "2026-02", "gross_pay": 3100.0, "cohort_median": 3200.0, "z_score": 5.0},
        # correction_pct kept under the 10% auto-correct tier so this case
        # is compliant no matter which action the (cold-start) bandit picks.
        context={"confidence": 0.9, "correction_pct": 3.0, "tenure_days": 900, "training_overdue_days": 0},
    )


def _low_confidence_anomaly():
    return Anomaly(
        employee_id="EMP2",
        anomaly_type="leave_abuse",
        confidence=0.4,
        evidence={"q1_total_leave_days": 10},
        context={"confidence": 0.4, "notice_days": 2, "leave_type": "casual_leave", "tenure_days": 500},
    )


def _compliance_violating_anomaly():
    # tenure_days < 90 with an implied earned_leave-style veto trigger isn't
    # directly applicable here; instead use a low-confidence auto-correct
    # trigger via the payroll_correction_requires_confidence rule by forcing
    # a context where confidence is below the 0.75 compliance floor but the
    # bandit (untouched, cold-start) might still pick auto-correct.
    return Anomaly(
        employee_id="EMP3",
        anomaly_type="payroll_outlier",
        confidence=0.99,
        evidence={"month": "2026-02", "gross_pay": 9000.0, "cohort_median": 3200.0, "z_score": 6.0},
        # correction_pct way over the 10% auto-correct tier -> hard veto regardless of confidence
        context={"confidence": 0.99, "correction_pct": 80.0, "tenure_days": 900, "training_overdue_days": 0},
    )


def test_high_confidence_clean_anomaly_auto_executes(tmp_path, monkeypatch):
    bandit = LinUCBBandit()
    tracer = Tracer(run_id="test-pipeline", traces_dir=tmp_path)
    monkeypatch.setattr(hitl_queue, "QUEUE_PATH", tmp_path / "hitl_queue.json")

    result = process_anomaly(
        _high_confidence_clean_payroll_anomaly(),
        bandit,
        turn_id=1,
        tracer=tracer,
        use_memory=False,
        bandit_path=tmp_path / "bandit_state.npz",
    )

    assert result.outcome == "auto_executed"
    assert result.hitl_item_id is None


def test_low_confidence_anomaly_is_queued_for_review(tmp_path, monkeypatch):
    from src.rl.features import encode_context

    bandit = LinUCBBandit(alpha=0.1)
    anomaly = _low_confidence_anomaly()
    # Bias the bandit toward escalate-to-manager for this exact context so
    # the test isolates the "low confidence -> human review" path from the
    # separately-tested "auto-correct proposed at low confidence -> veto" path.
    ctx = encode_context(anomaly)
    for _ in range(30):
        bandit.update("escalate-to-manager", ctx, reward=1.0)

    tracer = Tracer(run_id="test-pipeline-2", traces_dir=tmp_path)
    queue_path = tmp_path / "hitl_queue.json"
    monkeypatch.setattr(hitl_queue, "QUEUE_PATH", queue_path)

    result = process_anomaly(
        _low_confidence_anomaly(), bandit, turn_id=1, tracer=tracer, use_memory=False,
        bandit_path=tmp_path / "bandit_state.npz",
    )

    assert result.outcome == "queued_for_review"
    assert result.hitl_item_id is not None
    pending = hitl_queue.list_pending(path=queue_path)
    assert len(pending) == 1
    assert pending[0]["employee_id"] == "EMP2"


def test_low_confidence_auto_correct_proposal_gets_vetoed_not_silently_auto_executed(tmp_path, monkeypatch):
    """A cold-start (or otherwise) bandit that happens to propose
    auto-correct for a low-confidence anomaly must be blocked by compliance
    -- it should never slip through as auto_executed."""
    bandit = LinUCBBandit()  # default cold-start ties favor the first action, auto-correct
    tracer = Tracer(run_id="test-pipeline-veto-lowconf", traces_dir=tmp_path)
    queue_path = tmp_path / "hitl_queue.json"
    monkeypatch.setattr(hitl_queue, "QUEUE_PATH", queue_path)

    result = process_anomaly(
        _low_confidence_anomaly(), bandit, turn_id=1, tracer=tracer, use_memory=False,
        bandit_path=tmp_path / "bandit_state.npz",
    )

    assert result.outcome != "auto_executed"


def test_compliance_violation_forces_veto_regardless_of_confidence(tmp_path, monkeypatch):
    bandit = LinUCBBandit(alpha=0.1)
    tracer = Tracer(run_id="test-pipeline-3", traces_dir=tmp_path)
    queue_path = tmp_path / "hitl_queue.json"
    bandit_path = tmp_path / "bandit_state.npz"
    monkeypatch.setattr(hitl_queue, "QUEUE_PATH", queue_path)

    # Force the bandit to prefer auto-correct for this exact context so the
    # veto path (not a low-confidence path) is what's actually exercised.
    from src.rl.features import encode_context

    anomaly = _compliance_violating_anomaly()
    ctx = encode_context(anomaly)
    for _ in range(40):
        bandit.update("auto-correct", ctx, reward=1.0)

    result = process_anomaly(
        anomaly, bandit, turn_id=1, tracer=tracer, use_memory=False, bandit_path=bandit_path
    )

    assert result.rl_action == "auto-correct"
    assert result.outcome == "vetoed_queued"
    assert result.reward == -1.5


def test_trace_step_records_rl_fields(tmp_path, monkeypatch):
    bandit = LinUCBBandit()
    tracer = Tracer(run_id="test-pipeline-4", traces_dir=tmp_path)
    queue_path = tmp_path / "hitl_queue.json"
    monkeypatch.setattr(hitl_queue, "QUEUE_PATH", queue_path)

    process_anomaly(
        _low_confidence_anomaly(), bandit, turn_id=7, tracer=tracer, use_memory=False,
        bandit_path=tmp_path / "bandit_state.npz",
    )

    steps = Tracer.read_run("test-pipeline-4", traces_dir=tmp_path)
    assert len(steps) == 1
    step = steps[0]
    assert step["agent_name"] == "rl_policy"
    assert step["rl_action_selected"] in {
        "auto-correct", "escalate-to-manager", "escalate-to-HR", "flag-for-audit", "no-action",
    }
    assert step["compliance_veto"] in (True, False)
