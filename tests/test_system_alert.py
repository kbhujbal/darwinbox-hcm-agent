from src.graph.system_alert import ingest_alert
from src.hitl import queue as hitl_queue
from src.observability.tracer import Tracer
from src.rl.bandit import LinUCBBandit


def _employee(employee_id, training_completed=True, training_overdue_days=0):
    return {
        "employee_id": employee_id,
        "name": "Test Employee",
        "department": "Engineering",
        "grade": "L3",
        "tenure_days": 500,
        "manager_id": None,
        "monthly_pay": [{"month": "2026-01", "gross_pay": 7000.0}],
        "weekly_overtime": [{"week": f"2026-W{i:02d}", "hours": 4.0} for i in range(1, 13)],
        "leave_records": [],
        "training_completed": training_completed,
        "training_overdue_days": training_overdue_days,
        "performance_rating": 3.5,
    }


def test_ingest_alert_processes_corroborated_anomaly(tmp_path, monkeypatch):
    monkeypatch.setattr(hitl_queue, "QUEUE_PATH", tmp_path / "hitl_queue.json")
    employees = [_employee("EMP_LATE", training_completed=False, training_overdue_days=40)]
    bandit = LinUCBBandit()
    tracer = Tracer(run_id="test-alert", traces_dir=tmp_path)

    result = ingest_alert(
        {"employee_id": "EMP_LATE", "source": "attendance_system", "alert_type": "compliance_check"},
        bandit,
        turn_id=1,
        tracer=tracer,
        employees=employees,
        use_memory=False,
        bandit_path=tmp_path / "bandit_state.npz",
    )

    assert result is not None
    assert result.anomaly.anomaly_type == "compliance_violation"

    steps = Tracer.read_run("test-alert", traces_dir=tmp_path)
    assert steps[0]["signal_type"] == "system_alert"


def test_ingest_alert_returns_none_for_unknown_employee(tmp_path):
    bandit = LinUCBBandit()
    tracer = Tracer(run_id="test-alert-2", traces_dir=tmp_path)
    result = ingest_alert(
        {"employee_id": "DOES_NOT_EXIST", "source": "payroll_engine", "alert_type": "x"},
        bandit,
        turn_id=1,
        tracer=tracer,
        employees=[_employee("EMP_CLEAN")],
        use_memory=False,
    )
    assert result is None


def test_ingest_alert_returns_none_when_no_anomaly_corroborated(tmp_path):
    bandit = LinUCBBandit()
    tracer = Tracer(run_id="test-alert-3", traces_dir=tmp_path)
    result = ingest_alert(
        {"employee_id": "EMP_CLEAN", "source": "payroll_engine", "alert_type": "x"},
        bandit,
        turn_id=1,
        tracer=tracer,
        employees=[_employee("EMP_CLEAN")],
        use_memory=False,
    )
    assert result is None
