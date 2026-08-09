"""Ingests a mock system-generated alert (payroll engine / attendance
system) and routes it through the same anomaly pipeline as a scheduled
scan, but targeted at just the employee named in the alert — this is the
third trigger class (reactive_nl, scheduled_scan, system_alert).
"""
from __future__ import annotations

from src.anomaly import scoring
from src.graph.anomaly_pipeline import ProcessedAnomaly, process_anomaly
from src.observability.tracer import Tracer
from src.rl.bandit import LinUCBBandit


def ingest_alert(
    alert: dict,
    bandit: LinUCBBandit,
    turn_id: int,
    tracer: Tracer,
    employees: list[dict] | None = None,
    use_memory: bool = True,
    bandit_path=None,
) -> ProcessedAnomaly | None:
    """alert: {"employee_id": str, "source": str, "alert_type": str, "details": dict}.

    Re-runs detection scoped to just that one employee (an upstream system
    alert is a hint to look, not a pre-scored anomaly) and processes the
    highest-confidence finding through the standard pipeline. Returns None
    if nothing in our own detectors corroborates the alert.
    """
    employees = employees if employees is not None else scoring.load_employees()
    employee = next((e for e in employees if e["employee_id"] == alert["employee_id"]), None)
    if employee is None:
        return None

    cohort_stats = scoring.compute_cohort_stats(employees)
    candidates = (
        scoring.detect_payroll_outliers([employee], cohort_stats)
        + scoring.detect_leave_abuse([employee])
        + scoring.detect_compliance_violations([employee])
    )
    if not candidates:
        return None

    anomaly = max(candidates, key=lambda a: a.confidence)
    return process_anomaly(
        anomaly,
        bandit,
        turn_id,
        tracer,
        signal_type="system_alert",
        use_memory=use_memory,
        bandit_path=bandit_path,
    )
