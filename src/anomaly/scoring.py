"""Anomaly Detection Agent's core logic: pure statistics and rule checks,
no LLM involved — this is the cost-optimization lever that lets scanning
hundreds of employee records cost nothing. Each detector returns Anomaly
objects with a confidence score and a *prior* suggested action; the actual
action decision is made by the RL bandit (src/rl/bandit.py), which uses
these as one of its context signals rather than following them blindly.
"""
from __future__ import annotations

import datetime
import json
import statistics

from src import config
from src.anomaly.models import Anomaly

Q1_MONTHS = {"2026-01", "2026-02", "2026-03"}


def load_employees(path=None) -> list[dict]:
    path = path or (config.DATA_DIR / "employees_dataset.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _weekday(date_str: str) -> int:
    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime.date(y, m, d).weekday()  # 0=Mon .. 6=Sun


def _grade_band(grade: str) -> str:
    return "manager" if grade in ("L4", "L5") else "non_manager"


_MAD_TO_STDEV = 1.4826  # consistency constant so MAD approximates stdev under normality


def compute_cohort_stats(employees: list[dict]) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(department, grade, month) -> (median, scaled MAD) of gross_pay across the cohort.

    Median + MAD (not mean + stdev) deliberately: a single extreme outlier in
    a cohort of ~15-20 people drags the mean and inflates the stdev enough to
    mask its own z-score ("self-contamination"). The median and median
    absolute deviation barely move under one contaminated point, so the
    outlier's computed deviation stays large — this is what actually let
    detection recall go from ~0.5 to ~1.0 on the injected outliers.
    """
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for e in employees:
        for mp in e["monthly_pay"]:
            key = (e["department"], e["grade"], mp["month"])
            buckets.setdefault(key, []).append(mp["gross_pay"])

    stats = {}
    for key, values in buckets.items():
        median = statistics.median(values)
        abs_devs = [abs(v - median) for v in values]
        mad = statistics.median(abs_devs)
        scale = mad * _MAD_TO_STDEV if mad > 0 else max(median * 0.05, 1.0)
        stats[key] = (median, scale)
    return stats


def detect_payroll_outliers(
    employees: list[dict],
    cohort_stats: dict | None = None,
    z_threshold: float = 3.5,
) -> list[Anomaly]:
    cohort_stats = cohort_stats or compute_cohort_stats(employees)
    anomalies = []

    for e in employees:
        for mp in e["monthly_pay"]:
            key = (e["department"], e["grade"], mp["month"])
            median, scale = cohort_stats[key]
            if scale == 0:
                continue
            z = (mp["gross_pay"] - median) / scale
            if abs(z) < z_threshold:
                continue

            confidence = round(min(1.0, abs(z) / 6.0), 3)
            correction_pct = round(abs(mp["gross_pay"] - median) / median * 100, 2) if median else 0.0
            anomalies.append(
                Anomaly(
                    employee_id=e["employee_id"],
                    anomaly_type="payroll_outlier",
                    confidence=confidence,
                    evidence={
                        "month": mp["month"],
                        "gross_pay": mp["gross_pay"],
                        "cohort_median": round(median, 2),
                        "z_score": round(z, 2),
                    },
                    context={
                        "confidence": confidence,
                        "correction_pct": correction_pct,
                        "tenure_days": e["tenure_days"],
                        "training_overdue_days": e["training_overdue_days"],
                    },
                    suggested_action_prior="auto-correct" if confidence >= 0.85 else "escalate-to-manager",
                )
            )
    return anomalies


def detect_leave_abuse(
    employees: list[dict],
    quarter_cap_days: int = 15,
    weekend_adjacency_ratio_threshold: float = 0.6,
) -> list[Anomaly]:
    anomalies = []

    for e in employees:
        q1_records = [r for r in e["leave_records"] if r["date"][:7] in Q1_MONTHS]
        if not q1_records:
            continue

        total_days = sum(r["num_days"] for r in q1_records)
        weekend_adjacent = sum(1 for r in q1_records if _weekday(r["date"]) in (0, 4))
        ratio = weekend_adjacent / len(q1_records)

        over_cap = total_days > quarter_cap_days
        clustered = len(q1_records) >= 4 and ratio >= weekend_adjacency_ratio_threshold
        if not (over_cap or clustered):
            continue

        cap_excess = max(0, total_days - quarter_cap_days)
        confidence = round(min(1.0, 0.5 + 0.05 * cap_excess + 0.3 * ratio), 3)
        min_notice = min((r["notice_days"] for r in q1_records), default=0)
        anomalies.append(
            Anomaly(
                employee_id=e["employee_id"],
                anomaly_type="leave_abuse",
                confidence=confidence,
                evidence={
                    "q1_total_leave_days": total_days,
                    "weekend_adjacent_count": weekend_adjacent,
                    "weekend_adjacency_ratio": round(ratio, 2),
                },
                context={
                    "confidence": confidence,
                    "notice_days": min_notice,
                    "leave_type": q1_records[0]["leave_type"],
                    "tenure_days": e["tenure_days"],
                },
                suggested_action_prior="flag-for-audit" if confidence >= 0.9 else "escalate-to-manager",
            )
        )
    return anomalies


def detect_compliance_violations(
    employees: list[dict],
    overtime_cap: float = 12.0,
    persistent_week_count: int = 2,
    severe_single_week_hours: float = 20.0,
) -> list[Anomaly]:
    anomalies = []

    for e in employees:
        if e["training_overdue_days"] > 0:
            confidence = round(min(1.0, e["training_overdue_days"] / 60.0), 3)
            anomalies.append(
                Anomaly(
                    employee_id=e["employee_id"],
                    anomaly_type="compliance_violation",
                    confidence=confidence,
                    evidence={"reason": "training_overdue", "overdue_days": e["training_overdue_days"]},
                    context={
                        "confidence": confidence,
                        "training_overdue_days": e["training_overdue_days"],
                        "tenure_days": e["tenure_days"],
                    },
                    suggested_action_prior="auto-correct" if confidence < 0.7 else "escalate-to-HR",
                )
            )

        breach_weeks = [w for w in e["weekly_overtime"] if w["hours"] > overtime_cap]
        if len(breach_weeks) >= persistent_week_count:
            max_hours = max(w["hours"] for w in breach_weeks)
            confidence = round(min(1.0, len(breach_weeks) / 6.0 + max_hours / 40.0), 3)
            anomalies.append(
                Anomaly(
                    employee_id=e["employee_id"],
                    anomaly_type="compliance_violation",
                    confidence=confidence,
                    evidence={
                        "reason": "overtime_cap_breach",
                        "breach_week_count": len(breach_weeks),
                        "max_hours": max_hours,
                    },
                    context={
                        "confidence": confidence,
                        "overtime_hours_week": max_hours,
                        "tenure_days": e["tenure_days"],
                    },
                    suggested_action_prior=(
                        "flag-for-audit" if max_hours >= severe_single_week_hours else "escalate-to-manager"
                    ),
                )
            )
    return anomalies


def scan(
    employees: list[dict] | None = None,
    department_filter: str | None = None,
    leave_days_threshold: int | None = None,
) -> list[Anomaly]:
    employees = employees if employees is not None else load_employees()
    if department_filter:
        employees = [e for e in employees if e["department"].lower() == department_filter.lower()]

    cohort_stats = compute_cohort_stats(employees)
    anomalies = detect_payroll_outliers(employees, cohort_stats)
    anomalies += detect_leave_abuse(
        employees, quarter_cap_days=leave_days_threshold or 15
    )
    anomalies += detect_compliance_violations(employees)
    return anomalies
