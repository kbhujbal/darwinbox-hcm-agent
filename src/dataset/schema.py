"""Typed shape of a synthetic employee record. Plain dataclasses (not
pydantic) — this is generated once and read many times, no validation
boundary that needs enforcing at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonthlyPay:
    month: str  # YYYY-MM
    gross_pay: float


@dataclass
class WeeklyOvertime:
    week: str  # YYYY-Www
    hours: float


@dataclass
class LeaveRecord:
    leave_type: str  # earned_leave | casual_leave | sick_leave
    date: str  # YYYY-MM-DD
    num_days: int
    notice_days: int


@dataclass
class Employee:
    employee_id: str
    name: str
    department: str
    grade: str  # L1 (junior) .. L5 (senior)
    tenure_days: int
    manager_id: str | None
    monthly_pay: list[MonthlyPay] = field(default_factory=list)
    weekly_overtime: list[WeeklyOvertime] = field(default_factory=list)
    leave_records: list[LeaveRecord] = field(default_factory=list)
    training_completed: bool = True
    training_overdue_days: int = 0
    performance_rating: float = 3.0
