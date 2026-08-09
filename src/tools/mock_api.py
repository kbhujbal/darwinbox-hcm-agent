"""Mock HR backend. Structured JSON in/out, simulated latency, and a
configurable random failure rate so executor.py has something real to
retry against.
"""
from __future__ import annotations

import hashlib
import random
import time

from src import config


class MockAPIError(Exception):
    """Raised to simulate an upstream HR system failure (timeout, 5xx, etc.)."""


MOCK_EMPLOYEES = {
    "E1001": {"name": "Asha Rao", "earned_leave": 12, "casual_leave": 4, "sick_leave": 9, "tenure_days": 900},
    "E1002": {"name": "Marcus Chen", "earned_leave": 6, "casual_leave": 1, "sick_leave": 12, "tenure_days": 1500},
    "E1003": {"name": "Fatima Al-Sayed", "earned_leave": 18, "casual_leave": 6, "sick_leave": 3, "tenure_days": 2000},
    # Still in probation (<90 days) — deliberately, to exercise the
    # probation_no_earned_leave compliance rule in the Action Agent flow.
    "E1004": {"name": "Diego Fernandez", "earned_leave": 2, "casual_leave": 0, "sick_leave": 7, "tenure_days": 45},
}

_DEFAULT_EMPLOYEE = {"name": "Employee", "earned_leave": 10, "casual_leave": 3, "sick_leave": 8, "tenure_days": 400}


def _maybe_fail(op: str) -> None:
    time.sleep(random.uniform(0.05, 0.15))
    if random.random() < config.MOCK_API_FAILURE_RATE:
        raise MockAPIError(f"Upstream HR API timeout while executing '{op}'.")


def _employee(employee_id: str) -> dict:
    return MOCK_EMPLOYEES.get(employee_id, _DEFAULT_EMPLOYEE)


def get_employee(employee_id: str) -> dict:
    """Public accessor — e.g. for the Compliance Agent to read tenure_days
    without dispatching a tool call."""
    return _employee(employee_id)


def check_leave_balance(employee_id: str, leave_type: str) -> dict:
    _maybe_fail("check_leave_balance")
    emp = _employee(employee_id)
    available = emp.get(leave_type, 0)
    return {
        "employee_id": employee_id,
        "employee_name": emp["name"],
        "leave_type": leave_type,
        "available_days": available,
        "as_of": "current cycle",
    }


def apply_leave(
    employee_id: str,
    leave_type: str,
    start_date: str,
    num_days: int,
    reason: str | None = None,
) -> dict:
    _maybe_fail("apply_leave")
    emp = _employee(employee_id)
    available = emp.get(leave_type, 0)

    if num_days > available:
        return {
            "employee_id": employee_id,
            "status": "rejected",
            "message": (
                f"Requested {num_days} day(s) of {leave_type.replace('_', ' ')} but only "
                f"{available} day(s) are available."
            ),
        }

    digest = hashlib.sha256(f"{employee_id}{start_date}{leave_type}".encode()).hexdigest()[:8]
    return {
        "employee_id": employee_id,
        "status": "approved",
        "application_id": f"LV-{digest.upper()}",
        "leave_type": leave_type,
        "start_date": start_date,
        "num_days": num_days,
        "reason": reason,
        "message": f"Leave application submitted and approved for {num_days} day(s) starting {start_date}.",
    }


def get_payslip(employee_id: str, month: str) -> dict:
    _maybe_fail("get_payslip")
    emp = _employee(employee_id)
    seed = int(hashlib.sha256(f"{employee_id}{month}".encode()).hexdigest(), 16)
    rng = random.Random(seed)

    basic = rng.randint(4000, 9000)
    allowances = round(basic * 0.25)
    gross = basic + allowances
    tax = round(gross * 0.12)
    retirement = round(basic * 0.08)
    insurance = 120
    deductions = tax + retirement + insurance
    net = gross - deductions

    return {
        "employee_id": employee_id,
        "employee_name": emp["name"],
        "month": month,
        "breakdown": {
            "basic_salary": basic,
            "allowances": allowances,
            "gross_pay": gross,
            "deductions": {
                "tax_withholding": tax,
                "retirement_fund": retirement,
                "health_insurance": insurance,
                "total": deductions,
            },
            "net_pay": net,
        },
    }


def correct_payroll_discrepancy(
    employee_id: str, month: str, adjustment_amount: float, reason: str
) -> dict:
    _maybe_fail("correct_payroll_discrepancy")
    digest = hashlib.sha256(f"{employee_id}{month}correction".encode()).hexdigest()[:8]
    return {
        "employee_id": employee_id,
        "status": "corrected",
        "correction_id": f"PC-{digest.upper()}",
        "month": month,
        "adjustment_amount": adjustment_amount,
        "message": f"Payroll for {month} adjusted by {adjustment_amount:+.2f} ({reason}).",
    }


def remind_compliance_training(
    employee_id: str, training_name: str = "Mandatory Compliance Training"
) -> dict:
    _maybe_fail("remind_compliance_training")
    return {
        "employee_id": employee_id,
        "status": "reminder_sent",
        "training_name": training_name,
        "message": f"Reminder sent to {employee_id} to complete '{training_name}'.",
    }
