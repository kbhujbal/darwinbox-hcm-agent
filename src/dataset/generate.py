"""Generates a synthetic employee dataset (~600 records by default) with
attendance, payroll, leave, and performance/training data, plus a fixed
percentage of records per category deliberately made anomalous — so
src/anomaly/scoring.py has real signal to find, not just noise.

Ground truth (which employees were deliberately made anomalous, and how) is
written to a *separate* file so detection code and the demo never see the
answer key — only tests/evals read it, to measure precision/recall.

Usage:
    python -m src.dataset.generate --count 600 --seed 42
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics

from src import config
from src.dataset.schema import Employee, LeaveRecord, MonthlyPay, WeeklyOvertime

DEPARTMENTS = [
    "Engineering",
    "Sales",
    "Finance",
    "HR",
    "Operations",
    "Marketing",
    "Customer Support",
]
GRADES = ["L1", "L2", "L3", "L4", "L5"]
GRADE_WEIGHTS = [0.35, 0.30, 0.20, 0.10, 0.05]
BASE_SALARY_BY_GRADE = {"L1": 3500, "L2": 5000, "L3": 7000, "L4": 10000, "L5": 14000}
DEPT_MULTIPLIER = {
    "Engineering": 1.10,
    "Sales": 1.00,
    "Finance": 1.05,
    "HR": 0.95,
    "Operations": 0.95,
    "Marketing": 1.00,
    "Customer Support": 0.90,
}
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
WEEKS = [f"2026-W{w:02d}" for w in range(1, 13)]


def _all_weekdays_in(year: int, month: int) -> list[str]:
    import calendar
    import datetime

    days_in_month = calendar.monthrange(year, month)[1]
    return [
        f"{year}-{month:02d}-{d:02d}"
        for d in range(1, days_in_month + 1)
        if datetime.date(year, month, d).weekday() < 5  # Mon-Fri only
    ]


# Every working day in Q1 2026 — spans all weekdays (not just one, which
# would happen if the pool only sampled dates 7 days apart).
Q1_LEAVE_DATES = [d for m in (1, 2, 3) for d in _all_weekdays_in(2026, m)]


def _weekday_of(date_str: str) -> int:
    import datetime

    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime.date(y, m, d).weekday()


# Baseline (non-abusive) leave-taking draws from the full pool but down-weights
# Mon/Fri so ordinary noise doesn't accidentally look like weekend clustering.
NON_ABUSE_LEAVE_DATE_POOL = Q1_LEAVE_DATES
NON_ABUSE_LEAVE_DATE_WEIGHTS = [0.5 if _weekday_of(d) in (0, 4) else 1.5 for d in Q1_LEAVE_DATES]

FIRST_NAMES = [
    "Aditi", "Ben", "Carlos", "Deepa", "Elena", "Farid", "Grace", "Hiro",
    "Ines", "Jamal", "Kavya", "Liam", "Mei", "Noah", "Olga", "Priya",
    "Quinn", "Ravi", "Sara", "Tom", "Uma", "Victor", "Wanjiru", "Xin",
    "Yusuf", "Zara",
]
LAST_NAMES = [
    "Sharma", "Nguyen", "Garcia", "Kowalski", "Silva", "Khan", "Mueller",
    "Tanaka", "Okoye", "Petrov", "Costa", "Iyer", "Larsen", "Haddad",
    "Fischer", "Reyes", "Suzuki", "Novak", "Osei", "Rossi",
]


def _weekday(date_str: str) -> int:
    y, m, d = (int(x) for x in date_str.split("-"))
    # Zeller-free: use datetime for correctness.
    import datetime

    return datetime.date(y, m, d).weekday()  # 0=Mon .. 6=Sun


def _gen_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _gen_baseline_employee(rng: random.Random, idx: int) -> Employee:
    department = rng.choice(DEPARTMENTS)
    grade = rng.choices(GRADES, weights=GRADE_WEIGHTS, k=1)[0]
    tenure_days = rng.randint(10, 3650)

    base = BASE_SALARY_BY_GRADE[grade] * DEPT_MULTIPLIER[department]
    individual_factor = rng.gauss(1.0, 0.08)
    monthly_pay = [
        MonthlyPay(month=m, gross_pay=round(base * individual_factor * rng.gauss(1.0, 0.03), 2))
        for m in MONTHS
    ]

    weekly_overtime = [
        WeeklyOvertime(week=w, hours=max(0.0, round(rng.gauss(4.0, 2.0), 1))) for w in WEEKS
    ]

    leave_records = []
    for _ in range(rng.randint(2, 5)):
        leave_type = rng.choices(
            ["earned_leave", "casual_leave", "sick_leave"], weights=[0.5, 0.3, 0.2], k=1
        )[0]
        # Weighted away from Mon/Fri: ordinary leave-taking isn't usually
        # opportunistically timed around weekends — that pattern is reserved
        # for the deliberately injected leave_abuse cases below, so it stays
        # a clean signal rather than something baseline noise also produces.
        date = rng.choices(NON_ABUSE_LEAVE_DATE_POOL, weights=NON_ABUSE_LEAVE_DATE_WEIGHTS, k=1)[0]
        notice_days = rng.randint(1, 3) if leave_type == "earned_leave" else rng.randint(0, 2)
        leave_records.append(
            LeaveRecord(leave_type=leave_type, date=date, num_days=rng.choice([1, 1, 2]), notice_days=notice_days)
        )

    training_completed = rng.random() < 0.98
    training_overdue_days = 0 if training_completed else rng.randint(5, 60)

    return Employee(
        employee_id=f"EMP{idx:04d}",
        name=_gen_name(rng),
        department=department,
        grade=grade,
        tenure_days=tenure_days,
        manager_id=None,
        monthly_pay=monthly_pay,
        weekly_overtime=weekly_overtime,
        leave_records=leave_records,
        training_completed=training_completed,
        training_overdue_days=training_overdue_days,
        performance_rating=round(max(1.0, min(5.0, rng.gauss(3.4, 0.7))), 1),
    )


def _assign_managers(employees: list[Employee], rng: random.Random) -> None:
    by_dept: dict[str, list[Employee]] = {}
    for e in employees:
        by_dept.setdefault(e.department, []).append(e)

    for e in employees:
        candidates = [
            m for m in by_dept[e.department] if GRADES.index(m.grade) > GRADES.index(e.grade)
        ]
        e.manager_id = rng.choice(candidates).employee_id if candidates else None


def _cohort_stats(employees: list[Employee]) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(department, grade, month) -> (mean, stdev) of gross_pay."""
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for e in employees:
        for mp in e.monthly_pay:
            buckets.setdefault((e.department, e.grade, mp.month), []).append(mp.gross_pay)
    stats = {}
    for key, values in buckets.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else max(mean * 0.05, 1.0)
        stats[key] = (mean, stdev)
    return stats


def _inject_payroll_outliers(employees: list[Employee], rng: random.Random, ground_truth: dict) -> None:
    stats = _cohort_stats(employees)
    sample = rng.sample(employees, k=max(1, round(len(employees) * 0.05)))
    for e in sample:
        mp = rng.choice(e.monthly_pay)
        mean, stdev = stats[(e.department, e.grade, mp.month)]
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(4.0, 6.0)
        mp.gross_pay = round(mean + direction * magnitude * stdev, 2)
        ground_truth.setdefault(e.employee_id, []).append(
            {"type": "payroll_outlier", "month": mp.month}
        )


def _inject_leave_abuse(employees: list[Employee], rng: random.Random, ground_truth: dict) -> None:
    sample = rng.sample(employees, k=max(1, round(len(employees) * 0.05)))
    for e in sample:
        e.leave_records = [
            r for r in e.leave_records if r.leave_type != "casual_leave"
        ]  # clear noise, replace with the abuse pattern
        weekend_adjacent_dates = [d for d in Q1_LEAVE_DATES if _weekday(d) in (0, 4)]  # Mon/Fri
        chosen = rng.sample(weekend_adjacent_dates, k=min(6, len(weekend_adjacent_dates)))
        for d in chosen:
            e.leave_records.append(
                LeaveRecord(leave_type="casual_leave", date=d, num_days=1, notice_days=0)
            )
        ground_truth.setdefault(e.employee_id, []).append(
            {"type": "leave_abuse", "q1_leave_days": sum(r.num_days for r in e.leave_records)}
        )


def _inject_compliance_violations(employees: list[Employee], rng: random.Random, ground_truth: dict) -> None:
    sample = rng.sample(employees, k=max(1, round(len(employees) * 0.05)))
    for e in sample:
        if rng.random() < 0.5:
            e.training_completed = False
            e.training_overdue_days = rng.randint(10, 90)
            ground_truth.setdefault(e.employee_id, []).append(
                {"type": "compliance_violation", "reason": "training_overdue"}
            )
        else:
            weeks = rng.sample(e.weekly_overtime, k=min(3, len(e.weekly_overtime)))
            for w in weeks:
                w.hours = round(rng.uniform(14, 20), 1)
            ground_truth.setdefault(e.employee_id, []).append(
                {"type": "compliance_violation", "reason": "overtime_cap_breach"}
            )


def generate(count: int = 600, seed: int = 42) -> tuple[list[Employee], dict]:
    rng = random.Random(seed)
    employees = [_gen_baseline_employee(rng, i + 1) for i in range(count)]
    _assign_managers(employees, rng)

    ground_truth: dict[str, list[dict]] = {}
    _inject_payroll_outliers(employees, rng, ground_truth)
    _inject_leave_abuse(employees, rng, ground_truth)
    _inject_compliance_violations(employees, rng, ground_truth)

    return employees, ground_truth


def save(employees: list[Employee], ground_truth: dict, out_dir=None) -> None:
    out_dir = out_dir or config.DATA_DIR
    dataset_path = out_dir / "employees_dataset.json"
    ground_truth_path = out_dir / "employees_ground_truth.json"

    dataset_path.write_text(
        json.dumps([dataclasses.asdict(e) for e in employees], indent=2), encoding="utf-8"
    )
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    print(f"Wrote {len(employees)} employees to {dataset_path}")
    print(f"Wrote ground truth for {len(ground_truth)} anomalous employees to {ground_truth_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    employees, ground_truth = generate(count=args.count, seed=args.seed)
    save(employees, ground_truth)


if __name__ == "__main__":
    main()
