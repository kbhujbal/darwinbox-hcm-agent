from src.anomaly import scoring


def _employee(
    employee_id,
    department="Engineering",
    grade="L3",
    tenure_days=500,
    monthly_pay=None,
    weekly_overtime=None,
    leave_records=None,
    training_completed=True,
    training_overdue_days=0,
):
    return {
        "employee_id": employee_id,
        "name": "Test Employee",
        "department": department,
        "grade": grade,
        "tenure_days": tenure_days,
        "manager_id": None,
        "monthly_pay": monthly_pay or [{"month": "2026-01", "gross_pay": 7000.0}],
        "weekly_overtime": weekly_overtime
        or [{"week": f"2026-W{i:02d}", "hours": 4.0} for i in range(1, 13)],
        "leave_records": leave_records or [],
        "training_completed": training_completed,
        "training_overdue_days": training_overdue_days,
        "performance_rating": 3.5,
    }


def _cohort_with_one_outlier():
    baseline = [
        _employee(f"EMP{i}", monthly_pay=[{"month": "2026-01", "gross_pay": 7000.0 + i * 20}])
        for i in range(10)
    ]
    outlier = _employee("EMP_OUTLIER", monthly_pay=[{"month": "2026-01", "gross_pay": 20000.0}])
    return baseline + [outlier]


def test_detects_payroll_outlier_and_not_baseline():
    employees = _cohort_with_one_outlier()
    anomalies = scoring.detect_payroll_outliers(employees)
    flagged_ids = {a.employee_id for a in anomalies}

    assert "EMP_OUTLIER" in flagged_ids
    assert "EMP0" not in flagged_ids  # a baseline employee stays unflagged


def test_payroll_outlier_confidence_and_context_shape():
    employees = _cohort_with_one_outlier()
    anomalies = scoring.detect_payroll_outliers(employees)
    outlier = next(a for a in anomalies if a.employee_id == "EMP_OUTLIER")

    assert 0.0 < outlier.confidence <= 1.0
    assert "correction_pct" in outlier.context
    assert "confidence" in outlier.context


def test_detects_leave_abuse_weekend_clustering():
    # Mondays/Fridays in Jan 2026: 2, 5, 9, 12, 16, 19, 23, 26, 30
    abusive_dates = ["2026-01-02", "2026-01-05", "2026-01-09", "2026-01-12", "2026-01-16"]
    records = [
        {"leave_type": "casual_leave", "date": d, "num_days": 1, "notice_days": 0}
        for d in abusive_dates
    ]
    employees = [_employee("EMP_ABUSER", leave_records=records)]
    anomalies = scoring.detect_leave_abuse(employees)

    assert len(anomalies) == 1
    assert anomalies[0].employee_id == "EMP_ABUSER"


def test_normal_leave_pattern_not_flagged():
    # Spread across the quarter, well under the day cap, not weekend-clustered.
    records = [
        {"leave_type": "earned_leave", "date": "2026-02-11", "num_days": 2, "notice_days": 3},
        {"leave_type": "sick_leave", "date": "2026-03-18", "num_days": 1, "notice_days": 0},
    ]
    employees = [_employee("EMP_NORMAL", leave_records=records)]
    assert scoring.detect_leave_abuse(employees) == []


def test_detects_training_overdue():
    employees = [_employee("EMP_LATE", training_completed=False, training_overdue_days=45)]
    anomalies = scoring.detect_compliance_violations(employees)

    assert any(a.employee_id == "EMP_LATE" and a.evidence["reason"] == "training_overdue" for a in anomalies)


def test_detects_persistent_overtime_breach():
    weekly_overtime = [{"week": f"2026-W{i:02d}", "hours": 4.0} for i in range(1, 13)]
    weekly_overtime[0]["hours"] = 15.0
    weekly_overtime[1]["hours"] = 16.0
    employees = [_employee("EMP_OVERWORKED", weekly_overtime=weekly_overtime)]
    anomalies = scoring.detect_compliance_violations(employees)

    assert any(
        a.employee_id == "EMP_OVERWORKED" and a.evidence["reason"] == "overtime_cap_breach"
        for a in anomalies
    )


def test_single_overtime_spike_is_not_persistent_enough_to_flag():
    weekly_overtime = [{"week": f"2026-W{i:02d}", "hours": 4.0} for i in range(1, 13)]
    weekly_overtime[0]["hours"] = 15.0  # only one week over cap
    employees = [_employee("EMP_ONE_SPIKE", weekly_overtime=weekly_overtime)]
    anomalies = scoring.detect_compliance_violations(employees)

    assert not any(a.employee_id == "EMP_ONE_SPIKE" for a in anomalies)


def test_scan_department_filter():
    eng = _employee("EMP_ENG", department="Engineering", training_completed=False, training_overdue_days=20)
    sales = _employee("EMP_SALES", department="Sales", training_completed=False, training_overdue_days=20)
    employees = [eng, sales]

    result = scoring.scan(employees, department_filter="Engineering")
    flagged_ids = {a.employee_id for a in result}

    assert "EMP_ENG" in flagged_ids
    assert "EMP_SALES" not in flagged_ids


def test_anomaly_description_is_readable_string():
    employees = [_employee("EMP_LATE", training_completed=False, training_overdue_days=45)]
    anomaly = scoring.detect_compliance_violations(employees)[0]
    assert "EMP_LATE" in anomaly.description()
    assert "compliance_violation" in anomaly.description()
