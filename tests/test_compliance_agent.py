from src.graph import compliance_agent


def test_loads_between_10_and_15_rules():
    rules = compliance_agent.load_rules()
    assert 10 <= len(rules) <= 15


def test_no_violation_when_within_limits():
    result = compliance_agent.evaluate({"overtime_hours_week": 8})
    assert result.vetoed is False
    assert result.rules_checked == 1


def test_vetoes_overtime_over_cap():
    result = compliance_agent.evaluate({"overtime_hours_week": 15})
    assert result.vetoed is True
    assert any(v.rule_id == "max_overtime_per_week" for v in result.violations)


def test_condition_field_scopes_rule_to_matching_leave_type():
    # earned_leave_notice_period only applies when leave_type == earned_leave
    ok = compliance_agent.evaluate({"notice_days": 0, "leave_type": "casual_leave"})
    assert not any(v.rule_id == "earned_leave_notice_period" for v in ok.violations)

    violated = compliance_agent.evaluate({"notice_days": 0, "leave_type": "earned_leave"})
    assert any(v.rule_id == "earned_leave_notice_period" for v in violated.violations)


def test_probation_blocks_earned_leave():
    result = compliance_agent.evaluate({"tenure_days": 30, "leave_type": "earned_leave"})
    assert result.vetoed is True
    assert any(v.rule_id == "probation_no_earned_leave" for v in result.violations)


def test_applies_to_actions_scopes_rule_to_matching_action():
    context = {"correction_pct": 25}
    vetoed_for_auto_correct = compliance_agent.evaluate(context, proposed_action="auto-correct")
    not_applicable_for_escalate = compliance_agent.evaluate(context, proposed_action="escalate-to-manager")

    assert vetoed_for_auto_correct.vetoed is True
    assert not_applicable_for_escalate.vetoed is False


def test_low_confidence_auto_correct_is_vetoed():
    result = compliance_agent.evaluate({"confidence": 0.4}, proposed_action="auto-correct")
    assert result.vetoed is True
    assert any(v.rule_id == "payroll_correction_requires_confidence" for v in result.violations)


def test_missing_fields_are_skipped_not_violated():
    # No relevant fields in context at all -> nothing to check, nothing vetoed.
    result = compliance_agent.evaluate({"unrelated_field": 1})
    assert result.vetoed is False
    assert result.rules_checked == 0


def test_multiple_violations_all_reported():
    result = compliance_agent.evaluate(
        {"overtime_hours_week": 25, "tenure_days": 10, "leave_type": "earned_leave"}
    )
    assert result.vetoed is True
    assert len(result.violations) >= 2
