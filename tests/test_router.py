from src.graph import orchestrator


def test_action_requests_route_without_llm_call():
    action_examples = [
        "Apply for 3 days leave starting June 15",
        "Check my leave balance",
        "Can I get my payslip for June?",
        "What's my salary slip look like this month?",
        "How many sick leave days do I have left?",
    ]
    for text in action_examples:
        assert orchestrator._regex_route(text) == "action", text


def test_policy_requests_route_without_llm_call():
    policy_examples = [
        "What is our maternity leave policy?",
        "What is the notice period for resignation?",
        "How many days of earned leave am I entitled to?",
        "What's the probation period for new hires?",
        "Explain the overtime policy.",
    ]
    for text in policy_examples:
        assert orchestrator._regex_route(text) == "policy", text


def test_anomaly_query_requests_route_without_llm_call():
    examples = [
        "Flag anyone in Engineering who has taken more than 15 days leave in Q1",
        "Show me payroll outliers this quarter",
        "List employees missing mandatory training",
        "Are there any overtime cap breaches this month?",
    ]
    for text in examples:
        assert orchestrator._regex_route(text) == "anomaly_query", text


def test_ambiguous_request_has_no_regex_match():
    assert orchestrator._regex_route("hey, can you help me with something?") is None


def test_pending_action_short_circuits_to_action(monkeypatch):
    class DummyTracer:
        def log_step(self, **kwargs):
            return None

    state = {
        "user_input": "sick leave",
        "turn_id": 2,
        "pending_action": {"tool_name": "check_leave_balance", "arguments": {}},
    }
    result = orchestrator.route_request(state, DummyTracer())
    assert result["route"] == "action"
