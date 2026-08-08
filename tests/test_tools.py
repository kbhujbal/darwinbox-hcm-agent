import pytest

from src import config
from src.tools import executor, mock_api


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(mock_api.time, "sleep", lambda *_: None)
    monkeypatch.setattr(executor.time, "sleep", lambda *_: None)


def test_succeeds_on_first_try_when_no_failures(monkeypatch):
    monkeypatch.setattr(config, "MOCK_API_FAILURE_RATE", 0.0)
    result = executor.execute_tool(
        "check_leave_balance", {"employee_id": "E1001", "leave_type": "earned_leave"}
    )
    assert result.success is True
    assert result.attempts == 1
    assert result.output["available_days"] == 12


def test_falls_back_gracefully_after_max_retries(monkeypatch):
    monkeypatch.setattr(config, "MOCK_API_FAILURE_RATE", 1.0)
    result = executor.execute_tool(
        "check_leave_balance", {"employee_id": "E1001", "leave_type": "earned_leave"}
    )
    assert result.success is False
    assert result.attempts == config.TOOL_MAX_RETRIES
    assert result.output["status"] == "error"
    assert "temporarily unavailable" in result.output["message"]


def test_unknown_tool_raises():
    with pytest.raises(ValueError):
        executor.execute_tool("delete_employee", {})


def test_apply_leave_rejects_when_over_balance(monkeypatch):
    monkeypatch.setattr(config, "MOCK_API_FAILURE_RATE", 0.0)
    result = executor.execute_tool(
        "apply_leave",
        {
            "employee_id": "E1004",  # only 2 earned_leave days available
            "leave_type": "earned_leave",
            "start_date": "2026-06-15",
            "num_days": 5,
        },
    )
    assert result.success is True  # the tool call itself succeeded
    assert result.output["status"] == "rejected"


def test_apply_leave_approves_within_balance(monkeypatch):
    monkeypatch.setattr(config, "MOCK_API_FAILURE_RATE", 0.0)
    result = executor.execute_tool(
        "apply_leave",
        {
            "employee_id": "E1001",
            "leave_type": "earned_leave",
            "start_date": "2026-06-15",
            "num_days": 3,
        },
    )
    assert result.output["status"] == "approved"
    assert result.output["application_id"].startswith("LV-")


def test_get_payslip_is_deterministic_per_employee_and_month(monkeypatch):
    monkeypatch.setattr(config, "MOCK_API_FAILURE_RATE", 0.0)
    r1 = executor.execute_tool("get_payslip", {"employee_id": "E1002", "month": "2026-06"})
    r2 = executor.execute_tool("get_payslip", {"employee_id": "E1002", "month": "2026-06"})
    assert r1.output["breakdown"] == r2.output["breakdown"]
