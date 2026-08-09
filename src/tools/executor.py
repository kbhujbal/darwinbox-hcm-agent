"""Tool dispatch with exponential-backoff retry and a structured fallback
response when the mock HR API keeps failing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src import config
from src.tools import mock_api

TOOL_FUNCTIONS = {
    "check_leave_balance": mock_api.check_leave_balance,
    "apply_leave": mock_api.apply_leave,
    "get_payslip": mock_api.get_payslip,
    "correct_payroll_discrepancy": mock_api.correct_payroll_discrepancy,
    "remind_compliance_training": mock_api.remind_compliance_training,
}


@dataclass
class ToolCallResult:
    tool_name: str
    arguments: dict
    output: dict
    success: bool
    attempts: int
    latency_ms: float


def execute_tool(tool_name: str, arguments: dict) -> ToolCallResult:
    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    fn = TOOL_FUNCTIONS[tool_name]
    start = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, config.TOOL_MAX_RETRIES + 1):
        try:
            output = fn(**arguments)
            latency_ms = (time.perf_counter() - start) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                output=output,
                success=True,
                attempts=attempt,
                latency_ms=latency_ms,
            )
        except mock_api.MockAPIError as exc:
            last_error = exc
            if attempt < config.TOOL_MAX_RETRIES:
                time.sleep(config.TOOL_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    latency_ms = (time.perf_counter() - start) * 1000
    fallback_output = {
        "status": "error",
        "message": (
            f"The HR system is temporarily unavailable ({last_error}). "
            "Please try again in a few minutes or contact HR support directly."
        ),
    }
    return ToolCallResult(
        tool_name=tool_name,
        arguments=arguments,
        output=fallback_output,
        success=False,
        attempts=config.TOOL_MAX_RETRIES,
        latency_ms=latency_ms,
    )
