"""Shared LangGraph state. Every node reads/writes this dict — there is no
direct agent-to-agent calling in this codebase, only state passed through
the graph, which is the same hard requirement Part 2 imposes on the
Supervisor/sub-agent architecture.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    agent: Optional[str]


class PendingAction(TypedDict, total=False):
    tool_name: str
    arguments: dict
    missing_fields: list[str]


def _append_history(existing: list[Turn], new: list[Turn]) -> list[Turn]:
    return existing + new


class HCMState(TypedDict):
    thread_id: str
    employee_id: str
    turn_id: int
    trace_run_id: str

    user_input: str
    history: Annotated[list[Turn], _append_history]

    route: Optional[Literal["policy", "action", "clarify"]]
    pending_action: Optional[PendingAction]

    final_response: Optional[str]
