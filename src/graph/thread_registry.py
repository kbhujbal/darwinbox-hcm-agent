"""Tracks which conversation thread is "active" for each mock employee, and
the list of past threads they can resume — so switching the employee in the
UI switches to *that employee's* conversation (not whichever thread the
browser tab happened to start), and "start new conversation" doesn't throw
the old one away.

Persisted to a small JSON file rather than kept in Streamlit session state,
so it survives app restarts, consistent with how conversation state itself
is checkpointed to SQLite rather than held in memory.
"""
from __future__ import annotations

import json
import time
import uuid

from src import config

REGISTRY_PATH = config.ROOT_DIR / "thread_registry.json"


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save(data: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _employee_entry(data: dict, employee_id: str) -> dict:
    return data.setdefault(employee_id, {"active_thread_id": None, "threads": []})


def start_new_thread(employee_id: str) -> str:
    data = _load()
    entry = _employee_entry(data, employee_id)
    thread_id = f"employee-{employee_id}-{uuid.uuid4().hex[:8]}"
    entry["threads"].insert(0, {"thread_id": thread_id, "created_at": time.time()})
    entry["active_thread_id"] = thread_id
    _save(data)
    return thread_id


def get_active_thread(employee_id: str) -> str:
    data = _load()
    entry = data.get(employee_id)
    if entry and entry.get("active_thread_id"):
        return entry["active_thread_id"]
    return start_new_thread(employee_id)


def set_active_thread(employee_id: str, thread_id: str) -> None:
    data = _load()
    entry = _employee_entry(data, employee_id)
    entry["active_thread_id"] = thread_id
    _save(data)


def list_threads(employee_id: str) -> list[dict]:
    data = _load()
    return data.get(employee_id, {}).get("threads", [])
