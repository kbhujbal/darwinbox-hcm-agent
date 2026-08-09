"""Pending human-approval queue.

Not modeled as a LangGraph node — a human might respond hours later, and a
LangGraph invoke() is synchronous, so a queued anomaly simply ends that
graph run at a "queued" state. This module is resolved out-of-band, either
by the Streamlit Approvals tab or by src/rl/simulated_reviewer.py, both of
which call reward computation + bandit update + episodic-memory write
directly once a decision is made.

JSON-backed (not SQLite) — the whole queue is small, gets read/rewritten as
a unit, and staying human-readable makes it easy to inspect during the demo.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from src import config

QUEUE_PATH = config.ROOT_DIR / "hitl_queue.json"
DEFAULT_TIMEOUT_SECONDS = 24 * 3600  # 24 hours
SAFE_DEFAULT_ACTION = "flag-for-audit"

_DECISION_TO_STATUS = {"approve": "approved", "reject": "rejected", "modify": "modified"}


@dataclass
class HITLItem:
    item_id: str
    employee_id: str
    anomaly_type: str
    evidence: dict
    proposed_action: str
    confidence: float
    reasoning: str
    context: dict = field(default_factory=dict)  # Anomaly.context, needed to re-derive the bandit's feature vector at decision time
    status: str = "pending"  # pending | approved | rejected | modified | timed_out
    decision: str | None = None  # the resolved action, once decided
    rejection_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None


def _load(path=None) -> list[dict]:
    path = path or QUEUE_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save(items: list[dict], path=None) -> None:
    path = path or QUEUE_PATH
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def enqueue(
    employee_id: str,
    anomaly_type: str,
    evidence: dict,
    proposed_action: str,
    confidence: float,
    reasoning: str,
    context: dict | None = None,
    path=None,
) -> HITLItem:
    item = HITLItem(
        item_id=str(uuid.uuid4())[:8],
        employee_id=employee_id,
        anomaly_type=anomaly_type,
        evidence=evidence,
        proposed_action=proposed_action,
        confidence=confidence,
        reasoning=reasoning,
        context=context or {},
    )
    items = _load(path)
    items.append(asdict(item))
    _save(items, path)
    return item


def resolve_timeouts(timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, path=None) -> list[dict]:
    items = _load(path)
    now = time.time()
    resolved = []
    for item in items:
        if item["status"] == "pending" and (now - item["created_at"]) > timeout_seconds:
            item["status"] = "timed_out"
            item["decision"] = SAFE_DEFAULT_ACTION
            item["decided_at"] = now
            resolved.append(item)
    if resolved:
        _save(items, path)
    return resolved


def list_pending(
    resolve_timeouts_first: bool = True, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, path=None
) -> list[dict]:
    if resolve_timeouts_first:
        resolve_timeouts(timeout_seconds, path)
    return [i for i in _load(path) if i["status"] == "pending"]


def decide(
    item_id: str,
    decision: str,
    modified_action: str | None = None,
    rejection_reason: str | None = None,
    path=None,
) -> dict:
    if decision not in _DECISION_TO_STATUS:
        raise ValueError(f"Unknown decision: {decision!r}")

    items = _load(path)
    for item in items:
        if item["item_id"] == item_id:
            if item["status"] != "pending":
                raise ValueError(f"Item {item_id} already resolved (status={item['status']})")
            item["status"] = _DECISION_TO_STATUS[decision]
            if decision == "approve":
                item["decision"] = item["proposed_action"]
            elif decision == "modify":
                item["decision"] = modified_action
            item["rejection_reason"] = rejection_reason
            item["decided_at"] = time.time()
            _save(items, path)
            return item
    raise KeyError(f"No HITL item with id {item_id!r}")


def get(item_id: str, path=None) -> dict | None:
    return next((i for i in _load(path) if i["item_id"] == item_id), None)


def all_items(path=None) -> list[dict]:
    return _load(path)
