"""Compliance Agent: evaluates a proposed action's context against
data/compliance_rules.yaml and issues a hard veto for violations — even if
the Supervisor or RL policy recommended the action. Rules are structured
predicates loaded from YAML, never embedded in a prompt.

A rule only applies if its `field` is present in the context being
evaluated (so the same ruleset covers leave applications, payroll
corrections, and anomaly auto-corrections without separate rule sets), and
if any `condition_field`/`condition_value` / `applies_to_actions` filters
match.
"""
from __future__ import annotations

import operator as op
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from src import config
from src.observability.tracer import Tracer

_OPERATORS = {
    "<=": op.le,
    ">=": op.ge,
    "<": op.lt,
    ">": op.gt,
    "==": op.eq,
    "!=": op.ne,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}

RULES_PATH = config.DATA_DIR / "compliance_rules.yaml"


@dataclass
class RuleViolation:
    rule_id: str
    description: str
    field: str
    expected: str
    actual: Any


@dataclass
class ComplianceResult:
    vetoed: bool
    violations: list[RuleViolation] = field(default_factory=list)
    rules_checked: int = 0


_rules_cache: list[dict] | None = None


def load_rules(path=None, force_reload: bool = False) -> list[dict]:
    global _rules_cache
    if _rules_cache is None or force_reload:
        path = path or RULES_PATH
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _rules_cache = data["rules"]
    return _rules_cache


def evaluate(
    context: dict,
    proposed_action: str | None = None,
    rules: list[dict] | None = None,
) -> ComplianceResult:
    rules = rules if rules is not None else load_rules()
    violations: list[RuleViolation] = []
    checked = 0

    for rule in rules:
        field_name = rule["field"]
        if field_name not in context:
            continue

        applies_to = rule.get("applies_to_actions")
        if applies_to and proposed_action not in applies_to:
            continue

        cond_field = rule.get("condition_field")
        if cond_field is not None and context.get(cond_field) != rule.get("condition_value"):
            continue

        checked += 1
        comparator = _OPERATORS[rule["operator"]]
        actual = context[field_name]
        expected = rule["value"]
        if not comparator(actual, expected):
            violations.append(
                RuleViolation(
                    rule_id=rule["id"],
                    description=rule["description"],
                    field=field_name,
                    expected=f"{rule['operator']} {expected}",
                    actual=actual,
                )
            )

    return ComplianceResult(vetoed=bool(violations), violations=violations, rules_checked=checked)


def check_compliance_step(
    context: dict,
    proposed_action: str,
    turn_id: int,
    tracer: Tracer,
    agent_name: str = "compliance_agent",
) -> ComplianceResult:
    """Same as evaluate(), but also logs a trace step — the entry point
    graph nodes should call."""
    start = time.perf_counter()
    result = evaluate(context, proposed_action)
    latency_ms = (time.perf_counter() - start) * 1000

    tracer.log_step(
        turn_id=turn_id,
        agent_name=agent_name,
        input={"proposed_action": proposed_action, "context": context},
        output={
            "vetoed": result.vetoed,
            "rules_checked": result.rules_checked,
            "violations": [v.__dict__ for v in result.violations],
        },
        latency_ms=latency_ms,
        compliance_veto=result.vetoed,
    )
    return result
