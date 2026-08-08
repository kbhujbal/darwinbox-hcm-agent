"""Structured per-step trace logging: one JSON line per graph-node step,
written to traces/{run_id}.jsonl.

The schema is deliberately additive — Part 1 populates agent/tool/latency/
token/cost fields; a future Part 2 can add rl_action_selected, reward, and
compliance_veto fields to the same record without migrating old traces.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import config


@dataclass
class TraceStep:
    run_id: str
    turn_id: int
    agent_name: str
    input: dict
    output: dict
    tool_calls: list = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    timestamp: float = field(default_factory=time.time)


class Tracer:
    def __init__(self, run_id: str | None = None, traces_dir: str | Path | None = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.traces_dir = Path(traces_dir or config.TRACES_DIR)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.traces_dir / f"{self.run_id}.jsonl"

    def log_step(
        self,
        turn_id: int,
        agent_name: str,
        input: dict,
        output: dict,
        tool_calls: list | None = None,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        model: str | None = None,
    ) -> TraceStep:
        step = TraceStep(
            run_id=self.run_id,
            turn_id=turn_id,
            agent_name=agent_name,
            input=input,
            output=output,
            tool_calls=tool_calls or [],
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            model=model,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(step), default=str) + "\n")
        return step

    @staticmethod
    def read_run(run_id: str, traces_dir: str | Path | None = None) -> list[dict]:
        traces_dir = Path(traces_dir or config.TRACES_DIR)
        path = traces_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    @staticmethod
    def list_runs(traces_dir: str | Path | None = None) -> list[str]:
        traces_dir = Path(traces_dir or config.TRACES_DIR)
        if not traces_dir.exists():
            return []
        return sorted((p.stem for p in traces_dir.glob("*.jsonl")), reverse=True)
