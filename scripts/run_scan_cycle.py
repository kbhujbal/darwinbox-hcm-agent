"""On-demand 'cron' trigger: one full scan cycle -> episodic memory ->
RL bandit -> compliance -> auto-execute or queue for human review.

This represents what a scheduled job would do without needing a real
always-on background scheduler for a local demo (see PART2.md).

Usage:
    python scripts/run_scan_cycle.py [--department Engineering]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.anomaly_agent import run_scan
from src.graph.anomaly_pipeline import process_anomaly
from src.observability.tracer import Tracer
from src.rl.bandit import DEFAULT_STATE_PATH, LinUCBBandit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--department", default=None, help="Restrict the scan to one department.")
    args = parser.parse_args()

    bandit = LinUCBBandit.load(DEFAULT_STATE_PATH)
    tracer = Tracer(run_id="ops")

    anomalies = run_scan(turn_id=1, tracer=tracer, department_filter=args.department)
    scope = f" in {args.department}" if args.department else ""
    print(f"Scanned {len(anomalies)} anomalies{scope}.\n")

    outcomes: dict[str, int] = {}
    for i, anomaly in enumerate(anomalies):
        result = process_anomaly(anomaly, bandit, turn_id=i + 2, tracer=tracer)
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
        print(
            f"  {anomaly.employee_id} ({anomaly.anomaly_type}, conf={anomaly.confidence:.2f}) "
            f"-> {result.rl_action} -> {result.outcome}"
        )

    print("\nSummary:")
    for outcome, count in outcomes.items():
        print(f"  {outcome}: {count}")
    print(f"\nBandit state saved to {DEFAULT_STATE_PATH}")
    print(f"Trace log: {tracer.path}")


if __name__ == "__main__":
    main()
