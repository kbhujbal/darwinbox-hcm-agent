"""Runs N simulated feedback cycles end-to-end: scan -> episodic memory ->
RL propose -> compliance check -> (auto-execute | queue) -> simulated
reviewer resolves the queue -> reward -> bandit update. This is the
required "run at least 2 feedback cycles, show action proposals measurably
change" demonstration, plus the RL diagnostics (cumulative reward curve,
action-distribution shift) needed for the Loom walkthrough.

Starts from a FRESH bandit each run (not the persisted one) so the
before/after story is clean and reproducible; final state is still saved to
disk at the end, and rl_state/bandit_state.npz persisting across separate
process runs is verified independently (see tests/test_bandit.py and
PART2.md's manual restart check).

Usage:
    python scripts/run_feedback_cycles.py --cycles 3 --department Engineering

Requires GEMINI_API_KEY — makes real (small) embedding calls for episodic
memory. Restricting --department keeps the anomaly count (and API usage)
manageable; omit it to scan the full dataset.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.anomaly import scoring
from src.anomaly.models import Anomaly
from src.graph.anomaly_pipeline import process_anomaly
from src.hitl import queue as hitl_queue
from src.observability.tracer import Tracer
from src.rl import episodic_memory
from src.rl.bandit import DEFAULT_STATE_PATH, LinUCBBandit
from src.rl.resolution import resolve_decision
from src.rl.simulated_reviewer import simulate_decision


def run_cycle(cycle_num: int, bandit: LinUCBBandit, tracer: Tracer, rng: random.Random, department_filter=None):
    anomalies = scoring.scan(department_filter=department_filter)
    action_counts = Counter()
    total_reward = 0.0
    proposals = {}

    for i, anomaly in enumerate(anomalies):
        result = process_anomaly(
            anomaly, bandit, turn_id=i + 1, tracer=tracer, signal_type="scheduled_scan", use_memory=True
        )
        action_counts[result.rl_action] += 1
        proposals[(anomaly.employee_id, anomaly.anomaly_type)] = (result.rl_action, anomaly.confidence)
        if result.reward is not None:
            total_reward += result.reward

    pending = hitl_queue.list_pending()
    for item in pending:
        anomaly = Anomaly(
            employee_id=item["employee_id"],
            anomaly_type=item["anomaly_type"],
            confidence=item["confidence"],
            evidence=item["evidence"],
            context=item["context"],
        )
        decision = simulate_decision(anomaly, item["proposed_action"], rng)
        result = resolve_decision(
            item["item_id"],
            decision.decision,
            bandit,
            modified_action=decision.modified_action,
            rejection_reason=decision.reason,
        )
        total_reward += result.reward

    return {
        "cycle": cycle_num,
        "anomaly_count": len(anomalies),
        "action_counts": dict(action_counts),
        "total_reward": total_reward,
        "proposals": proposals,
    }


def print_before_after(cycle_results: list[dict]) -> None:
    first, last = cycle_results[0], cycle_results[-1]
    shared = set(first["proposals"]) & set(last["proposals"])
    print(f"\nBefore/after comparison ({len(shared)} anomalies seen in both cycle 1 and cycle {last['cycle']}):")
    for key in list(shared)[:10]:
        before_action, before_conf = first["proposals"][key]
        after_action, after_conf = last["proposals"][key]
        marker = " <- action changed" if before_action != after_action else ""
        print(
            f"  {key[0]} ({key[1]}): {before_action} (conf {before_conf:.2f}) -> "
            f"{after_action} (conf {after_conf:.2f}){marker}"
        )


def plot_diagnostics(cycle_results: list[dict]) -> Path:
    cycles = [r["cycle"] for r in cycle_results]
    cumulative, running = [], 0.0
    for r in cycle_results:
        running += r["total_reward"]
        cumulative.append(running)

    per_cycle_reward = [r["total_reward"] for r in cycle_results]
    all_actions = sorted({a for r in cycle_results for a in r["action_counts"]})
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5))

    # Per-cycle reward is the more legible "is it learning" signal when
    # per-anomaly rewards are mostly negative (a fairly strict simulated
    # reviewer): a monotonically-summed cumulative total will trend down
    # regardless of whether each cycle is actually improving on the last.
    ax0.plot(cycles, per_cycle_reward, marker="o", color="tab:green")
    ax0.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax0.set_title("Total reward per cycle")
    ax0.set_xlabel("Cycle")
    ax0.set_ylabel("Reward (higher = better)")
    ax0.grid(alpha=0.3)

    ax1.plot(cycles, cumulative, marker="o")
    ax1.set_title("Cumulative reward across feedback cycles")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Cumulative reward")
    ax1.grid(alpha=0.3)

    bottom = [0] * len(cycle_results)
    for action in all_actions:
        values = [r["action_counts"].get(action, 0) for r in cycle_results]
        ax2.bar(cycles, values, bottom=bottom, label=action)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax2.set_title("Action distribution shift across cycles")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path = config.ROOT_DIR / "traces" / "rl_diagnostics.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--department", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="LinUCB exploration constant. Lower values exploit learned preferences sooner; "
        "higher values keep exploring longer. Default matches the bandit's own default.",
    )
    parser.add_argument(
        "--reset-memory", action="store_true",
        help="Clear the episodic-memory collection before starting, so this run's before/after "
        "story isn't influenced by incidents recorded by a previous run.",
    )
    args = parser.parse_args()

    if args.reset_memory:
        episodic_memory.reset_memory()
        print("Cleared episodic memory collection.\n")

    rng = random.Random(args.seed)
    bandit = LinUCBBandit(alpha=args.alpha)
    tracer = Tracer(run_id="feedback-cycles")

    cycle_results = []
    for c in range(1, args.cycles + 1):
        print(f"--- Cycle {c} ---")
        result = run_cycle(c, bandit, tracer, rng, department_filter=args.department)
        cycle_results.append(result)
        print(
            f"  {result['anomaly_count']} anomalies · actions {result['action_counts']} "
            f"· cycle reward {result['total_reward']:.2f}"
        )

    print_before_after(cycle_results)

    plot_path = plot_diagnostics(cycle_results)
    bandit.save(DEFAULT_STATE_PATH)

    print(f"\nSaved RL diagnostics plot to {plot_path}")
    print(f"Bandit state persisted to {DEFAULT_STATE_PATH}")
    print(f"Trace log: {tracer.path}")


if __name__ == "__main__":
    main()
