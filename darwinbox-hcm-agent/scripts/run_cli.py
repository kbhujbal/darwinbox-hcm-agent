"""Headless terminal chat loop for the HCM workflow engine.

Usage:
    python scripts/run_cli.py [--thread-id ID] [--employee-id E1001]

Multi-turn state is checkpointed to conversation_state.sqlite keyed by
--thread-id, so re-running with the same --thread-id resumes the same
conversation even after restarting the process.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.build_graph import build_graph, run_turn
from src.observability.tracer import Tracer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default=None, help="Conversation/session id to resume.")
    parser.add_argument("--employee-id", default="E1001", help="Simulated logged-in employee.")
    args = parser.parse_args()

    thread_id = args.thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    tracer = Tracer(run_id=thread_id)
    app = build_graph(tracer)

    print(f"Darwinbox HCM Assistant — thread '{thread_id}', employee '{args.employee_id}'")
    print(f"Trace log: {tracer.path}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        response = run_turn(app, thread_id, args.employee_id, user_text)
        print(f"assistant> {response}\n")


if __name__ == "__main__":
    main()
