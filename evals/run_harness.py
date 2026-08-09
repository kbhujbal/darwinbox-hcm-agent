"""Executes the 15 evaluation cases (evals/cases.py) and prints/saves a
pass/fail report with reasoning for each.

Usage:
    python evals/run_harness.py

Cases marked requires_api are skipped (not failed) if GEMINI_API_KEY isn't
set, so this always runs offline — re-run with a key set for the full 15/15.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from evals.cases import ALL_CASES


def main() -> None:
    has_api_key = bool(config.GEMINI_API_KEY)
    results = []

    for case in ALL_CASES:
        if case.requires_api and not has_api_key:
            results.append({"id": case.id, "category": case.category, "description": case.description, "status": "SKIPPED", "reasoning": "requires GEMINI_API_KEY, not set"})
            continue
        try:
            result = case.run()
            status = "PASS" if result.passed else "FAIL"
            results.append({"id": case.id, "category": case.category, "description": case.description, "status": status, "reasoning": result.reasoning})
        except Exception as exc:  # noqa: BLE001
            results.append({"id": case.id, "category": case.category, "description": case.description, "status": "ERROR", "reasoning": f"{type(exc).__name__}: {exc}"})

    print(f"{'ID':<5}{'Category':<14}{'Status':<8}Description")
    print("-" * 90)
    for r in results:
        print(f"{r['id']:<5}{r['category']:<14}{r['status']:<8}{r['description']}")
        print(f"      -> {r['reasoning']}")

    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}
    for r in results:
        counts[r["status"]] += 1

    print("\n" + "=" * 40)
    print(f"PASS: {counts['PASS']}  FAIL: {counts['FAIL']}  ERROR: {counts['ERROR']}  SKIPPED: {counts['SKIPPED']}")
    print(f"Total: {len(results)}")

    out_path = config.ROOT_DIR / "traces" / "evaluation_report.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved report to {out_path}")

    if counts["FAIL"] or counts["ERROR"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
