"""CLI wrapper: generate the synthetic employee dataset.

Usage:
    python scripts/generate_dataset.py [--count 600] [--seed 42]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.generate import main  # noqa: E402

if __name__ == "__main__":
    main()
