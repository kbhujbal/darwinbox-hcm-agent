"""LinUCB contextual bandit for action-selection.

Why LinUCB (see PART2.md for the full justification): the decision is
single-shot — one action per anomaly, not a multi-step sequence — so there's
no credit-assignment problem PPO/REINFORCE is built for. LinUCB's per-arm
update is closed-form linear algebra (no gradient descent, no training
instability), its state is two small matrices per arm (trivial to persist),
and its learned weights (`theta`) can be inspected directly to explain why
behavior shifted.

Standard LinUCB (Li et al., 2010): for each arm a, maintain A_a (d x d,
init identity) and b_a (d, init zero). Predicted value for context x:
    theta_a = A_a^-1 b_a
    p_a = theta_a . x + alpha * sqrt(x^T A_a^-1 x)
Pick argmax p_a. After observing reward r for the chosen arm:
    A_a += x x^T ; b_a += r * x
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src import config

ACTIONS = ["auto-correct", "escalate-to-manager", "escalate-to-HR", "flag-for-audit", "no-action"]

# Context vector layout — hand-designed, numeric, explainable (see
# src/rl/features.py for the encoder). Kept separate from the semantic
# embedding used for episodic-memory retrieval.
CONTEXT_DIM = 9

DEFAULT_STATE_PATH = config.ROOT_DIR / "rl_state" / "bandit_state.npz"


class LinUCBBandit:
    def __init__(self, actions: list[str] = ACTIONS, dim: int = CONTEXT_DIM, alpha: float = 1.0):
        self.actions = actions
        self.dim = dim
        self.alpha = alpha
        self.A = {a: np.identity(dim) for a in actions}
        self.b = {a: np.zeros(dim) for a in actions}

    def _theta(self, action: str) -> np.ndarray:
        return np.linalg.solve(self.A[action], self.b[action])

    def score(self, context: np.ndarray) -> dict[str, float]:
        scores = {}
        for a in self.actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta @ context)
            bonus = self.alpha * float(np.sqrt(context @ A_inv @ context))
            scores[a] = mean + bonus
        return scores

    def select(self, context: np.ndarray) -> tuple[str, dict[str, float]]:
        scores = self.score(context)
        best = max(scores, key=scores.get)
        return best, scores

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

    def action_distribution(self) -> dict[str, float]:
        """A rough 'how much has this arm learned' signal: ||theta|| per
        arm normalized to sum to 1 — used for the RL diagnostics plot to
        show the policy's preference shifting across feedback cycles."""
        norms = {a: float(np.linalg.norm(self._theta(a))) for a in self.actions}
        total = sum(norms.values()) or 1.0
        return {a: v / total for a, v in norms.items()}

    def save(self, path: Path | str = DEFAULT_STATE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for a in self.actions:
            payload[f"A::{a}"] = self.A[a]
            payload[f"b::{a}"] = self.b[a]
        np.savez(path, actions=np.array(self.actions), dim=self.dim, alpha=self.alpha, **payload)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_STATE_PATH) -> "LinUCBBandit":
        path = Path(path)
        if not path.exists():
            return cls()

        data = np.load(path, allow_pickle=False)
        actions = list(data["actions"])
        bandit = cls(actions=actions, dim=int(data["dim"]), alpha=float(data["alpha"]))
        for a in actions:
            bandit.A[a] = data[f"A::{a}"]
            bandit.b[a] = data[f"b::{a}"]
        return bandit
