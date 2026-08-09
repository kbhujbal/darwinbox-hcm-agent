import numpy as np
import pytest

from src.rl.bandit import ACTIONS, CONTEXT_DIM, LinUCBBandit


def _context(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=CONTEXT_DIM)


def test_cold_start_scores_are_equal_across_arms():
    bandit = LinUCBBandit()
    ctx = _context()
    scores = bandit.score(ctx)
    values = list(scores.values())
    assert all(abs(v - values[0]) < 1e-9 for v in values)


def test_bandit_learns_to_prefer_repeatedly_rewarded_arm():
    bandit = LinUCBBandit(alpha=0.3)  # lower exploration bonus so learned preference dominates
    ctx = _context(seed=1)

    for _ in range(20):
        bandit.update("auto-correct", ctx, reward=1.0)
        bandit.update("no-action", ctx, reward=-1.0)

    chosen, scores = bandit.select(ctx)
    assert chosen == "auto-correct"
    assert scores["auto-correct"] > scores["no-action"]


def test_rewarded_arm_eventually_beats_cold_untouched_arms():
    # Low alpha so cold-start exploration bonus doesn't dominate once the
    # rewarded arm has enough data — this is the "does it actually learn,
    # not just explore forever" check.
    bandit = LinUCBBandit(alpha=0.05)
    ctx = _context(seed=5)
    for _ in range(50):
        bandit.update("flag-for-audit", ctx, reward=1.0)

    chosen, _ = bandit.select(ctx)
    assert chosen == "flag-for-audit"


def test_bandit_shifts_away_from_punished_arm_over_cycles():
    bandit = LinUCBBandit(alpha=0.3)
    ctx = _context(seed=2)

    _, scores_before = bandit.select(ctx)
    for _ in range(10):
        bandit.update("escalate-to-HR", ctx, reward=-1.0)
    _, scores_after = bandit.select(ctx)

    assert scores_after["escalate-to-HR"] < scores_before["escalate-to-HR"]


def test_save_and_load_roundtrip_preserves_state_exactly(tmp_path):
    bandit = LinUCBBandit(alpha=0.3)
    ctx = _context(seed=3)
    for _ in range(15):
        bandit.update("flag-for-audit", ctx, reward=1.0)

    path = tmp_path / "bandit_state.npz"
    bandit.save(path)
    reloaded = LinUCBBandit.load(path)

    for action in ACTIONS:
        np.testing.assert_allclose(reloaded.A[action], bandit.A[action])
        np.testing.assert_allclose(reloaded.b[action], bandit.b[action])

    # Selection is a pure function of state, so it must match too.
    assert reloaded.select(ctx)[0] == bandit.select(ctx)[0]
    np.testing.assert_allclose(list(reloaded.score(ctx).values()), list(bandit.score(ctx).values()))


def test_load_missing_file_returns_fresh_bandit(tmp_path):
    bandit = LinUCBBandit.load(tmp_path / "does_not_exist.npz")
    assert bandit.actions == ACTIONS


def test_action_distribution_sums_to_one():
    bandit = LinUCBBandit()
    ctx = _context(seed=4)
    bandit.update("auto-correct", ctx, reward=1.0)
    dist = bandit.action_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert set(dist.keys()) == set(ACTIONS)
