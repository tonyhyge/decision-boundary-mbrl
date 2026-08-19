"""
Unit tests for Stochastic Choice GridWorld MDP.
"""
import numpy as np
import pytest
from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import (
    value_iteration,
    policy_evaluation,
    compute_occupancy,
    expected_discounted_return,
)
from src.metrics.diagnostics import compute_action_margins


def test_gridworld_creation_and_simplex():
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    assert grid.num_states == 25
    assert grid.num_actions == 4
    assert grid.gamma == 0.95

    # Check simplex validity
    sums = np.sum(grid.transitions, axis=2)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert np.all(grid.transitions >= 0.0)


def test_gridworld_absorbing_goal():
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    goal_s = grid.goal_state
    for a in range(grid.num_actions):
        assert grid.transitions[goal_s, a, goal_s] == 1.0
        assert grid.rewards[goal_s, a] == 0.0


def test_gridworld_dp_and_margins():
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    V_star, Q_star, pi_star = value_iteration(grid)

    assert V_star[grid.goal_state] == 0.0  # Absorbing zero-reward goal
    assert V_star[0] > 0.0  # Start state has positive expected return to goal

    # Check occupancy
    d_s, d_sa = compute_occupancy(grid, pi_star)
    assert np.isclose(np.sum(d_s), 1.0, atol=1e-5)
    assert np.isclose(np.sum(d_sa), 1.0, atol=1e-5)

    # Check margins
    margins = compute_action_margins(Q_star, pi_star)
    assert margins.shape == (25, 4)

    # Verify heterogeneous action gaps
    gaps = []
    for s in range(grid.num_states):
        if s != grid.goal_state:
            opt_a = int(pi_star[s])
            comp_actions = [a for a in range(4) if a != opt_a]
            min_gap = min(margins[s, a] for a in comp_actions)
            gaps.append(min_gap)

    gaps = np.array(gaps)
    # Check that we have both small gap states (near-ties) and large gap states
    assert np.min(gaps) < 0.20
    assert np.max(gaps) > 0.50
