"""
Unit tests for Tabular MDP and Dynamic Programming methods.
"""
import pytest
import numpy as np
from src.envs.tabular_mdp import TabularMDP
from src.envs.fork_mdp import ForkMDP, make_fork_mdp
from src.planning.dp import (
    value_iteration,
    policy_evaluation,
    compute_q_from_v,
    compute_occupancy,
    expected_discounted_return,
)


def test_fork_mdp_creation_and_simplex():
    mdp = make_fork_mdp(branch_length=2, p_left=0.85, p_right=0.75, r_left=1.0, r_right=0.6)
    assert mdp.num_states == 6  # s0, sL1, sL2, sR1, sR2, s_term
    assert mdp.num_actions == 2
    assert np.allclose(mdp.transitions.sum(axis=-1), 1.0)
    assert np.all(mdp.transitions >= 0.0)


def test_value_iteration_fork_mdp():
    mdp = make_fork_mdp(branch_length=2, p_left=0.85, p_right=0.75, r_left=1.0, r_right=0.6, gamma=0.95)
    V_star, Q_star, pi_star = value_iteration(mdp)

    # In standard Fork MDP, left branch has higher return (p=0.85, R=1.0) than right branch (p=0.75, R=0.6)
    assert pi_star[0] == 0  # Action 0 (Left branch) must be optimal at s_0
    assert Q_star[0, 0] > Q_star[0, 1]
    assert np.isclose(V_star[0], Q_star[0, 0])


def test_policy_evaluation_consistency():
    mdp = make_fork_mdp(branch_length=2, gamma=0.95)
    V_star, Q_star, pi_star = value_iteration(mdp)
    
    # Evaluating optimal policy must yield identical values to V_star
    V_pi = policy_evaluation(mdp, pi_star)
    assert np.allclose(V_star, V_pi, atol=1e-6)

    # Suboptimal policy (choosing Right branch at s_0)
    pi_subopt = pi_star.copy()
    pi_subopt[0] = 1
    V_subopt = policy_evaluation(mdp, pi_subopt)
    assert V_subopt[0] < V_star[0]


def test_occupancy_normalization():
    mdp = make_fork_mdp(branch_length=2, gamma=0.95)
    _, _, pi_star = value_iteration(mdp)
    d_s, d_sa = compute_occupancy(mdp, pi_star)

    assert np.isclose(d_s.sum(), 1.0)
    assert np.isclose(d_sa.sum(), 1.0)
    assert np.all(d_s >= -1e-12)
    assert np.all(d_sa >= -1e-12)


def test_expected_return():
    mdp = make_fork_mdp(branch_length=2, gamma=0.95)
    _, _, pi_star = value_iteration(mdp)
    ret_opt = expected_discounted_return(mdp, pi_star)
    
    pi_sub = pi_star.copy()
    pi_sub[0] = 1
    ret_sub = expected_discounted_return(mdp, pi_sub)

    assert ret_opt > ret_sub
