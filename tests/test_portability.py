"""
Unit tests for World-Model Portability benchmark mechanisms and architectures.
"""
import pytest
import numpy as np
import torch

from src.envs.gridworld_mdp import make_stochastic_choice_gridworld
from src.planning.dp import value_iteration, expected_discounted_return
from src.models.tabular_learned_model import collect_gridworld_experience
from src.models.portability_models import (
    WeightedCategoricalWorldModel,
    ProbabilisticEnsembleWorldModel,
)
from src.correction.portability_weighting import (
    compute_normalized_weights,
    compute_sample_scores,
)


def test_weight_normalization():
    """Verify that sample weights are strictly normalized to mean 1.0."""
    scores = np.array([0.1, 0.5, 0.9, 0.0, 0.2])
    for lam in [0.0, 0.5, 1.0, 2.0, 5.0]:
        w = compute_normalized_weights(scores, lam)
        assert len(w) == len(scores)
        assert np.isclose(np.mean(w), 1.0, atol=1e-10)


def test_shuffled_control_invariants():
    """Verify that shuffled negative control preserves the exact weight multiset."""
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    dataset = collect_gridworld_experience(grid, num_trajectories=10, max_steps=20, seed=42)
    
    # Train dummy initial model
    init_model = WeightedCategoricalWorldModel(25, 4)
    init_model.fit(dataset, epochs=5, seed=42)
    init_mdp = init_model.create_learned_mdp(grid)
    p_mat = init_model.get_transition_matrix()

    rng = np.random.default_rng(123)
    _, est_w = compute_sample_scores(dataset, "estimated_crossing", grid, init_mdp, p_mat, rng=rng, lambda_val=1.5)
    _, shuf_w = compute_sample_scores(dataset, "shuffled_crossing", grid, init_mdp, p_mat, rng=rng, lambda_val=1.5)

    assert len(est_w) == len(shuf_w)
    assert np.isclose(np.mean(est_w), 1.0, atol=1e-10)
    assert np.isclose(np.mean(shuf_w), 1.0, atol=1e-10)
    assert np.isclose(np.std(est_w), np.std(shuf_w), atol=1e-10)
    # Sorted weights should be identical
    assert np.allclose(np.sort(est_w), np.sort(shuf_w), atol=1e-10)


def test_host_a_and_host_b_training():
    """Verify that Host A and Host B fit without error and output valid simplex transition matrices."""
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    dataset = collect_gridworld_experience(grid, num_trajectories=15, max_steps=25, seed=42)

    # Host A
    model_a = WeightedCategoricalWorldModel(25, 4)
    losses_a = model_a.fit(dataset, epochs=10, seed=42)
    assert len(losses_a) == 10
    mdp_a = model_a.create_learned_mdp(grid)
    assert mdp_a.transitions.shape == (25, 4, 25)
    assert np.allclose(np.sum(mdp_a.transitions, axis=-1), 1.0, atol=1e-6)

    # Host B (Ensemble)
    model_b = ProbabilisticEnsembleWorldModel(25, 4, ensemble_size=3)
    losses_b = model_b.fit(dataset, epochs=10, base_seed=42)
    assert len(losses_b) == 3
    mdp_b = model_b.create_learned_mdp(grid)
    assert mdp_b.transitions.shape == (25, 4, 25)
    assert np.allclose(np.sum(mdp_b.transitions, axis=-1), 1.0, atol=1e-6)

    # Plan on learned MDPs
    V_a, Q_a, pi_a = value_iteration(mdp_a)
    V_b, Q_b, pi_b = value_iteration(mdp_b)
    assert len(pi_a) == 25
    assert len(pi_b) == 25

    # Evaluate return in true environment
    ret_a = expected_discounted_return(grid, pi_a)
    ret_b = expected_discounted_return(grid, pi_b)
    assert isinstance(ret_a, float)
    assert isinstance(ret_b, float)
