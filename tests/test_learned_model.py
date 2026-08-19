"""
Unit tests for learned neural world model and estimation fidelity.
"""
import numpy as np
import pytest
import torch
from src.envs.gridworld_mdp import make_stochastic_choice_gridworld
from src.models.tabular_learned_model import (
    collect_gridworld_experience,
    LearnedWorldModel,
    evaluate_estimation_fidelity,
)


def test_trajectory_collection():
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    dataset = collect_gridworld_experience(grid, num_trajectories=15, max_steps=20, seed=42)
    assert len(dataset) > 50
    assert all(0 <= s < 25 for s in dataset.states)
    assert all(0 <= a < 4 for a in dataset.actions)


def test_learned_world_model_training_and_simplex():
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    dataset = collect_gridworld_experience(grid, num_trajectories=30, max_steps=30, seed=42)

    model = LearnedWorldModel(num_states=25, num_actions=4, gamma=0.95)
    losses = model.fit(dataset, epochs=25, lr=0.01, seed=42)

    assert len(losses) == 25
    assert losses[-1] < losses[0]  # Loss decreased

    p_hat = model.get_transition_matrix()
    assert p_hat.shape == (25, 4, 25)
    assert np.all(p_hat >= 0.0)
    assert np.allclose(np.sum(p_hat, axis=2), 1.0, atol=1e-5)


def test_estimation_fidelity_metrics():
    grid = make_stochastic_choice_gridworld(height=5, width=5, seed=42)
    dataset = collect_gridworld_experience(grid, num_trajectories=40, max_steps=30, seed=42)

    model = LearnedWorldModel(num_states=25, num_actions=4, gamma=0.95)
    model.fit(dataset, epochs=30, lr=0.01, seed=42)

    learned_mdp = model.create_learned_mdp(grid)
    fidelity = evaluate_estimation_fidelity(grid, learned_mdp)

    assert "margin_mae" in fidelity
    assert "crossing_auroc" in fidelity
    assert "fraction_action_agreement" in fidelity
    assert 0.0 <= fidelity["crossing_auroc"] <= 1.0
    assert 0.0 <= fidelity["fraction_action_agreement"] <= 1.0
