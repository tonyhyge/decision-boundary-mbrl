"""
Unit tests for Corruptions and Counterfactual Restoration.
"""
import pytest
import numpy as np
from src.envs.fork_mdp import make_fork_mdp
from src.corruptions.injector import LocalizedError, CorruptedMDP, inject_random_corruptions


def test_corrupted_mdp_restoration():
    mdp = make_fork_mdp(branch_length=2)
    corrupted_mdp = inject_random_corruptions(mdp, num_corruptions=3, noise_scale=0.4, rng=np.random.default_rng(123))

    assert len(corrupted_mdp.errors) == 3

    # Check that restoring a single component works as expected
    err_0 = corrupted_mdp.errors[0]
    restored_mdp = corrupted_mdp.restore_component(0)
    
    # Restored transition at (s, a) should match true_p exactly
    assert np.allclose(restored_mdp.transitions[err_0.state, err_0.action, :], err_0.true_p)

    # Restoring all errors should yield true transitions
    restored_all = corrupted_mdp.restore_subset(list(range(len(corrupted_mdp.errors))))
    assert np.allclose(restored_all.transitions, mdp.transitions)


def test_counterfactual_correction_value_non_negative():
    mdp = make_fork_mdp(branch_length=2)
    # Inject a corruption that flips the decision at s_0
    corrupt_transitions = mdp.transitions.copy()
    # Heavily degrade left branch at s_0, a_0
    corrupt_transitions[0, 0, 1] = 0.1
    corrupt_transitions[0, 0, -1] = 0.9

    err = LocalizedError(
        error_id=0,
        state=0,
        action=0,
        true_p=mdp.transitions[0, 0, :],
        corrupt_p=corrupt_transitions[0, 0, :],
        error_l1=0.5 * float(np.sum(np.abs(corrupt_transitions[0, 0, :] - mdp.transitions[0, 0, :]))),
        error_kl=1.0,
        error_mse=0.1,
    )

    corrupted_mdp = CorruptedMDP(mdp, [err])
    c_0, j_corr, _ = corrupted_mdp.compute_counterfactual_correction_value(0)
    
    # Correcting the decision flip error should yield strictly positive correction value
    assert c_0 > 0.0
    assert j_corr > corrupted_mdp.j_corrupt
