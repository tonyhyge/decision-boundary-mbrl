"""
Unit tests for Matched-Pair Error Construction.
"""
import pytest
import numpy as np
from src.envs.fork_mdp import make_fork_mdp
from src.corruptions.matched_pairs import generate_matched_error_pairs


def test_matched_pair_properties():
    mdp = make_fork_mdp(branch_length=2)
    pairs = generate_matched_error_pairs(mdp, candidate_states=[0])

    assert len(pairs) > 0
    for p in pairs:
        # Total variation error must match between compressive and expansive error
        assert np.isclose(p.compressive_error.error_l1, p.expansive_error.error_l1, atol=1e-4)

        # Unsigned value sensitivity must be approximately equal
        delta_p_comp = p.compressive_error.corrupt_p - p.compressive_error.true_p
        delta_p_exp = p.expansive_error.corrupt_p - p.expansive_error.true_p

        g_comp = np.abs(np.dot(delta_p_comp, p.v_star))
        g_exp = np.abs(np.dot(delta_p_exp, p.v_star))
        assert np.isclose(g_comp, g_exp, atol=1e-4)
