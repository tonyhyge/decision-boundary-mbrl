"""
Unit tests for CartPole Continuous Dynamics & Competent Value Function.
"""
import numpy as np
import pytest
from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction, CartPoleNeuralDynamics


def test_cartpole_step_physics():
    dyn = CartPoleDynamics()
    s0 = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    s1, r, done = dyn.step(s0, action=1)
    assert s1.shape == (4,)
    assert not done
    assert r == 1.0
    # Force push should accelerate cart
    assert s1[1] > 0.0


def test_competent_value_function_margin():
    val_fn = CompetentCartPoleValueFunction()
    # At exact equilibrium center [0,0,0,0], symmetry makes actions equal margin
    s_eq = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    m_eq = val_fn.compute_action_margin(s_eq)
    assert m_eq >= 0.0

    # At tilted angle, one action is clearly preferred
    s_tilt = np.array([0.0, 0.0, 0.10, 0.20], dtype=np.float64)
    m_tilt = val_fn.compute_action_margin(s_tilt)
    opt_a = val_fn.get_optimal_action(s_tilt)
    assert m_tilt > 0.0
    assert opt_a in [0, 1]
