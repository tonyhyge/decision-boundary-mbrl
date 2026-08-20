"""
Unit Tests for Stage 3B: Continuous-Action Geometry and Perturbation Law.

Verifies:
  1. ExactLQRGeometrySolver correctly computes Q_P(s, a), gradient nabla_a Q, and Hessian H_a.
  2. ImplicitPerturbationAuditor validates delta_a_pred = - H_a^(-1) grad_delta_q with high cosine alignment.
  3. ContinuousUnprivilegedSignalExtractor extracts all continuous risk signals without NaNs.
"""
import numpy as np
import pytest

from src.envs.continuous_control_geometry import ContinuousCartPoleEnv, ExactLQRGeometrySolver
from src.continuous_geometry.implicit_perturbation import ImplicitPerturbationAuditor
from src.continuous_geometry.continuous_risk_detector import (
    ContinuousEnsembleDynamics,
    ContinuousUnprivilegedSignalExtractor,
)


@pytest.fixture
def continuous_setup():
    env = ContinuousCartPoleEnv()
    solver = ExactLQRGeometrySolver(env=env, r_cost=0.1)
    auditor = ImplicitPerturbationAuditor(env=env, solver=solver)
    return env, solver, auditor


def test_exact_lqr_derivatives(continuous_setup):
    env, solver, _ = continuous_setup
    s = np.array([0.05, -0.02, 0.03, 0.01])
    a_star = solver.get_optimal_action(s)

    # Gradient at optimum should be zero
    grad_star = solver.compute_action_gradient(s, a_star)
    assert abs(grad_star) < 1e-4, f"Gradient at optimum should be 0, got {grad_star}"

    # Hessian should be negative (concave Q)
    h_a = solver.compute_action_hessian(s)
    assert h_a < 0.0, f"Hessian should be negative, got {h_a}"

    # Check finite-difference approximation of gradient and Hessian
    eps = 1e-4
    q_plus = solver.evaluate_q(s, a_star + eps)
    q_minus = solver.evaluate_q(s, a_star - eps)
    q_star = solver.evaluate_q(s, a_star)

    grad_num = (q_plus - q_minus) / (2.0 * eps)
    h_num = (q_plus - 2.0 * q_star + q_minus) / (eps ** 2)

    assert abs(grad_num - grad_star) < 1e-3
    assert abs(h_num - h_a) / abs(h_a) < 1e-2


def test_implicit_perturbation_law(continuous_setup):
    env, solver, auditor = continuous_setup
    rng = np.random.default_rng(42)

    # Test perturbation response on 20 random states
    states = rng.uniform(low=[-0.2, -0.2, -0.08, -0.10], high=[0.2, 0.2, 0.08, 0.10], size=(20, 4))
    cos_sims = []

    for s in states:
        pert = rng.normal(scale=0.005, size=4)
        res = auditor.evaluate_local_perturbation(s, lambda st, ac: pert)
        assert res["actual_regret"] >= 0.0

        if abs(res["actual_displacement"]) > 1e-4:
            cos = float(np.sign(res["delta_a_pred"]) * np.sign(res["actual_displacement"]))
            cos_sims.append(cos)

    assert len(cos_sims) > 0
    assert np.mean(cos_sims) > 0.80, f"Directional alignment should be high, got {np.mean(cos_sims)}"


def test_continuous_unprivileged_signal_extractor():
    env = ContinuousCartPoleEnv()
    solver = ExactLQRGeometrySolver(env=env, r_cost=0.1)
    ensemble = ContinuousEnsembleDynamics(num_models=3, hidden_dim=32, seed=42)

    # Synthetic train
    rng = np.random.default_rng(123)
    x_synth = rng.normal(scale=0.1, size=(50, 5)).astype(np.float32)
    y_synth = rng.normal(scale=0.01, size=(50, 4)).astype(np.float32)
    ensemble.train_ensemble(x_synth, y_synth, epochs=3, batch_size=32, seed=42)

    extractor = ContinuousUnprivilegedSignalExtractor(ensemble=ensemble, solver=solver, env=env)
    s = np.array([0.02, -0.01, 0.04, 0.02])
    signals = extractor.extract_state_signals(s)

    expected_keys = [
        "a_hat",
        "a_star",
        "displacement",
        "regret",
        "sigma_dyn",
        "inv_h_hat",
        "curvature",
        "sigma_grad_q",
        "sigma_a",
        "s_cont",
    ]
    for k in expected_keys:
        assert k in signals
        assert not np.isnan(signals[k])

    assert signals["displacement"] >= 0.0
    assert signals["regret"] >= 0.0
    assert signals["s_cont"] >= 0.0
