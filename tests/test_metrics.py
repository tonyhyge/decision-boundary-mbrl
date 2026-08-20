"""
Unit tests for Diagnostics and Statistical Metrics.
"""
import pytest
import numpy as np
import pandas as pd
from src.metrics.diagnostics import (
    compute_action_margins,
    compute_margin_deformation,
    compute_boundary_pressure,
    compute_value_sensitivity,
    evaluate_incremental_r2,
    evaluate_matched_pair_effect,
)


def test_action_margin_computation():
    Q = np.array([
        [1.0, 0.7, 0.4],
        [0.2, 0.9, 0.5],
    ])
    opt_a = np.array([0, 1])
    margins = compute_action_margins(Q, opt_a)

    # State 0: opt is 0 (val 1.0) -> margins = [0.0, 0.3, 0.6]
    assert np.allclose(margins[0], [0.0, 0.3, 0.6])
    # State 1: opt is 1 (val 0.9) -> margins = [0.7, 0.0, 0.4]
    assert np.allclose(margins[1], [0.7, 0.0, 0.4])


def test_boundary_pressure_sign():
    m_true = np.array([[0.0, 0.5]])
    # Compressive: margin drops from 0.5 to 0.1 -> Delta m = -0.4 -> B = -(-0.4)/(0.5) = +0.8 > 0
    m_comp = np.array([[0.0, 0.1]])
    b_comp = compute_boundary_pressure(m_true, m_comp, eps=1e-6)
    assert b_comp[0, 1] > 0.0

    # Expansive: margin rises from 0.5 to 0.8 -> Delta m = +0.3 -> B = -(+0.3)/(0.5) = -0.6 < 0
    m_exp = np.array([[0.0, 0.8]])
    b_exp = compute_boundary_pressure(m_true, m_exp, eps=1e-6)
    assert b_exp[0, 1] < 0.0


def test_incremental_r2_synthetic():
    rng = np.random.default_rng(42)
    n = 100
    x_noise = rng.normal(size=(n, 3))
    b_signal = rng.normal(size=n)
    # Target strongly depends on b_signal
    y = 0.1 * x_noise[:, 0] + 2.0 * b_signal + rng.normal(scale=0.1, size=n)

    df = pd.DataFrame({
        "error_l1": x_noise[:, 0],
        "occupancy": x_noise[:, 1],
        "true_action_gap": x_noise[:, 2],
        "boundary_pressure": b_signal,
        "correction_value": y,
    })

    res = evaluate_incremental_r2(
        df,
        target_col="correction_value",
        control_cols=["error_l1", "occupancy", "true_action_gap"],
        proposed_col="boundary_pressure",
    )

    assert res["delta_r2"] > 0.5
    assert res["p_val"] < 1e-4


def test_margin_stability_lemma():
    """
    Test Lemma 1: If 2 * e_Q(s) < m_P(s), then policy does not flip (Z_flip = 0).
    """
    # True Q values: opt action is 0 with val 1.0, runner-up is 1 with val 0.6 -> margin = 0.4
    Q_true = np.array([[1.0, 0.6]])
    margin_true = Q_true[0, 0] - Q_true[0, 1]  # 0.4

    # Case 1: Perturbation with 2 * e_Q < margin (e_Q = 0.15 < 0.20)
    # Q_hat = [0.85, 0.75] -> opt is still 0 (val 0.85 > 0.75), e_Q = 0.15, no flip
    Q_hat_stable = np.array([[0.85, 0.75]])
    e_Q_stable = np.max(np.abs(Q_hat_stable - Q_true))
    assert 2 * e_Q_stable < margin_true
    assert np.argmax(Q_hat_stable) == np.argmax(Q_true)  # Z_flip == 0

    # Case 2: Perturbation with 2 * e_Q > margin (e_Q = 0.25 > 0.20)
    # Q_hat = [0.75, 0.85] -> opt flips to 1, e_Q = 0.25
    Q_hat_flip = np.array([[0.75, 0.85]])
    e_Q_flip = np.max(np.abs(Q_hat_flip - Q_true))
    assert 2 * e_Q_flip >= margin_true
    assert np.argmax(Q_hat_flip) != np.argmax(Q_true)  # Z_flip == 1


def test_exact_binary_regret_identity():
    """
    Test Corollary 1: For |A|=2, Regret = 1/(1-gamma) * E_{d_P^pi_hat}[ m_P(s) * Z_flip(s) ].
    """
    gamma = 0.9
    # Two states: State 0 margin 0.5 (flipped), State 1 margin 0.8 (not flipped)
    margins = np.array([0.5, 0.8])
    z_flip = np.array([1, 0])
    occupancies = np.array([0.6, 0.4])  # sum to 1

    expected_regret = (1.0 / (1.0 - gamma)) * np.sum(occupancies * margins * z_flip)
    # expected_regret = 10 * (0.6 * 0.5 * 1 + 0.4 * 0.8 * 0) = 10 * 0.3 = 3.0
    assert np.isclose(expected_regret, 3.0)

