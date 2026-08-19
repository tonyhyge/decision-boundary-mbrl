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
