"""
Sample Prioritization & Loss Reweighting Mechanisms for World-Model Portability.

Implements the frozen intervention rule:
    w_i = (1 + lambda * s_i) / ( (1/N) * sum_j (1 + lambda * s_j) )

Supports 5 required experimental conditions:
  A. Uniform Baseline: w_i = 1.0
  B. Prediction-Error Weighting: s_i = normalized NLL under initial model
  C. Estimated Boundary-Crossing Weighting: s_i = p_hat_cross(s_i)
  D. Shuffled-Boundary Negative Control: w_i = permute(w_estimated)
  E. Oracle Boundary-Crossing Weighting: s_i = Z_cross_oracle(s_i)
  + Secondary Ablation: s_i = normalized B_tilde(s_i) (continuous boundary pressure)
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP
from src.planning.dp import value_iteration
from src.metrics.diagnostics import compute_action_margins, compute_boundary_pressure
from src.models.tabular_learned_model import TrajectoryDataset


def compute_normalized_weights(scores: np.ndarray, lambda_val: float) -> np.ndarray:
    """
    Compute normalized sample weights:
        w_i = (1 + lambda * s_i) / ( (1/N) * sum_j (1 + lambda * s_j) )
    Guarantees:
        mean(w_i) == 1.0
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return np.array([], dtype=np.float64)

    raw_w = 1.0 + lambda_val * s
    mean_w = np.mean(raw_w)
    if mean_w < 1e-12:
        return np.ones(n, dtype=np.float64)

    weights = raw_w / mean_w
    return weights


def compute_sample_scores(
    dataset: TrajectoryDataset,
    condition: str,
    true_mdp: ChoiceGridWorldMDP,
    initial_mdp: TabularMDP,
    initial_p_matrix: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    lambda_val: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raw priority scores s_i and normalized weights w_i for a given condition.

    Returns:
        scores (np.ndarray): Shape (N,)
        weights (np.ndarray): Shape (N,) with mean(weights) == 1.0
    """
    n = len(dataset)
    if n == 0:
        return np.array([]), np.array([])

    if rng is None:
        rng = np.random.default_rng(42)

    # 1. DP on true MDP and initial learned MDP
    V_star, Q_star, pi_star = value_iteration(true_mdp)
    m_true = compute_action_margins(Q_star, pi_star)

    V_hat, Q_hat, pi_hat = value_iteration(initial_mdp)
    m_hat = compute_action_margins(Q_hat, pi_hat)

    # Condition A: Uniform baseline
    if condition == "uniform":
        scores = np.zeros(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64)
        return scores, weights

    # Condition B: Prediction-error weighting
    elif condition == "prediction_error":
        errors = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = dataset.states[i]
            a = dataset.actions[i]
            next_s = dataset.next_states[i]
            prob = max(1e-6, float(initial_p_matrix[s, a, next_s]))
            errors[i] = -np.log(prob)  # Cross-entropy / NLL error

        # Normalize scores to [0, 1] range
        min_e, max_e = np.min(errors), np.max(errors)
        if max_e > min_e:
            scores = (errors - min_e) / (max_e - min_e)
        else:
            scores = np.zeros(n, dtype=np.float64)
        weights = compute_normalized_weights(scores, lambda_val)
        return scores, weights

    # Condition C: Estimated boundary-crossing weighting
    elif condition == "estimated_crossing":
        scores = np.zeros(n, dtype=np.float64)
        # Margin scale for calibrated temperature
        active_margins = [m_hat[s, a] for s in range(initial_mdp.num_states) if s != true_mdp.goal_state for a in range(initial_mdp.num_actions) if m_hat[s, a] > 1e-8]
        tau_m = float(np.median(active_margins)) if active_margins else 0.5

        for i in range(n):
            s = dataset.states[i]
            # State margin against runner-up action under initial model
            opt_a_hat = int(pi_hat[s])
            comp_margins = [m_hat[s, a] for a in range(initial_mdp.num_actions) if a != opt_a_hat]
            gap_hat = min(comp_margins) if comp_margins else 0.0

            # Calibrated crossing probability via soft margin thresholding:
            # p_hat_cross is large when gap_hat is small (near boundary)
            p_cross_hat = 1.0 / (1.0 + np.exp(gap_hat / (tau_m + 1e-4)))
            scores[i] = float(p_cross_hat)

        weights = compute_normalized_weights(scores, lambda_val)
        return scores, weights

    # Condition D: Shuffled-boundary negative control
    elif condition == "shuffled_crossing":
        # First compute estimated crossing weights
        _, est_weights = compute_sample_scores(
            dataset=dataset,
            condition="estimated_crossing",
            true_mdp=true_mdp,
            initial_mdp=initial_mdp,
            initial_p_matrix=initial_p_matrix,
            rng=rng,
            lambda_val=lambda_val,
        )
        # Randomly permute the weight vector across dataset samples
        shuffled_weights = rng.permutation(est_weights)
        # Verify exact invariant preservation: mean and histogram identical
        return np.zeros(n, dtype=np.float64), shuffled_weights

    # Condition E: Oracle boundary-crossing weighting
    elif condition == "oracle_crossing":
        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = dataset.states[i]
            # True decision flip at state s: model-induced greedy action != true optimal action
            z_cross = int(pi_hat[s] != pi_star[s])
            scores[i] = float(z_cross)

        weights = compute_normalized_weights(scores, lambda_val)
        return scores, weights

    # Optional secondary ablation: Continuous boundary pressure B_tilde
    elif condition == "continuous_pressure":
        B_all = compute_boundary_pressure(m_true, m_hat)
        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = dataset.states[i]
            a = dataset.actions[i]
            b_val = float(B_all[s, a])
            scores[i] = max(0.0, b_val)

        max_b = np.max(scores)
        if max_b > 1e-8:
            scores = scores / max_b
        weights = compute_normalized_weights(scores, lambda_val)
        return scores, weights

    else:
        raise ValueError(f"Unknown experimental condition: {condition}")
