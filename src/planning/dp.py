"""
Exact dynamic programming: Value Iteration, Policy Evaluation, Occupancies, and Return computation.
"""
from typing import Tuple, Union
import numpy as np
from src.envs.tabular_mdp import TabularMDP


def to_policy_matrix(policy: Union[np.ndarray, list], num_states: int, num_actions: int) -> np.ndarray:
    """
    Convert a 1D action array or 2D policy matrix into a standardized (num_states, num_actions) stochastic matrix.
    """
    policy_arr = np.array(policy)
    if policy_arr.ndim == 1:
        assert len(policy_arr) == num_states, f"Policy length {len(policy_arr)} != num_states {num_states}"
        mat = np.zeros((num_states, num_actions), dtype=np.float64)
        for s, a in enumerate(policy_arr):
            mat[s, int(a)] = 1.0
        return mat
    elif policy_arr.ndim == 2:
        assert policy_arr.shape == (num_states, num_actions), f"Invalid policy shape {policy_arr.shape}"
        # Normalize rows to ensure simplex
        row_sums = policy_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return policy_arr / row_sums
    else:
        raise ValueError(f"Unsupported policy dimension: {policy_arr.ndim}")


def value_iteration(
    mdp: TabularMDP, tol: float = 1e-12, max_iter: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform exact Value Iteration to compute optimal value, Q-values, and greedy policy.
    
    Returns:
        V_star (np.ndarray): Shape (num_states,), optimal state values V*(s).
        Q_star (np.ndarray): Shape (num_states, num_actions), optimal action values Q*(s, a).
        pi_star (np.ndarray): Shape (num_states,), optimal deterministic actions a*(s).
    """
    V = np.zeros(mdp.num_states, dtype=np.float64)
    gamma = mdp.gamma
    P = mdp.transitions  # (|S|, |A|, |S|)
    R = mdp.rewards      # (|S|, |A|)

    for it in range(max_iter):
        # Q(s, a) = R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s')
        # Shape: (|S|, |A|) = (|S|, |A|) + gamma * sum_over_s'( (|S|, |A|, |S|) * (|S|,) )
        Q = R + gamma * np.einsum("ijk,k->ij", P, V)
        V_new = np.max(Q, axis=1)

        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break

    Q_star = R + gamma * np.einsum("ijk,k->ij", P, V)
    pi_star = np.argmax(Q_star, axis=1)

    return V, Q_star, pi_star


def compute_q_from_v(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """Compute Q(s, a) = R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s')."""
    return mdp.rewards + mdp.gamma * np.einsum("ijk,k->ij", mdp.transitions, V)


def policy_evaluation(mdp: TabularMDP, policy: Union[np.ndarray, list]) -> np.ndarray:
    """
    Perform exact Policy Evaluation for a given policy in the specified MDP.
    
    Solves the linear system:
        V^pi = (I - gamma * P^pi)^{-1} R^pi
    
    Returns:
        V_pi (np.ndarray): Shape (num_states,), state values under policy.
    """
    pi_mat = to_policy_matrix(policy, mdp.num_states, mdp.num_actions)
    # P_pi(s, s') = sum_a pi(a | s) * P(s' | s, a) -> shape (|S|, |S|)
    P_pi = np.einsum("ia,iaj->ij", pi_mat, mdp.transitions)
    # R_pi(s) = sum_a pi(a | s) * R(s, a) -> shape (|S|,)
    R_pi = np.einsum("ia,ia->i", pi_mat, mdp.rewards)

    I = np.eye(mdp.num_states, dtype=np.float64)
    A = I - mdp.gamma * P_pi
    V_pi = np.linalg.solve(A, R_pi)
    return V_pi


def compute_occupancy(
    mdp: TabularMDP, policy: Union[np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized discounted state occupancy d^pi(s) and state-action occupancy d^pi(s, a).
    
    d^pi(s) = (1 - gamma) * mu_0^T (I - gamma * P^pi)^{-1}
    d^pi(s, a) = d^pi(s) * pi(a | s)
    
    Returns:
        d_s (np.ndarray): Shape (num_states,), sum(d_s) = 1.0.
        d_sa (np.ndarray): Shape (num_states, num_actions), sum(d_sa) = 1.0.
    """
    pi_mat = to_policy_matrix(policy, mdp.num_states, mdp.num_actions)
    P_pi = np.einsum("ia,iaj->ij", pi_mat, mdp.transitions)

    I = np.eye(mdp.num_states, dtype=np.float64)
    # (I - gamma * P^pi)^T d = (1 - gamma) mu_0
    A_T = (I - mdp.gamma * P_pi).T
    b = (1.0 - mdp.gamma) * mdp.initial_dist
    d_s = np.linalg.solve(A_T, b)
    
    # Clip negative values due to numerical precision
    d_s = np.maximum(d_s, 0.0)
    d_s = d_s / np.sum(d_s)

    d_sa = d_s[:, None] * pi_mat
    return d_s, d_sa


def expected_discounted_return(mdp: TabularMDP, policy: Union[np.ndarray, list]) -> float:
    """
    Compute the true expected discounted return J(pi; P) = E_{s_0 ~ mu_0}[V^pi(s_0)].
    """
    V_pi = policy_evaluation(mdp, policy)
    return float(np.dot(mdp.initial_dist, V_pi))
