"""
Implicit Function Perturbation Law in Continuous-Action Reinforcement Learning.

Validates:
  delta_a_pred = - H_a^(-1) * nabla_a delta_Q(s, a*)
  versus actual continuous decision displacement: D_a(s) = a_hat(s) - a*(s).
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.stats as stats

from src.envs.continuous_control_geometry import ContinuousCartPoleEnv, ExactLQRGeometrySolver


class ImplicitPerturbationAuditor:
    """
    Evaluates theoretical first-order implicit-function perturbation predictions
    against true empirical decision displacement and decision regret.
    """

    def __init__(
        self,
        env: ContinuousCartPoleEnv,
        solver: ExactLQRGeometrySolver,
    ):
        self.env = env
        self.solver = solver

    def evaluate_local_perturbation(
        self,
        state: np.ndarray,
        delta_p_fn: callable,  # Function (s, a) -> delta_s_pred
    ) -> Dict[str, float]:
        """
        Evaluate exact perturbation response at state s under perturbation delta_f(s, a).
        """
        a_star = self.solver.get_optimal_action(state)
        h_a = self.solver.compute_action_hessian(state)  # H_a = nabla_a^2 Q_P < 0
        inv_h = 1.0 / h_a

        # True next state and true gradient at a_star
        # Numerically evaluate nabla_a delta_Q(s, a*) using finite difference
        eps_a = 1e-4
        a_plus = a_star + eps_a
        a_minus = a_star - eps_a

        # Model predicted next states
        true_ns_star, _, _ = self.env.step(state, a_star)
        pred_ns_star = true_ns_star + delta_p_fn(state, a_star)

        true_ns_plus, _, _ = self.env.step(state, a_plus)
        pred_ns_plus = true_ns_plus + delta_p_fn(state, a_plus)

        true_ns_minus, _, _ = self.env.step(state, a_minus)
        pred_ns_minus = true_ns_minus + delta_p_fn(state, a_minus)

        # Model Q values
        gamma = self.solver.gamma
        q_pred_star = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_star**2 + gamma * self.solver.evaluate_v(pred_ns_star))
        q_pred_plus = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_plus**2 + gamma * self.solver.evaluate_v(pred_ns_plus))
        q_pred_minus = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_minus**2 + gamma * self.solver.evaluate_v(pred_ns_minus))

        # True Q values
        q_true_star = self.solver.evaluate_q(state, a_star)
        q_true_plus = self.solver.evaluate_q(state, a_plus)
        q_true_minus = self.solver.evaluate_q(state, a_minus)

        # delta_Q values
        delta_q_star = q_pred_star - q_true_star
        delta_q_plus = q_pred_plus - q_true_plus
        delta_q_minus = q_pred_minus - q_true_minus

        # nabla_a delta_Q at a_star
        grad_delta_q = (delta_q_plus - delta_q_minus) / (2.0 * eps_a)

        # Theoretical prediction via implicit function theorem:
        # delta_a_pred = - H_a^(-1) * grad_delta_q
        delta_a_pred = - inv_h * grad_delta_q

        # Find actual model-optimal action a_hat via fine 1D line search
        action_grid = np.linspace(-1.0, 1.0, 401)
        q_pred_grid = []
        for a_cand in action_grid:
            t_ns, _, _ = self.env.step(state, a_cand)
            p_ns = t_ns + delta_p_fn(state, a_cand)
            qp = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_cand**2 + gamma * self.solver.evaluate_v(p_ns))
            q_pred_grid.append(qp)

        a_hat = float(action_grid[np.argmax(q_pred_grid)])
        actual_displacement = a_hat - a_star

        # Directional alignment: sign match / cosine
        sign_match = int(np.sign(delta_a_pred) == np.sign(actual_displacement)) if abs(actual_displacement) > 1e-4 else 1
        abs_error = float(abs(delta_a_pred - actual_displacement))
        rel_error = float(abs_error / (abs(actual_displacement) + 1e-4))

        # Decision Regret
        regret = self.solver.compute_decision_regret(state, a_hat)
        theoretical_regret = float(0.5 * abs(h_a) * delta_a_pred**2)

        return {
            "a_star": a_star,
            "a_hat": a_hat,
            "actual_displacement": actual_displacement,
            "delta_a_pred": delta_a_pred,
            "sign_match": sign_match,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "grad_delta_q": grad_delta_q,
            "h_a": h_a,
            "inv_h": inv_h,
            "actual_regret": regret,
            "theoretical_regret": theoretical_regret,
        }
