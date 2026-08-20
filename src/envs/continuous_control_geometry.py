"""
Continuous Action Control Environment and Exact Local Geometry Solver for Stage 3B.

Provides:
  - ContinuousCartPoleEnv: Fully continuous state (4D) and continuous action a in [-1, 1] (torque/force).
  - ExactLQRGeometrySolver: Provides exact Riccati solution for V_P(s), Q_P(s, a),
    exact optimal action a*(s), action gradient nabla_a Q_P(s, a), and exact Hessian H_a = nabla_a^2 Q_P(s, a).
  - Configurable action curvature via control penalty R_cost and physics parameters.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn


class ContinuousCartPoleEnv:
    """
    Continuous-action CartPole environment:
      State: s = [x, x_dot, theta, theta_dot] in R^4
      Action: a in [-1.0, 1.0] representing normalized continuous force F = a * force_mag
    """

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        dt: float = 0.02,
    ):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = masscart + masspole
        self.length = length
        self.polemass_length = masspole * length
        self.force_mag = force_mag
        self.dt = dt

    def step(self, state: np.ndarray, action: float) -> Tuple[np.ndarray, float, bool]:
        """Physics step with continuous action a in [-1, 1]."""
        x, x_dot, theta, theta_dot = state
        force = np.clip(action, -1.0, 1.0) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        done = bool(
            x < -2.4 or x > 2.4 or theta < -12 * 2 * np.pi / 360 or theta > 12 * 2 * np.pi / 360
        )
        # Continuous quadratic stage reward centered at equilibrium
        reward = float(-(1.0 * x**2 + 0.1 * x_dot**2 + 10.0 * theta**2 + 0.5 * theta_dot**2 + 0.1 * action**2))
        return next_state, reward, done

    def get_linearized_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous-time linearization around upright equilibrium (s=0, a=0):
          s_dot = A_c * s + B_c * a
        Returns discrete-time dynamics matrices (A_d, B_d) via matrix exponential.
        """
        # Linearized continuous-time system
        M = self.total_mass
        m = self.masspole
        l = self.length
        g = self.gravity
        denom = l * (4.0 / 3.0 - m / M)

        A_c = np.zeros((4, 4), dtype=np.float64)
        A_c[0, 1] = 1.0
        A_c[2, 3] = 1.0
        A_c[1, 2] = -m * g / M
        A_c[3, 2] = g / denom

        B_c = np.zeros((4, 1), dtype=np.float64)
        B_c[1, 0] = self.force_mag / M
        B_c[3, 0] = -self.force_mag / (M * denom)

        # Euler discretization: A_d = I + A_c * dt, B_d = B_c * dt
        A_d = np.eye(4, dtype=np.float64) + A_c * self.dt
        B_d = B_c * self.dt
        return A_d, B_d


class ExactLQRGeometrySolver:
    """
    Exact Discrete-time Algebraic Riccati Equation (DARE) solver providing exact
    value function V_P(s), Q_P(s, a), optimal action a*(s), gradient nabla_a Q,
    and exact Hessian H_a = nabla_a^2 Q(s, a).
    """

    def __init__(
        self,
        env: ContinuousCartPoleEnv,
        q_diag: Optional[List[float]] = None,
        r_cost: float = 0.1,
        gamma: float = 0.98,
    ):
        self.env = env
        self.q_diag = q_diag or [1.0, 0.1, 10.0, 0.5]
        self.r_cost = r_cost
        self.gamma = gamma

        self.Q_mat = np.diag(self.q_diag)
        self.R_mat = np.array([[self.r_cost]], dtype=np.float64)

        # Solve Discrete Algebraic Riccati Equation for discounted infinite-horizon
        A, B = self.env.get_linearized_system()
        self.A = np.sqrt(gamma) * A
        self.B = np.sqrt(gamma) * B

        # DARE: P = A^T P A - (A^T P B)(R + B^T P B)^(-1)(B^T P A) + Q
        self.P = la.solve_discrete_are(self.A, self.B, self.Q_mat, self.R_mat)

        # LQR feedback gain: K = (R + B^T P B)^(-1) B^T P A
        self.H_scalar = float((self.r_cost + self.B.T @ self.P @ self.B).item())
        self.K = (1.0 / self.H_scalar) * (self.B.T @ self.P @ self.A)  # Shape (1, 4)

    def evaluate_v(self, state: np.ndarray) -> float:
        """V(s) = - s^T P s."""
        return float((- state.T @ self.P @ state).item())

    def evaluate_q(self, state: np.ndarray, action: float) -> float:
        """
        Q_P(s, a) = - s^T Q s - R a^2 + gamma * V(A s + B a)
        Quadratic in action a: Q_P(s, a) = - 0.5 * (-H_a) * (a - a*(s))^2 + const(s).
        """
        next_s = self.A @ state + self.B.flatten() * action
        stage_r = float((- state.T @ self.Q_mat @ state - self.r_cost * action**2).item())
        v_next = self.evaluate_v(next_s)
        return stage_r + v_next

    def get_optimal_action(self, state: np.ndarray) -> float:
        """a*(s) = - K s."""
        a_star = float((- self.K @ state).item())
        return float(np.clip(a_star, -1.0, 1.0))

    def compute_action_gradient(self, state: np.ndarray, action: float) -> float:
        """
        nabla_a Q_P(s, a) = - 2 * (R + B^T P B) * (a - a*(s)) = H_a * (a - a*(s)).
        At a = a*(s), nabla_a Q_P = 0.
        """
        a_star = float((- self.K @ state).item())
        # H_a = - 2 * (R + B^T P B)
        h_a = - 2.0 * self.H_scalar
        return float(h_a * (action - a_star))

    def compute_action_hessian(self, state: Optional[np.ndarray] = None) -> float:
        """
        H_a = nabla_a^2 Q_P(s, a) = - 2 * (R + B^T P B) < 0.
        Curvature (concavity) is |H_a| = 2 * (R + B^T P B).
        Flatness vulnerability is |H_a|^(-1).
        """
        return float(- 2.0 * self.H_scalar)

    def compute_decision_regret(self, state: np.ndarray, chosen_action: float) -> float:
        """
        Decision Regret: R_dec(s) = Q_P(s, a*) - Q_P(s, chosen_action)
        = 0.5 * |H_a| * (chosen_action - a*(s))^2.
        """
        a_star = self.get_optimal_action(state)
        q_star = self.evaluate_q(state, a_star)
        q_chosen = self.evaluate_q(state, chosen_action)
        return float(max(0.0, q_star - q_chosen))
