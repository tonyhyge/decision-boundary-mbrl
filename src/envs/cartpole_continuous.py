"""
CartPole Continuous Dynamics and Competent Value Function for External Validity.

Provides:
  - Exact nonlinear CartPole continuous-time dynamics discretization.
  - Pre-computed competent linear-quadratic / MLP value function Q(s, a).
  - Action margin m(s) = |Q(s, 0) - Q(s, 1)|.
  - Neural dynamics model f_theta(s, a) predicting state deltas.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CartPoleDynamics:
    """Exact 4D CartPole continuous-time ODE integrator (Euler/RK4 step)."""

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

    def step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        """Integrate continuous physics step from state s given discrete action a in {0, 1}."""
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
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
        reward = 1.0 if not done else 0.0
        return next_state, reward, done


class CompetentCartPoleValueFunction:
    """
    Analytic / Fitted LQR-based value function for CartPole:
      V(s) approx -s^T P s (quadratic Lyapunov cost-to-go)
      Q(s, a) = r(s, a) + gamma * V(f(s, a))
      margin m(s) = |Q(s, 0) - Q(s, 1)|
    """

    def __init__(self, dynamics: Optional[CartPoleDynamics] = None, gamma: float = 0.98):
        self.dynamics = dynamics or CartPoleDynamics()
        self.gamma = gamma
        # LQR gain matrix for linearized CartPole around [0, 0, 0, 0]
        self.P = np.diag([1.0, 0.5, 10.0, 1.0])
        self.K = np.array([-0.8, -1.2, -18.5, -3.2])

    def evaluate_v(self, state: np.ndarray) -> float:
        # Cost-to-go inverted to positive return proxy: 1 / (1 + s^T P s) * max_steps
        cost = float(state.T @ self.P @ state)
        return float(50.0 / (1.0 + cost))

    def evaluate_q(self, state: np.ndarray, action: int) -> float:
        next_s, r, done = self.dynamics.step(state, action)
        if done:
            return r
        return r + self.gamma * self.evaluate_v(next_s)

    def compute_action_margin(self, state: np.ndarray) -> float:
        q0 = self.evaluate_q(state, 0)
        q1 = self.evaluate_q(state, 1)
        return float(abs(q0 - q1))

    def get_optimal_action(self, state: np.ndarray) -> int:
        q0 = self.evaluate_q(state, 0)
        q1 = self.evaluate_q(state, 1)
        return 1 if q1 >= q0 else 0


class CartPoleNeuralDynamics(nn.Module):
    """Predicts state delta: Delta_s = f_theta(s, a)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, sa_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(sa_tensor)

    def predict_next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            sa = np.concatenate([state, [1.0 if action == 1 else -1.0]])
            inp = torch.tensor(sa, dtype=torch.float32).unsqueeze(0)
            delta = self.net(inp).numpy()[0]
        return state + delta
