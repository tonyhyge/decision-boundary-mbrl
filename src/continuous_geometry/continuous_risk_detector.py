"""
Continuous-Action Risk Signals and Unprivileged Decision-Risk Detectors.

Provides:
  - ContinuousEnsembleDynamics: Neural dynamics ensemble for continuous (s, a).
  - ContinuousUnprivilegedSignalExtractor: Extracts:
      1. sigma_dyn(s): Dynamics epistemic uncertainty
      2. inv_H_hat(s): Estimated inverse curvature (flatness vulnerability)
      3. sigma_grad_Q(s): Action-gradient epistemic disagreement
      4. sigma_a(s): Action ensemble disagreement
      5. S_cont(s): Continuous decision-risk score = sigma_grad_Q / (|H_hat| + eps)
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.continuous_control_geometry import ContinuousCartPoleEnv, ExactLQRGeometrySolver


class ContinuousNeuralDynamics(nn.Module):
    """Predicts next-state delta Delta_s = f_theta(s, a) for continuous s in R^4, a in [-1, 1]."""

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

    def predict_next_state(self, state: np.ndarray, action: float) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            sa = np.concatenate([state, [float(action)]])
            inp = torch.tensor(sa, dtype=torch.float32).unsqueeze(0)
            delta = self.net(inp).numpy()[0]
        return state + delta


class ContinuousEnsembleDynamics(nn.Module):
    """Bootstrap Ensemble of K neural dynamics networks for continuous control."""

    def __init__(self, num_models: int = 7, hidden_dim: int = 64, seed: int = 42):
        super().__init__()
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.seed = seed

        torch.manual_seed(seed)
        self.models = nn.ModuleList([
            ContinuousNeuralDynamics(hidden_dim=hidden_dim)
            for _ in range(num_models)
        ])

    def predict_all_next_states(self, state: np.ndarray, action: float) -> np.ndarray:
        """Predict next states for all K models given (state, action). Returns (K, 4)."""
        self.eval()
        with torch.no_grad():
            sa = np.concatenate([state, [float(action)]])
            inp = torch.tensor(sa, dtype=torch.float32).unsqueeze(0)
            deltas = np.array([m(inp).numpy()[0] for m in self.models])
        return state + deltas

    def train_ensemble(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> List[float]:
        rng = np.random.default_rng(seed)
        criterion = nn.MSELoss()
        n_samples = len(x_data)
        final_losses = []

        for k, model in enumerate(self.models):
            torch.manual_seed(seed + k * 100)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            model.train()
            last_loss = 0.0

            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            bx_train = torch.tensor(x_data[boot_idx], dtype=torch.float32)
            by_train = torch.tensor(y_data[boot_idx], dtype=torch.float32)

            for epoch in range(epochs):
                perm = rng.permutation(n_samples)
                for b in range(0, n_samples, batch_size):
                    idx = perm[b : b + batch_size]
                    bx = bx_train[idx]
                    by = by_train[idx]

                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    last_loss = float(loss.item())

            final_losses.append(last_loss)

        return final_losses


class ContinuousUnprivilegedSignalExtractor:
    """
    Extracts continuous decision-risk signals and exact decision regret evaluation:
      1. Dynamics epistemic disagreement sigma_dyn
      2. Estimated inverse curvature / flatness vulnerability inv_H_hat
      3. Action-gradient epistemic disagreement sigma_grad_Q
      4. Action ensemble disagreement sigma_a
      5. Continuous decision-risk score S_cont
    """

    def __init__(
        self,
        ensemble: ContinuousEnsembleDynamics,
        solver: ExactLQRGeometrySolver,
        env: ContinuousCartPoleEnv,
    ):
        self.ensemble = ensemble
        self.solver = solver
        self.env = env
        self.action_grid = np.linspace(-1.0, 1.0, 101)

    def extract_state_signals(
        self,
        state: np.ndarray,
        eps_ratio: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Extract model-internal continuous signals without true dynamics f_P.
        """
        num_models = self.ensemble.num_models
        gamma = self.solver.gamma

        # 1. Model-predicted Q-values across action grid for each ensemble member
        # Q_hat_k(s, a) = - s^T Q s - R a^2 + gamma * V_ref(f_hat_k(s, a))
        q_grid_models = np.zeros((num_models, len(self.action_grid)), dtype=np.float64)

        for i, a_val in enumerate(self.action_grid):
            next_states_k = self.ensemble.predict_all_next_states(state, a_val)  # (K, 4)
            stage_r = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_val**2)
            for k in range(num_models):
                q_grid_models[k, i] = stage_r + gamma * self.solver.evaluate_v(next_states_k[k])

        # Mean predicted Q landscape
        mean_q_grid = np.mean(q_grid_models, axis=0)
        best_idx = int(np.argmax(mean_q_grid))
        a_hat = float(self.action_grid[best_idx])

        # Optimal action for each model member
        opt_actions_k = np.array([self.action_grid[np.argmax(q_grid_models[k, :])] for k in range(num_models)])
        sigma_a = float(np.std(opt_actions_k, ddof=1)) if num_models > 1 else 0.0

        # Numerical gradient and Hessian of Q around a_hat
        eps_a = 0.02
        a_plus = np.clip(a_hat + eps_a, -1.0, 1.0)
        a_minus = np.clip(a_hat - eps_a, -1.0, 1.0)
        actual_step = (a_plus - a_minus)

        # Gradients for each model member at a_hat
        grads_k = []
        for k in range(num_models):
            ns_plus = self.ensemble.predict_all_next_states(state, a_plus)[k]
            ns_minus = self.ensemble.predict_all_next_states(state, a_minus)[k]
            qp = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_plus**2 + gamma * self.solver.evaluate_v(ns_plus))
            qm = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_minus**2 + gamma * self.solver.evaluate_v(ns_minus))
            g_k = (qp - qm) / max(1e-4, actual_step)
            grads_k.append(g_k)

        grads_k = np.array(grads_k)
        sigma_grad_q = float(np.std(grads_k, ddof=1)) if num_models > 1 else 0.0

        # Estimated action Hessian (curvature) of mean Q around a_hat
        # H_hat = (Q(a+) - 2 Q(a_hat) + Q(a-)) / (eps_a^2)
        ns_hat_k = self.ensemble.predict_all_next_states(state, a_hat)
        q_mean_hat = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_hat**2 + gamma * np.mean([self.solver.evaluate_v(ns_hat_k[k]) for k in range(num_models)]))
        q_mean_plus = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_plus**2 + gamma * np.mean([self.solver.evaluate_v(self.ensemble.predict_all_next_states(state, a_plus)[k]) for k in range(num_models)]))
        q_mean_minus = float(- state.T @ self.solver.Q_mat @ state - self.solver.r_cost * a_minus**2 + gamma * np.mean([self.solver.evaluate_v(self.ensemble.predict_all_next_states(state, a_minus)[k]) for k in range(num_models)]))

        h_hat = float((q_mean_plus - 2.0 * q_mean_hat + q_mean_minus) / max(1e-6, (eps_a ** 2)))
        curvature = float(max(1e-3, abs(h_hat)))
        inv_h_hat = float(1.0 / curvature)

        # Dynamics epistemic disagreement across actions a in {-0.5, 0.0, 0.5}
        dyn_vars = []
        for a_test in [-0.5, 0.0, 0.5]:
            pred_ns_all = self.ensemble.predict_all_next_states(state, a_test)
            v_a = np.sum(np.var(pred_ns_all, axis=0, ddof=1)) if num_models > 1 else 0.0
            dyn_vars.append(v_a)
        sigma_dyn = float(np.sqrt(np.mean(dyn_vars)))

        # Continuous Decision-Risk Score: S_cont = sigma_grad_q * inv_H_hat
        s_cont = float(sigma_grad_q / (curvature + eps_ratio))

        # Ground truth evaluation metrics (for evaluation only)
        a_star = self.solver.get_optimal_action(state)
        displacement = float(abs(a_hat - a_star))
        regret = self.solver.compute_decision_regret(state, a_hat)

        return {
            "a_hat": a_hat,
            "a_star": a_star,
            "displacement": displacement,
            "regret": regret,
            "sigma_dyn": sigma_dyn,
            "inv_h_hat": inv_h_hat,
            "curvature": curvature,
            "sigma_grad_q": sigma_grad_q,
            "sigma_a": sigma_a,
            "s_cont": s_cont,
        }
