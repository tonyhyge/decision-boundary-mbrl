"""
Ensemble Neural Dynamics Model for CartPole Continuous Control.

Provides:
  - CartPoleEnsembleDynamics: An ensemble of K neural dynamics networks f_theta_k(s, a)
    predicting continuous state transitions (delta_s = f(s, a) - s).
  - Methods to compute individual model predictions, ensemble mean, and epistemic dispersion.
"""
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.cartpole_continuous import CartPoleDynamics, CartPoleNeuralDynamics


class EnsembleDynamicsModel(nn.Module):
    """
    Bootstrap Ensemble of K feedforward neural dynamics models for CartPole.
    Each model k predicts next state delta: Delta_s_k = f_theta_k(s, a).
    """

    def __init__(self, num_models: int = 7, hidden_dim: int = 64, seed: int = 42):
        super().__init__()
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.seed = seed

        torch.manual_seed(seed)
        self.models = nn.ModuleList([
            CartPoleNeuralDynamics(hidden_dim=hidden_dim)
            for _ in range(num_models)
        ])

    def forward(self, sa_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all K models.
        Input: sa_tensor of shape (batch_size, 5) or (5,)
        Returns: Tensor of shape (K, batch_size, 4) or (K, 4)
        """
        preds = [model(sa_tensor) for model in self.models]
        return torch.stack(preds, dim=0)

    def predict_all_next_states(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Predict next state for all K ensemble models given (state, action).
        Returns: np.ndarray of shape (K, 4) representing predicted next states s'_k.
        """
        self.eval()
        with torch.no_grad():
            sa = np.concatenate([state, [1.0 if action == 1 else -1.0]])
            inp = torch.tensor(sa, dtype=torch.float32).unsqueeze(0)
            deltas = self.forward(inp).squeeze(1).cpu().numpy()  # Shape: (K, 4)
        return state + deltas

    def predict_all_actions_next_states(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next states for all models under action 0 and action 1.
        Returns:
            next_states_a0: np.ndarray of shape (K, 4)
            next_states_a1: np.ndarray of shape (K, 4)
        """
        next_s0 = self.predict_all_next_states(state, action=0)
        next_s1 = self.predict_all_next_states(state, action=1)
        return next_s0, next_s1

    def train_ensemble(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        bootstrap: bool = True,
        seed: int = 42,
    ) -> List[float]:
        """
        Train ensemble members with independent bootstrap resampling of the training data.
        Returns average final loss per member.
        """
        rng = np.random.default_rng(seed)
        criterion = nn.MSELoss()
        n_samples = len(x_data)
        final_losses = []

        for k, model in enumerate(self.models):
            torch.manual_seed(seed + k * 100)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if bootstrap:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                bx_train = torch.tensor(x_data[boot_idx], dtype=torch.float32)
                by_train = torch.tensor(y_data[boot_idx], dtype=torch.float32)
            else:
                bx_train = torch.tensor(x_data, dtype=torch.float32)
                by_train = torch.tensor(y_data, dtype=torch.float32)

            model.train()
            last_loss = 0.0
            for epoch in range(epochs):
                perm = rng.permutation(n_samples)
                for b in range(0, n_samples, batch_size):
                    idx = perm[b : b + batch_size]
                    batch_x = bx_train[idx]
                    batch_y = by_train[idx]

                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    last_loss = float(loss.item())

            final_losses.append(last_loss)

        return final_losses
