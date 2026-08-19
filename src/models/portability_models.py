"""
Host Dynamics Models for World-Model Portability Benchmark.

Provides:
  - Host A: Deterministic Neural Transition Model (Categorical MLP).
  - Host B: Probabilistic Ensemble Dynamics Model (K-member MLP ensemble with posterior mean planning).
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP
from src.models.tabular_learned_model import TrajectoryDataset, CategoricalTransitionNet


class WeightedCategoricalWorldModel:
    """
    Host A: Single deterministic/categorical neural world model.
    Trained via sample-weighted cross-entropy loss:
        L_BA = (1/N) * sum_i w_i * L_CE(s'_i, f_theta(s_i, a_i))
    where sum_i w_i = N.
    """

    def __init__(
        self,
        num_states: int = 25,
        num_actions: int = 4,
        gamma: float = 0.95,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device
        self.net = CategoricalTransitionNet(num_states, num_actions, hidden_dim).to(device)

    def fit(
        self,
        dataset: TrajectoryDataset,
        sample_weights: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        seed: int = 42,
        normalize_weights: bool = True,
    ) -> List[float]:
        torch.manual_seed(seed)
        n = len(dataset)
        if n == 0:
            return []

        if sample_weights is None:
            weights = np.ones(n, dtype=np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=np.float32)
            if normalize_weights:
                # Ensure mean weight is exactly 1.0
                w_mean = np.mean(weights)
                if w_mean > 1e-8:
                    weights = weights / w_mean

        sa_onehot = np.zeros((n, self.num_states + self.num_actions), dtype=np.float32)
        for i in range(n):
            s = dataset.states[i]
            a = dataset.actions[i]
            sa_onehot[i, s] = 1.0
            sa_onehot[i, self.num_states + a] = 1.0

        targets = np.array(dataset.next_states, dtype=np.int64)

        x_tensor = torch.tensor(sa_onehot, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(reduction="none")

        losses = []
        indices = np.arange(n)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            num_batches = int(np.ceil(n / batch_size))

            for b in range(num_batches):
                b_idx = indices[b * batch_size : (b + 1) * batch_size]
                bx = x_tensor[b_idx]
                by = y_tensor[b_idx]
                bw = w_tensor[b_idx]

                optimizer.zero_grad()
                logits = self.net(bx)
                unweighted_loss = criterion(logits, by)
                loss = torch.mean(bw * unweighted_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(b_idx)

            losses.append(epoch_loss / n)

        return losses

    def get_transition_matrix(self) -> np.ndarray:
        """Extract full (|S|, |A|, |S|) simplex transition matrix."""
        self.net.eval()
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)

        with torch.no_grad():
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    sa = np.zeros((1, self.num_states + self.num_actions), dtype=np.float32)
                    sa[0, s] = 1.0
                    sa[0, self.num_states + a] = 1.0
                    bx = torch.tensor(sa, dtype=torch.float32, device=self.device)
                    logits = self.net(bx)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    p_hat[s, a, :] = probs

        goal_s = self.num_states - 1
        for a in range(self.num_actions):
            p_hat[goal_s, a, :] = 0.0
            p_hat[goal_s, a, goal_s] = 1.0

        return p_hat

    def create_learned_mdp(self, true_mdp: ChoiceGridWorldMDP) -> TabularMDP:
        p_hat = self.get_transition_matrix()
        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=p_hat,
            rewards=true_mdp.rewards.copy(),
            gamma=self.gamma,
            initial_dist=true_mdp.initial_dist.copy(),
        )


class ProbabilisticEnsembleWorldModel:
    """
    Host B: Probabilistic Ensemble Dynamics Model (K independent neural models).
    Each member is trained via sample-weighted cross-entropy loss on dataset D.
    At planning time:
        P_bar(s' | s, a) = (1/K) * sum_{k=1}^K P_{theta_k}(s' | s, a)
    """

    def __init__(
        self,
        num_states: int = 25,
        num_actions: int = 4,
        ensemble_size: int = 5,
        gamma: float = 0.95,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.gamma = gamma
        self.device = device

        self.members: List[CategoricalTransitionNet] = [
            CategoricalTransitionNet(num_states, num_actions, hidden_dim).to(device)
            for _ in range(ensemble_size)
        ]

    def fit(
        self,
        dataset: TrajectoryDataset,
        sample_weights: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        base_seed: int = 42,
        normalize_weights: bool = True,
    ) -> List[List[float]]:
        n = len(dataset)
        if n == 0:
            return []

        if sample_weights is None:
            weights = np.ones(n, dtype=np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=np.float32)
            if normalize_weights:
                w_mean = np.mean(weights)
                if w_mean > 1e-8:
                    weights = weights / w_mean

        sa_onehot = np.zeros((n, self.num_states + self.num_actions), dtype=np.float32)
        for i in range(n):
            s = dataset.states[i]
            a = dataset.actions[i]
            sa_onehot[i, s] = 1.0
            sa_onehot[i, self.num_states + a] = 1.0

        targets = np.array(dataset.next_states, dtype=np.int64)

        x_tensor = torch.tensor(sa_onehot, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        all_member_losses = []

        for k, member_net in enumerate(self.members):
            member_seed = base_seed * 1000 + k * 37 + 7
            torch.manual_seed(member_seed)

            optimizer = optim.Adam(member_net.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss(reduction="none")

            losses = []
            indices = np.arange(n)

            for epoch in range(epochs):
                np.random.shuffle(indices)
                epoch_loss = 0.0
                num_batches = int(np.ceil(n / batch_size))

                for b in range(num_batches):
                    b_idx = indices[b * batch_size : (b + 1) * batch_size]
                    bx = x_tensor[b_idx]
                    by = y_tensor[b_idx]
                    bw = w_tensor[b_idx]

                    optimizer.zero_grad()
                    logits = member_net(bx)
                    unweighted_loss = criterion(logits, by)
                    loss = torch.mean(bw * unweighted_loss)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * len(b_idx)

                losses.append(epoch_loss / n)

            all_member_losses.append(losses)

        return all_member_losses

    def get_transition_matrix(self) -> np.ndarray:
        """Compute posterior mean transition matrix across ensemble members."""
        member_mats = []
        for member_net in self.members:
            member_net.eval()
            p_k = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)
            with torch.no_grad():
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        sa = np.zeros((1, self.num_states + self.num_actions), dtype=np.float32)
                        sa[0, s] = 1.0
                        sa[0, self.num_states + a] = 1.0
                        bx = torch.tensor(sa, dtype=torch.float32, device=self.device)
                        logits = member_net(bx)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        p_k[s, a, :] = probs
            member_mats.append(p_k)

        p_mean = np.mean(member_mats, axis=0)

        # Absorbing goal
        goal_s = self.num_states - 1
        for a in range(self.num_actions):
            p_mean[goal_s, a, :] = 0.0
            p_mean[goal_s, a, goal_s] = 1.0

        return p_mean

    def create_learned_mdp(self, true_mdp: ChoiceGridWorldMDP) -> TabularMDP:
        p_mean = self.get_transition_matrix()
        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=p_mean,
            rewards=true_mdp.rewards.copy(),
            gamma=self.gamma,
            initial_dist=true_mdp.initial_dist.copy(),
        )
