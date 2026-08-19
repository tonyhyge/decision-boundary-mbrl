"""
Learned Neural Categorical Transition World Model for GridWorld MDPs.

Isolates world-model estimation error by fitting a clean MLP on finite trajectory data:
  - Input: One-hot encoded state-action pair (s, a).
  - Output: Categorical distribution over next states s' via softmax.
  - Planning: Exact dynamic programming in learned MDP.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats

from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP
from src.planning.dp import value_iteration
from src.metrics.diagnostics import compute_action_margins, compute_boundary_pressure


class TrajectoryDataset:
    """Stores transitions (s, a, r, s', done) collected from environment."""

    def __init__(self):
        self.states: List[int] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[int] = []
        self.dones: List[bool] = []

    def add(self, s: int, a: int, r: float, next_s: int, done: bool) -> None:
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(next_s)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)


def collect_gridworld_experience(
    mdp: ChoiceGridWorldMDP,
    num_trajectories: int = 80,
    max_steps: int = 40,
    epsilon: float = 0.35,
    seed: Optional[int] = None,
) -> TrajectoryDataset:
    """
    Collect finite exploration trajectories from GridWorld under an epsilon-greedy exploratory policy.
    """
    rng = np.random.default_rng(seed)
    V_star, Q_star, pi_star = value_iteration(mdp)
    dataset = TrajectoryDataset()

    for _ in range(num_trajectories):
        s = 0  # Start at (0,0)
        for _ in range(max_steps):
            if s == mdp.goal_state:
                break

            # Epsilon-greedy exploration
            if rng.uniform() < epsilon:
                a = int(rng.choice(mdp.num_actions))
            else:
                a = int(pi_star[s])

            # Sample next state from true transition dynamics
            p_next = mdp.transitions[s, a, :]
            next_s = int(rng.choice(mdp.num_states, p=p_next))
            r = float(mdp.rewards[s, a])
            done = (next_s == mdp.goal_state)

            dataset.add(s, a, r, next_s, done)
            s = next_s

    return dataset


class CategoricalTransitionNet(nn.Module):
    """Small Multi-Layer Perceptron predicting categorical next-state logits."""

    def __init__(self, num_states: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.in_dim = num_states + num_actions

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states),
        )

    def forward(self, sa_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(sa_onehot)


class LearnedWorldModel:
    """Encapsulates training, inference, and transition matrix extraction for neural world model."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
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
        epochs: int = 150,
        batch_size: int = 64,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> List[float]:
        torch.manual_seed(seed)
        n = len(dataset)
        if n == 0:
            return []

        # Prepare one-hot tensors
        sa_onehot = np.zeros((n, self.num_states + self.num_actions), dtype=np.float32)
        for i in range(n):
            s = dataset.states[i]
            a = dataset.actions[i]
            sa_onehot[i, s] = 1.0
            sa_onehot[i, self.num_states + a] = 1.0

        targets = np.array(dataset.next_states, dtype=np.int64)

        x_tensor = torch.tensor(sa_onehot, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

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

                optimizer.zero_grad()
                logits = self.net(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(b_idx)

            losses.append(epoch_loss / n)

        return losses

    def get_transition_matrix(self) -> np.ndarray:
        """Extract full (S, A, S) simplex transition matrix from trained network."""
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

        # Ensure goal state remains absorbing
        goal_s = self.num_states - 1
        for a in range(self.num_actions):
            p_hat[goal_s, a, :] = 0.0
            p_hat[goal_s, a, goal_s] = 1.0

        return p_hat

    def create_learned_mdp(self, true_mdp: ChoiceGridWorldMDP) -> TabularMDP:
        """Construct TabularMDP using learned dynamics and true reward structure."""
        p_hat = self.get_transition_matrix()
        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=p_hat,
            rewards=true_mdp.rewards.copy(),
            gamma=self.gamma,
            initial_dist=true_mdp.initial_dist.copy(),
        )


def evaluate_estimation_fidelity(
    true_mdp: ChoiceGridWorldMDP,
    learned_mdp: TabularMDP,
) -> Dict[str, float]:
    """
    Sub-Gate G4-A Evaluation:
      - Margin MAE: |m_hat - m_true|
      - Crossing Classification: AUROC & F1-score for Z_cross
      - Boundary Pressure Spearman Rank Correlation
    """
    V_true, Q_true, pi_true = value_iteration(true_mdp)
    V_hat, Q_hat, pi_hat = value_iteration(learned_mdp)

    m_true = compute_action_margins(Q_true, pi_true)
    m_hat = compute_action_margins(Q_hat, pi_hat)

    # 1. Margin MAE across all active states
    active_states = [s for s in range(true_mdp.num_states) if s != true_mdp.goal_state]
    margin_diffs = [abs(m_hat[s, a] - m_true[s, a]) for s in active_states for a in range(true_mdp.num_actions)]
    margin_mae = float(np.mean(margin_diffs))

    # 2. Crossing Detection AUROC
    # True boundary flips: state where learned optimal action differs from true optimal
    y_true_flips = np.array([int(pi_hat[s] != pi_true[s]) for s in active_states])
    
    # Boundary pressure computed from true baseline margin to learned margin
    B_hat = compute_boundary_pressure(m_true, m_hat)
    # Runner-up competitor action has the maximum boundary pressure
    b_scores = np.array([
        max(B_hat[s, a] for a in range(true_mdp.num_actions) if a != int(pi_true[s]))
        for s in active_states
    ])

    # AUROC calculation
    if len(np.unique(y_true_flips)) > 1:
        u_stat, _ = stats.mannwhitneyu(b_scores[y_true_flips == 1], b_scores[y_true_flips == 0], alternative="greater")
        n1 = np.sum(y_true_flips == 1)
        n0 = np.sum(y_true_flips == 0)
        auroc = float(u_stat / (n1 * n0))
    else:
        auroc = 0.50

    # 3. Spearman rank correlation between true margins and learned margins
    m_hat_flat = [m_hat[s, a] for s in active_states for a in range(true_mdp.num_actions)]
    m_true_flat = [m_true[s, a] for s in active_states for a in range(true_mdp.num_actions)]
    rho, p_rho = stats.spearmanr(m_hat_flat, m_true_flat)

    return {
        "margin_mae": margin_mae,
        "crossing_auroc": auroc,
        "boundary_rank_correlation": float(rho) if not np.isnan(rho) else 0.0,
        "fraction_action_agreement": float(np.mean([pi_hat[s] == pi_true[s] for s in active_states])),
    }
