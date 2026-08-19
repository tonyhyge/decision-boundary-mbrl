"""
Tabular MDP base class with exact probability simplex verification and transition utilities.
"""
from typing import List, Optional
import numpy as np


class TabularMDP:
    """
    Exact finite-state finite-action Markov Decision Process.
    
    Attributes:
        num_states (int): Number of discrete states |S|.
        num_actions (int): Number of discrete actions |A|.
        transitions (np.ndarray): Shape (|S|, |A|, |S|), P(s' | s, a).
        rewards (np.ndarray): Shape (|S|, |A|), expected immediate reward R(s, a).
        gamma (float): Discount factor in [0, 1).
        initial_dist (np.ndarray): Shape (|S|,), initial state distribution mu_0.
        state_names (List[str]): Optional human-readable state names.
        action_names (List[str]): Optional human-readable action names.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        transitions: np.ndarray,
        rewards: np.ndarray,
        gamma: float = 0.95,
        initial_dist: Optional[np.ndarray] = None,
        state_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.transitions = np.array(transitions, dtype=np.float64)
        self.rewards = np.array(rewards, dtype=np.float64)
        self.gamma = float(gamma)

        if initial_dist is None:
            self.initial_dist = np.zeros(self.num_states, dtype=np.float64)
            self.initial_dist[0] = 1.0
        else:
            self.initial_dist = np.array(initial_dist, dtype=np.float64)

        self.state_names = state_names or [f"s_{i}" for i in range(self.num_states)]
        self.action_names = action_names or [f"a_{j}" for j in range(self.num_actions)]

        self.validate()

    def validate(self, tol: float = 1e-6) -> None:
        """Validate shapes and simplex probability constraints."""
        assert self.transitions.shape == (
            self.num_states,
            self.num_actions,
            self.num_states,
        ), f"Invalid transitions shape: {self.transitions.shape}"
        assert self.rewards.shape == (
            self.num_states,
            self.num_actions,
        ), f"Invalid rewards shape: {self.rewards.shape}"
        assert (
            self.initial_dist.shape == (self.num_states,)
        ), f"Invalid initial_dist shape: {self.initial_dist.shape}"
        assert 0.0 <= self.gamma < 1.0, f"Discount gamma must be in [0, 1), got {self.gamma}"

        # Non-negativity
        assert np.all(self.transitions >= -tol), "Negative transition probabilities detected"
        assert np.all(self.initial_dist >= -tol), "Negative initial probabilities detected"

        # Simplex normalization
        row_sums = self.transitions.sum(axis=-1)
        assert np.allclose(
            row_sums, 1.0, atol=tol
        ), f"Transition probabilities must sum to 1.0 along last axis. Max deviation: {np.max(np.abs(row_sums - 1.0))}"

        init_sum = self.initial_dist.sum()
        assert np.allclose(
            init_sum, 1.0, atol=tol
        ), f"Initial distribution must sum to 1.0. Sum: {init_sum}"

    def copy(self) -> "TabularMDP":
        """Create a deep copy of the MDP."""
        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=np.copy(self.transitions),
            rewards=np.copy(self.rewards),
            gamma=self.gamma,
            initial_dist=np.copy(self.initial_dist),
            state_names=list(self.state_names),
            action_names=list(self.action_names),
        )

    def with_transitions(self, new_transitions: np.ndarray) -> "TabularMDP":
        """Return a new TabularMDP with updated transition dynamics."""
        return TabularMDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transitions=new_transitions,
            rewards=np.copy(self.rewards),
            gamma=self.gamma,
            initial_dist=np.copy(self.initial_dist),
            state_names=list(self.state_names),
            action_names=list(self.action_names),
        )
