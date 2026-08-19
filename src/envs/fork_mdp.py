"""
Parameterized Stochastic Fork MDP for Oracle Falsification & Matched Error Studies.
"""
from typing import List, Tuple, Optional
import numpy as np
from src.envs.tabular_mdp import TabularMDP


class ForkMDP(TabularMDP):
    """
    Stochastic Fork-MDP with controlled decision branch at initial state s_0.
    
    Structure:
      - s_0: Choice point (a_0 -> branch 0 / Left, a_1 -> branch 1 / Right, ..., a_{M-1})
      - Branch b has `branch_length` intermediate states.
      - Each transition along branch b has success probability p_success[b] (advances to next state)
        and failure probability (1 - p_success[b]) (transitions to s_term or early absorbing).
      - Reaching the end of branch b yields reward R_terminal[b].
      - Intermediate steps incur step_cost.
      - s_term: Absorbing state with 0 reward.
    """

    def __init__(
        self,
        branch_length: int = 2,
        num_branches: int = 2,
        p_success: Optional[List[float]] = None,
        r_terminal: Optional[List[float]] = None,
        step_cost: float = 0.01,
        gamma: float = 0.95,
    ):
        self.branch_length = branch_length
        self.num_branches = num_branches

        if p_success is None:
            # Default: Branch 0 (Left) has p=0.85, Branch 1 (Right) has p=0.75
            p_success = [0.85, 0.75] if num_branches == 2 else [0.85, 0.75, 0.65][:num_branches]
        if r_terminal is None:
            # Default: Branch 0 (Left) has R=1.0, Branch 1 (Right) has R=0.6
            r_terminal = [1.0, 0.6] if num_branches == 2 else [1.0, 0.6, 0.3][:num_branches]

        self.p_success = p_success
        self.r_terminal = r_terminal
        self.step_cost = step_cost

        # States: s_0 (1 state) + num_branches * branch_length (intermediate states) + s_term (1 state)
        num_states = 1 + num_branches * branch_length + 1
        term_state = num_states - 1
        num_actions = max(num_branches, 1)

        transitions = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        rewards = np.zeros((num_states, num_actions), dtype=np.float64)

        state_names = ["s_0"]
        for b in range(num_branches):
            branch_label = "L" if b == 0 else ("R" if b == 1 else f"B{b}")
            for l in range(1, branch_length + 1):
                state_names.append(f"s_{branch_label}{l}")
        state_names.append("s_term")

        action_names = [f"a_branch_{b}" for b in range(num_branches)]

        # Helper to get state index for branch b, step l (1-indexed)
        def branch_state_idx(b: int, l: int) -> int:
            return 1 + b * branch_length + (l - 1)

        # 1. State s_0 (Choice Point)
        for b in range(num_branches):
            first_branch_state = branch_state_idx(b, 1)
            p_s = self.p_success[b]
            transitions[0, b, first_branch_state] = p_s
            transitions[0, b, term_state] = 1.0 - p_s
            rewards[0, b] = -self.step_cost

        # 2. Intermediate branch states
        for b in range(num_branches):
            p_s = self.p_success[b]
            for l in range(1, branch_length + 1):
                s_idx = branch_state_idx(b, l)
                if l < branch_length:
                    next_state = branch_state_idx(b, l + 1)
                    # Action 0: Standard progression
                    transitions[s_idx, 0, next_state] = p_s
                    transitions[s_idx, 0, term_state] = 1.0 - p_s
                    rewards[s_idx, 0] = -self.step_cost

                    # Action 1 (and higher): Alternative transition with different risk/reward
                    # Action 1 has slightly higher transition noise, creating a controlled action gap
                    p_alt = max(0.2, p_s - 0.20)
                    for a in range(1, num_actions):
                        transitions[s_idx, a, next_state] = p_alt
                        transitions[s_idx, a, term_state] = 1.0 - p_alt
                        rewards[s_idx, a] = -self.step_cost - 0.05
                else:
                    # End of branch -> leads to terminal state with final reward
                    for a in range(num_actions):
                        # Action 0 gets full branch reward
                        if a == 0:
                            transitions[s_idx, a, term_state] = 1.0
                            rewards[s_idx, a] = self.r_terminal[b]
                        else:
                            # Action 1 has lower reward
                            transitions[s_idx, a, term_state] = 1.0
                            rewards[s_idx, a] = self.r_terminal[b] * 0.7

        # 3. Terminal state (absorbing)
        for a in range(num_actions):
            transitions[term_state, a, term_state] = 1.0
            rewards[term_state, a] = 0.0

        initial_dist = np.zeros(num_states, dtype=np.float64)
        initial_dist[0] = 1.0

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            transitions=transitions,
            rewards=rewards,
            gamma=gamma,
            initial_dist=initial_dist,
            state_names=state_names,
            action_names=action_names,
        )


def make_fork_mdp(
    branch_length: int = 2,
    p_left: float = 0.85,
    p_right: float = 0.75,
    r_left: float = 1.0,
    r_right: float = 0.85,
    step_cost: float = 0.01,
    gamma: float = 0.95,
    num_branches: int = 2,
) -> ForkMDP:
    """Factory helper to instantiate standard Fork MDPs with custom parameters."""
    if num_branches == 2:
        p_success = [p_left, p_right]
        r_terminal = [r_left, r_right]
    else:
        p_success = [p_left, p_right, 0.65][:num_branches]
        r_terminal = [r_left, r_right, 0.50][:num_branches]

    return ForkMDP(
        branch_length=branch_length,
        num_branches=num_branches,
        p_success=p_success,
        r_terminal=r_terminal,
        step_cost=step_cost,
        gamma=gamma,
    )
