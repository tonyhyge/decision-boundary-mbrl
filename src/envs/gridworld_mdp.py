"""
Stochastic Choice GridWorld MDP Environment.

Features:
  - 2D grid with 4 discrete actions: UP(0), DOWN(1), LEFT(2), RIGHT(3).
  - Multiple alternative routes (safe perimeter vs risky central shortcut).
  - Controllable stochastic transitions (slippage probability).
  - Hazard penalty cells creating sharp value landscapes.
  - Competing near-tie decision states and large-gap corridor states.
  - Absorbing goal state with positive reward.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP


class ChoiceGridWorldMDP(TabularMDP):
    """
    Stochastic 2D Choice GridWorld with competing routes and controllable action gaps.
    """

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        transitions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        initial_distribution: Optional[np.ndarray] = None,
        gamma: float = 0.95,
        goal_pos: Tuple[int, int] = (4, 4),
        hazards: Optional[List[Tuple[int, int]]] = None,
        state_names: Optional[List[str]] = None,
    ):
        self.height = height
        self.width = width
        self.goal_pos = goal_pos
        self.hazards = hazards or []
        self.goal_state = self.pos_to_state(goal_pos[0], goal_pos[1])
        super().__init__(
            num_states=height * width,
            num_actions=4,
            transitions=transitions,
            rewards=rewards,
            gamma=gamma,
            initial_dist=initial_distribution,
            state_names=state_names,
            action_names=["UP", "DOWN", "LEFT", "RIGHT"],
        )

    def pos_to_state(self, r: int, c: int) -> int:
        return r * self.width + c

    def state_to_pos(self, s: int) -> Tuple[int, int]:
        return (s // self.width, s % self.width)


def make_stochastic_choice_gridworld(
    height: int = 5,
    width: int = 5,
    p_succ_safe: float = 0.92,
    p_succ_risky: float = 0.70,
    p_succ_default: float = 0.85,
    hazard_penalty: float = -2.0,
    goal_reward: float = 10.0,
    step_cost: float = -0.05,
    gamma: float = 0.95,
    seed: Optional[int] = None,
) -> ChoiceGridWorldMDP:
    """
    Construct a parameterized Choice GridWorld with 3 competing routes.

    Grid Layout (5x5):
      (0,0): Start
      (4,4): Goal (+10.0, absorbing)
      (1,2), (2,3): Hazard cells (penalty = hazard_penalty)
      Route 1 (North-East safe detour): High reliability (p_succ_safe)
      Route 2 (Central risky shortcut): Low reliability (p_succ_risky) with hazards
      Route 3 (South-West moderate detour): Intermediate reliability (p_succ_default)
    """
    rng = np.random.default_rng(seed)
    num_states = height * width
    num_actions = 4  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

    transitions = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    rewards = np.zeros((num_states, num_actions), dtype=np.float64)
    initial_dist = np.zeros(num_states, dtype=np.float64)
    initial_dist[0] = 1.0  # Start at (0,0)

    goal_r, goal_c = height - 1, width - 1
    goal_state = goal_r * width + goal_c
    hazards = [(1, 2), (2, 3)]
    hazard_states = [r * width + c for r, c in hazards]

    # Action deltas: (dr, dc)
    action_deltas = [
        (-1, 0),  # 0: UP
        (1, 0),   # 1: DOWN
        (0, -1),  # 2: LEFT
        (0, 1),   # 3: RIGHT
    ]

    # Orthogonal slip actions for each action
    orthogonal_actions = {
        0: [2, 3],  # UP -> slip LEFT or RIGHT
        1: [2, 3],  # DOWN -> slip LEFT or RIGHT
        2: [0, 1],  # LEFT -> slip UP or DOWN
        3: [0, 1],  # RIGHT -> slip UP or DOWN
    }

    def get_next_pos(r: int, c: int, a: int) -> Tuple[int, int]:
        dr, dc = action_deltas[a]
        nr = min(max(0, r + dr), height - 1)
        nc = min(max(0, c + dc), width - 1)
        return nr, nc

    for r in range(height):
        for c in range(width):
            s = r * width + c

            # Goal is absorbing
            if s == goal_state:
                for a in range(num_actions):
                    transitions[s, a, s] = 1.0
                    rewards[s, a] = 0.0
                continue

            # Determine cell success probability
            if r <= 1 and c >= 2:  # Safe Northern sector
                p_succ = p_succ_safe
            elif 1 <= r <= 3 and 1 <= c <= 3:  # Central risky sector
                p_succ = p_succ_risky
            else:  # Southern / default sector
                p_succ = p_succ_default

            p_slip_each = (1.0 - p_succ) / 2.0

            for a in range(num_actions):
                # Intended transition
                nr_int, nc_int = get_next_pos(r, c, a)
                ns_int = nr_int * width + nc_int
                transitions[s, a, ns_int] += p_succ

                # Orthogonal slips
                for a_slip in orthogonal_actions[a]:
                    nr_slip, nc_slip = get_next_pos(r, c, a_slip)
                    ns_slip = nr_slip * width + nc_slip
                    transitions[s, a, ns_slip] += p_slip_each

                # Base step reward
                r_val = step_cost

                # Hazard penalty if transitioning into a hazard cell
                # Expected reward over next states
                for ns in range(num_states):
                    if transitions[s, a, ns] > 0:
                        if ns in hazard_states:
                            r_val += transitions[s, a, ns] * hazard_penalty
                        elif ns == goal_state:
                            r_val += transitions[s, a, ns] * goal_reward

                rewards[s, a] = r_val

    # Create descriptive state names
    state_names = [f"Grid_({s // width},{s % width})" for s in range(num_states)]
    state_names[goal_state] = "Goal_(4,4)"

    return ChoiceGridWorldMDP(
        height=height,
        width=width,
        transitions=transitions,
        rewards=rewards,
        initial_distribution=initial_dist,
        gamma=gamma,
        goal_pos=(goal_r, goal_c),
        hazards=hazards,
        state_names=state_names,
    )
