r"""
Localized Model Error Injection and Single-Component Counterfactual Restoration.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP
from src.planning.dp import value_iteration, expected_discounted_return


@dataclass
class LocalizedError:
    r"""
    Metadata and vectors for a single localized dynamics error.
    
    Attributes:
        error_id (int): Unique identifier for this error component.
        state (int): Corrupted state index.
        action (int): Corrupted action index.
        true_p (np.ndarray): Shape (|S|,), true transition vector P(· | s, a).
        corrupt_p (np.ndarray): Shape (|S|,), corrupted transition vector \hat{P}(· | s, a).
        error_l1 (float): Total Variation distance 0.5 * ||P - \hat{P}||_1.
        error_kl (float): Forward KL divergence D_KL(P || \hat{P}).
        error_mse (float): Mean Squared Error between transition vectors.
    """
    error_id: int
    state: int
    action: int
    true_p: np.ndarray
    corrupt_p: np.ndarray
    error_l1: float
    error_kl: float
    error_mse: float


class CorruptedMDP:
    r"""
    Manages a true MDP alongside its corrupted counterpart \hat{M}, tracking individual error components.
    """

    def __init__(self, true_mdp: TabularMDP, errors: List[LocalizedError]):
        self.true_mdp = true_mdp
        self.errors = errors

        # Construct the corrupted transition matrix
        corrupt_transitions = np.copy(true_mdp.transitions)
        for err in errors:
            corrupt_transitions[err.state, err.action, :] = err.corrupt_p

        self.corrupted_mdp = true_mdp.with_transitions(corrupt_transitions)

        # Precompute baseline policies and returns
        self.v_true_star, self.q_true_star, self.pi_true_star = value_iteration(self.true_mdp)
        self.j_true_star = expected_discounted_return(self.true_mdp, self.pi_true_star)

        self.v_corrupt_star, self.q_corrupt_star, self.pi_corrupt_star = value_iteration(self.corrupted_mdp)
        self.j_corrupt = expected_discounted_return(self.true_mdp, self.pi_corrupt_star)

    def restore_component(self, error_idx: int) -> TabularMDP:
        r"""
        Create a counterfactual MDP \hat{P}^{, i \leftarrow P} where only component error_idx is restored.
        """
        target_error = self.errors[error_idx]
        restored_transitions = np.copy(self.corrupted_mdp.transitions)
        restored_transitions[target_error.state, target_error.action, :] = target_error.true_p
        return self.true_mdp.with_transitions(restored_transitions)

    def restore_subset(self, error_indices: List[int]) -> TabularMDP:
        """
        Create a counterfactual MDP where a subset of error indices is restored.
        """
        restored_transitions = np.copy(self.corrupted_mdp.transitions)
        for idx in error_indices:
            err = self.errors[idx]
            restored_transitions[err.state, err.action, :] = err.true_p
        return self.true_mdp.with_transitions(restored_transitions)

    def compute_counterfactual_correction_value(self, error_idx: int) -> Tuple[float, float, np.ndarray]:
        r"""
        Compute C_i = J(\pi^*_{\hat{P}^{, i \leftarrow P}}; P) - J(\pi^*_{\hat{P}}; P).
        
        Returns:
            c_i (float): Counterfactual correction value evaluated in the true environment.
            j_corrected (float): Return of the newly planned policy in the true environment.
            pi_corrected (np.ndarray): The policy planned on \hat{P}^{, i \leftarrow P}.
        """
        mdp_single_restored = self.restore_component(error_idx)
        _, _, pi_corrected = value_iteration(mdp_single_restored)
        j_corrected = expected_discounted_return(self.true_mdp, pi_corrected)
        c_i = j_corrected - self.j_corrupt
        return c_i, j_corrected, pi_corrected


def inject_random_corruptions(
    true_mdp: TabularMDP,
    num_corruptions: int,
    noise_scale: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    exclude_absorbing: bool = True,
) -> CorruptedMDP:
    """
    Inject random Dirichlet/Gaussian-perturbed simplex transitions into selected state-action pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Candidate state-actions
    candidates = []
    for s in range(true_mdp.num_states):
        if exclude_absorbing and (s == true_mdp.num_states - 1):
            continue
        for a in range(true_mdp.num_actions):
            candidates.append((s, a))

    chosen_indices = rng.choice(len(candidates), size=min(num_corruptions, len(candidates)), replace=False)
    errors: List[LocalizedError] = []

    for err_id, idx in enumerate(chosen_indices):
        s, a = candidates[idx]
        true_p = true_mdp.transitions[s, a, :].copy()

        # Generate Dirichlet perturbed simplex
        alpha = true_p * 10.0 + 0.1
        # Add random noise vector
        perturb = rng.dirichlet(alpha)
        corrupt_p = (1.0 - noise_scale) * true_p + noise_scale * perturb
        corrupt_p = np.maximum(corrupt_p, 1e-12)
        corrupt_p /= corrupt_p.sum()

        l1_err = 0.5 * float(np.sum(np.abs(corrupt_p - true_p)))
        # Safe KL computation
        mask = (true_p > 1e-12) & (corrupt_p > 1e-12)
        kl_err = float(np.sum(true_p[mask] * np.log(true_p[mask] / corrupt_p[mask])))
        mse_err = float(np.mean((corrupt_p - true_p) ** 2))

        errors.append(
            LocalizedError(
                error_id=err_id,
                state=s,
                action=a,
                true_p=true_p,
                corrupt_p=corrupt_p,
                error_l1=l1_err,
                error_kl=kl_err,
                error_mse=mse_err,
            )
        )

    return CorruptedMDP(true_mdp=true_mdp, errors=errors)


def inject_gridworld_multidistribution_errors(
    mdp,
    num_errors: int = 14,
    rng: Optional[np.random.Generator] = None,
) -> CorruptedMDP:
    """
    Inject structured multi-distribution corruptions across GridWorld states:
      - 4 errors at unvisited / low-occupancy states (high L1 error, NC3 control).
      - 5 errors at large-gap visited corridor states (moderate L1 error, sub-threshold, NC1 control).
      - 5 errors at near-tie bottleneck choice states (small L1 error, boundary crossing).
    """
    from src.planning.dp import compute_occupancy
    from src.metrics.diagnostics import compute_action_margins

    if rng is None:
        rng = np.random.default_rng(42)

    V_star, Q_star, pi_star = value_iteration(mdp)
    d_s, d_sa = compute_occupancy(mdp, pi_star)
    margins = compute_action_margins(Q_star, pi_star)

    # Classify candidate states
    unvisited_states = [s for s in range(mdp.num_states) if s != mdp.goal_state and d_s[s] < 0.005]
    if len(unvisited_states) < 3:
        sorted_by_d = np.argsort(d_s)
        unvisited_states = [s for s in sorted_by_d if s != mdp.goal_state][:6]

    visited_states = [s for s in range(mdp.num_states) if s != mdp.goal_state and d_s[s] >= 0.005]
    
    def min_comp_gap(s):
        opt_a = int(pi_star[s])
        return min(margins[s, a] for a in range(mdp.num_actions) if a != opt_a)

    visited_sorted = sorted(visited_states, key=min_comp_gap)
    near_tie_states = visited_sorted[:len(visited_sorted) // 2]
    large_gap_states = visited_sorted[len(visited_sorted) // 2:]

    corruptions: List[LocalizedError] = []
    err_id = 0

    # 1. Unvisited Errors (Large error, d ~ 0)
    for s in rng.choice(unvisited_states, size=min(4, len(unvisited_states)), replace=False):
        a = int(rng.choice(mdp.num_actions))
        true_p = mdp.transitions[s, a, :].copy()
        
        noise = rng.exponential(scale=1.0, size=mdp.num_states)
        noise[mdp.goal_state] = 0.0
        corrupt_p = true_p + 0.8 * (noise / noise.sum() - true_p)
        corrupt_p = np.maximum(corrupt_p, 0.0)
        corrupt_p /= corrupt_p.sum()

        corruptions.append(LocalizedError(
            error_id=err_id,
            state=int(s),
            action=int(a),
            true_p=true_p,
            corrupt_p=corrupt_p,
            error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))),
            error_kl=0.0,
            error_mse=float(np.mean((corrupt_p - true_p) ** 2)),
        ))
        err_id += 1

    # 2. Large-gap Visited Errors (Moderate error, pre-boundary)
    for s in rng.choice(large_gap_states, size=min(5, len(large_gap_states)), replace=False):
        opt_a = int(pi_star[s])
        a = opt_a
        true_p = mdp.transitions[s, a, :].copy()
        
        supp = np.where(true_p > 1e-5)[0]
        if len(supp) >= 2:
            supp_sorted = supp[np.argsort(V_star[supp])]
            s_low, s_high = supp_sorted[0], supp_sorted[-1]
            shift = min(true_p[s_high] * 0.4, 0.25)
            corrupt_p = true_p.copy()
            corrupt_p[s_high] -= shift
            corrupt_p[s_low] += shift
        else:
            corrupt_p = true_p.copy()

        corruptions.append(LocalizedError(
            error_id=err_id,
            state=int(s),
            action=int(a),
            true_p=true_p,
            corrupt_p=corrupt_p,
            error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))),
            error_kl=0.0,
            error_mse=float(np.mean((corrupt_p - true_p) ** 2)),
        ))
        err_id += 1

    # 3. Near-tie Bottleneck Errors (Small error, boundary-crossing)
    for s in rng.choice(near_tie_states, size=min(5, len(near_tie_states)), replace=False):
        opt_a = int(pi_star[s])
        comp_actions = [a_c for a_c in range(mdp.num_actions) if a_c != opt_a]
        comp_a = comp_actions[int(np.argmin([margins[s, a_c] for a_c in comp_actions]))]
        
        true_p = mdp.transitions[s, opt_a, :].copy()
        supp = np.where(true_p > 1e-5)[0]
        if len(supp) >= 2:
            supp_sorted = supp[np.argsort(V_star[supp])]
            s_low, s_high = supp_sorted[0], supp_sorted[-1]
            gap = margins[s, comp_a]
            v_span = max(1e-5, V_star[s_high] - V_star[s_low])
            crit_shift = gap / (mdp.gamma * v_span)
            shift = min(true_p[s_high] * 0.9, crit_shift * 1.35)
            corrupt_p = true_p.copy()
            corrupt_p[s_high] -= shift
            corrupt_p[s_low] += shift
        else:
            corrupt_p = true_p.copy()

        corruptions.append(LocalizedError(
            error_id=err_id,
            state=int(s),
            action=int(opt_a),
            true_p=true_p,
            corrupt_p=corrupt_p,
            error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))),
            error_kl=0.0,
            error_mse=float(np.mean((corrupt_p - true_p) ** 2)),
        ))
        err_id += 1

    return CorruptedMDP(true_mdp=mdp, errors=corruptions)
