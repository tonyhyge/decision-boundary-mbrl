"""
Matched-Pair Error Generation Protocol.
Generates paired corruptions with matched predictive loss and unsigned sensitivity but opposing margin deformation.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from src.envs.tabular_mdp import TabularMDP
from src.corruptions.injector import LocalizedError, CorruptedMDP
from src.planning.dp import value_iteration, compute_occupancy


@dataclass
class MatchedPair:
    """
    Container for an analytically matched pair of corruptions.
    """
    pair_id: int
    state: int
    action: int
    compressive_error: LocalizedError  # Pushes towards boundary (B > 0)
    expansive_error: LocalizedError    # Pushes away from boundary (B < 0)
    true_gap: float
    occupancy: float
    v_star: np.ndarray


def generate_matched_error_pairs(
    mdp: TabularMDP,
    candidate_states: Optional[List[int]] = None,
    perturbation_magnitudes: Optional[List[float]] = None,
) -> List[MatchedPair]:
    """
    Generate matched pairs of transition corruptions (delta P^+, delta P^-) on an MDP.
    
    For a given state s and action a:
      - We identify the highest-value next state s'_high and lowest-value next state s'_low according to V*_P.
      - A Compressive Perturbation shifts delta mass from s'_high to s'_low, reducing Q(s, a).
        If a == a*, this reduces the margin m(s, a_comp), yielding Delta m < 0 -> B > 0.
      - An Expansive Perturbation shifts delta mass from s'_low to s'_high, increasing Q(s, a).
        If a == a*, this increases the margin m(s, a_comp), yielding Delta m > 0 -> B < 0.
      - Both have EXACTLY equal Total Variation error L1 = delta, equal occupancy, and equal unsigned sensitivity.
    """
    V_star, Q_star, pi_star = value_iteration(mdp)
    _, d_sa = compute_occupancy(mdp, pi_star)

    if candidate_states is None:
        # Avoid the absorbing terminal state
        candidate_states = [s for s in range(mdp.num_states - 1)]

    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    matched_pairs: List[MatchedPair] = []
    pair_id = 0

    for s in candidate_states:
        for a in range(mdp.num_actions):
            true_p = mdp.transitions[s, a, :].copy()
            # Non-zero support next-states
            support = np.where(true_p > 1e-6)[0]
            if len(support) < 2:
                # Cannot construct a non-trivial two-point matched redistribution on single support
                continue

            # Sort reachable next-states by true optimal value V*(s')
            sorted_support = support[np.argsort(V_star[support])]
            s_low = sorted_support[0]
            s_high = sorted_support[-1]

            if V_star[s_high] <= V_star[s_low] + 1e-8:
                continue

            # Compute true action gap at state s
            opt_a = pi_star[s]
            if mdp.num_actions > 1:
                q_vals = Q_star[s, :]
                sorted_q = np.sort(q_vals)[::-1]
                true_gap = float(sorted_q[0] - sorted_q[1])
            else:
                true_gap = 0.0

            occ = float(d_sa[s, a])
            max_delta = min(true_p[s_high], true_p[s_low]) * 0.95

            for delta in perturbation_magnitudes:
                actual_delta = min(delta, max_delta)
                if actual_delta < 1e-4:
                    continue

                # 1. Negative value perturbation (reduces Q(s, a))
                p_neg = true_p.copy()
                p_neg[s_high] -= actual_delta
                p_neg[s_low] += actual_delta
                p_neg = np.maximum(p_neg, 0.0)
                p_neg /= p_neg.sum()

                # 2. Positive value perturbation (increases Q(s, a))
                p_pos = true_p.copy()
                p_pos[s_low] -= actual_delta
                p_pos[s_high] += actual_delta
                p_pos = np.maximum(p_pos, 0.0)
                p_pos /= p_pos.sum()

                l1_neg = 0.5 * float(np.sum(np.abs(p_neg - true_p)))
                l1_pos = 0.5 * float(np.sum(np.abs(p_pos - true_p)))

                mask_neg = (true_p > 1e-12) & (p_neg > 1e-12)
                kl_neg = float(np.sum(true_p[mask_neg] * np.log(true_p[mask_neg] / p_neg[mask_neg])))

                mask_pos = (true_p > 1e-12) & (p_pos > 1e-12)
                kl_pos = float(np.sum(true_p[mask_pos] * np.log(true_p[mask_pos] / p_pos[mask_pos])))

                mse_neg = float(np.mean((p_neg - true_p) ** 2))
                mse_pos = float(np.mean((p_pos - true_p) ** 2))

                is_opt_action = (a == opt_a)
                p_closing = p_neg if is_opt_action else p_pos
                p_opening = p_pos if is_opt_action else p_neg

                l1_closing = l1_neg if is_opt_action else l1_pos
                l1_opening = l1_pos if is_opt_action else l1_neg

                kl_closing = kl_neg if is_opt_action else kl_pos
                kl_opening = kl_pos if is_opt_action else kl_neg

                mse_closing = mse_neg if is_opt_action else mse_pos
                mse_opening = mse_pos if is_opt_action else mse_neg

                err_closing = LocalizedError(
                    error_id=pair_id * 2,
                    state=s,
                    action=a,
                    true_p=true_p,
                    corrupt_p=p_closing,
                    error_l1=l1_closing,
                    error_kl=kl_closing,
                    error_mse=mse_closing,
                )

                err_opening = LocalizedError(
                    error_id=pair_id * 2 + 1,
                    state=s,
                    action=a,
                    true_p=true_p,
                    corrupt_p=p_opening,
                    error_l1=l1_opening,
                    error_kl=kl_opening,
                    error_mse=mse_opening,
                )

                matched_pairs.append(
                    MatchedPair(
                        pair_id=pair_id,
                        state=s,
                        action=a,
                        compressive_error=err_closing,
                        expansive_error=err_opening,
                        true_gap=true_gap,
                        occupancy=occ,
                        v_star=V_star,
                    )
                )
                pair_id += 1

    return matched_pairs
