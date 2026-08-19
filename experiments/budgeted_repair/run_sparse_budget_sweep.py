"""
Stage 3B: Scarce-Budget Error Density Characterization Sweep.

Investigates whether Boundary Pressure (B_i) retains a competitive advantage
specifically under sparse-budget regimes (K/N <= 0.15) across varied error counts N in {8, 14, 20, 28}.
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import value_iteration, compute_occupancy
from src.corruptions.injector import LocalizedError, CorruptedMDP
from src.metrics.diagnostics import (
    compute_action_margins,
    compute_boundary_pressure,
    compute_value_sensitivity,
    compute_advantage_perturbation,
)
from src.baselines.rankers import get_all_rankers
from src.correction.budget import BudgetEvaluator
from src.corruptions.injector import inject_gridworld_multidistribution_errors


def run_stage3b_sparse_sweep(
    error_counts: List[int] = [8, 14, 20, 28],
    num_seeds: int = 20,
    base_seed: int = 142,
    output_dir: str = "results",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rankers = get_all_rankers()
    records = []

    print("================================================================================")
    print("   STAGE 3B: SCARCE-BUDGET ERROR DENSITY SWEEP (N in {8, 14, 20, 28})           ")
    print("================================================================================")

    for n_err in error_counts:
        print(f"\nEvaluating error density N = {n_err} across {num_seeds} seeds...")
        for seed_idx in range(num_seeds):
            seed = base_seed + seed_idx * 10 + n_err
            rng = np.random.default_rng(seed)

            n_unvisited = max(2, int(n_err * 0.28))
            n_large = max(2, int(n_err * 0.36))
            n_bottleneck = max(2, n_err - n_unvisited - n_large)

            grid = make_stochastic_choice_gridworld(height=5, width=5, seed=seed)

            # Classify candidate states
            V_star, Q_star, pi_star = value_iteration(grid)
            d_s, d_sa = compute_occupancy(grid, pi_star)
            m_true = compute_action_margins(Q_star, pi_star)

            unvisited_states = [s for s in range(grid.num_states) if s != grid.goal_state and d_s[s] < 0.005]
            if len(unvisited_states) < n_unvisited:
                sorted_by_d = np.argsort(d_s)
                unvisited_states = [s for s in sorted_by_d if s != grid.goal_state][:n_unvisited + 2]

            visited_states = [s for s in range(grid.num_states) if s != grid.goal_state and d_s[s] >= 0.005]
            def min_comp_gap(s):
                opt_a = int(pi_star[s])
                return min(m_true[s, a] for a in range(grid.num_actions) if a != opt_a)

            visited_sorted = sorted(visited_states, key=min_comp_gap)
            near_tie_states = visited_sorted[:len(visited_sorted) // 2]
            large_gap_states = visited_sorted[len(visited_sorted) // 2:]

            corruptions: List[LocalizedError] = []
            err_id = 0

            # 1. Unvisited Errors
            for s in rng.choice(unvisited_states, size=min(n_unvisited, len(unvisited_states)), replace=False):
                a = int(rng.choice(grid.num_actions))
                true_p = grid.transitions[s, a, :].copy()
                noise = rng.exponential(scale=1.0, size=grid.num_states)
                noise[grid.goal_state] = 0.0
                corrupt_p = true_p + 0.8 * (noise / noise.sum() - true_p)
                corrupt_p = np.maximum(corrupt_p, 0.0)
                corrupt_p /= corrupt_p.sum()
                corruptions.append(LocalizedError(
                    error_id=err_id, state=int(s), action=int(a), true_p=true_p, corrupt_p=corrupt_p,
                    error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))), error_kl=0.0,
                    error_mse=float(np.mean((corrupt_p - true_p) ** 2))
                ))
                err_id += 1

            # 2. Large-gap Visited Errors
            for s in rng.choice(large_gap_states, size=min(n_large, len(large_gap_states)), replace=False):
                opt_a = int(pi_star[s])
                true_p = grid.transitions[s, opt_a, :].copy()
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
                    error_id=err_id, state=int(s), action=int(opt_a), true_p=true_p, corrupt_p=corrupt_p,
                    error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))), error_kl=0.0,
                    error_mse=float(np.mean((corrupt_p - true_p) ** 2))
                ))
                err_id += 1

            # 3. Near-tie Bottleneck Errors
            for s in rng.choice(near_tie_states, size=min(n_bottleneck, len(near_tie_states)), replace=False):
                opt_a = int(pi_star[s])
                comp_actions = [a_c for a_c in range(grid.num_actions) if a_c != opt_a]
                comp_a = comp_actions[int(np.argmin([m_true[s, a_c] for a_c in comp_actions]))]
                true_p = grid.transitions[s, opt_a, :].copy()
                supp = np.where(true_p > 1e-5)[0]
                if len(supp) >= 2:
                    supp_sorted = supp[np.argsort(V_star[supp])]
                    s_low, s_high = supp_sorted[0], supp_sorted[-1]
                    gap = m_true[s, comp_a]
                    v_span = max(1e-5, V_star[s_high] - V_star[s_low])
                    crit_shift = gap / (grid.gamma * v_span)
                    shift = min(true_p[s_high] * 0.9, crit_shift * 1.35)
                    corrupt_p = true_p.copy()
                    corrupt_p[s_high] -= shift
                    corrupt_p[s_low] += shift
                else:
                    corrupt_p = true_p.copy()
                corruptions.append(LocalizedError(
                    error_id=err_id, state=int(s), action=int(opt_a), true_p=true_p, corrupt_p=corrupt_p,
                    error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))), error_kl=0.0,
                    error_mse=float(np.mean((corrupt_p - true_p) ** 2))
                ))
                err_id += 1

            c_mdp = CorruptedMDP(grid, corruptions)
            _, Q_corrupt, pi_corrupt = value_iteration(c_mdp.corrupted_mdp)
            m_corrupt = compute_action_margins(Q_corrupt, pi_corrupt)
            B_all = compute_boundary_pressure(m_true, m_corrupt)

            err_recs = []
            for e_idx, e in enumerate(c_mdp.errors):
                c_val, _, _ = c_mdp.compute_counterfactual_correction_value(e_idx)
                g_s, g_a = compute_value_sensitivity(e.corrupt_p - e.true_p, V_star)
                opt_a = int(pi_star[e.state])
                comp_actions = [a_c for a_c in range(grid.num_actions) if a_c != opt_a]
                gap = min(m_true[e.state, a_c] for a_c in comp_actions)

                delta_q_opt = float(grid.gamma * np.dot(e.corrupt_p - e.true_p, V_star)) if e.action == opt_a else 0.0
                a_s = compute_advantage_perturbation(delta_q_opt, 0.0)
                b_v = float(B_all[e.state, e.action])
                occ_v = float(d_sa[e.state, e.action])

                err_recs.append({
                    "error_l1": e.error_l1,
                    "occupancy": occ_v,
                    "true_action_gap": gap,
                    "value_sensitivity_abs": g_a,
                    "value_sensitivity_signed": g_s,
                    "advantage_sensitivity_signed": a_s,
                    "boundary_pressure": b_v,
                    "correction_value": max(0.0, c_val),
                })

            df_errors = pd.DataFrame(err_recs)
            evaluator = BudgetEvaluator(c_mdp, df_errors)

            n_actual_errors = len(c_mdp.errors)
            for ranker in rankers:
                curve = evaluator.evaluate_ranker(ranker, rng)
                for k in range(n_actual_errors + 1):
                    records.append({
                        "num_errors_N": n_actual_errors,
                        "seed": seed,
                        "ranker": ranker.name,
                        "budget_k": k,
                        "budget_fraction": k / n_actual_errors,
                        "recovery": curve[k],
                    })

    df_sweep = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "stage3b_sparse_budget_sweep.csv")
    df_sweep.to_csv(csv_path, index=False)
    print(f"\n-> Saved Stage 3B sweep data to {csv_path}")

    # Summary table across scarce budgets
    target_budget_fracs = [0.10, 0.15, 0.25, 0.50]
    table_rows = []
    for (n_err, rk_name), grp in df_sweep.groupby(["num_errors_N", "ranker"]):
        if rk_name in ["Oracle (C_i)", "Value Sensitivity (Unsigned |G|)", "Boundary Pressure (B_i)", "Prediction Error (L1)", "Random"]:
            r_dict = {"N": n_err, "Ranker": rk_name}
            for bf in target_budget_fracs:
                sub = grp[np.isclose(grp["budget_fraction"], bf, atol=0.04)]
                r_dict[f"Rec@{int(bf*100)}%"] = float(sub["recovery"].mean()) if not sub.empty else 0.0
            table_rows.append(r_dict)

    df_res = pd.DataFrame(table_rows)
    print("\n--- STAGE 3B: SCARCE-BUDGET COMPARISON SUMMARY ---")
    print(df_res.to_string(index=False))
    return df_sweep


if __name__ == "__main__":
    run_stage3b_sparse_sweep()
