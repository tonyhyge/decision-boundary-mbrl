"""
Stage 3 Adaptive Reranking Benchmark:
Compares static single-pass ranking vs. dynamic sequential reranking
under identical repair budgets across 50 GridWorld configurations.
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import (
    value_iteration,
    policy_evaluation,
    compute_occupancy,
    expected_discounted_return,
)
from src.corruptions.injector import LocalizedError, CorruptedMDP, inject_gridworld_multidistribution_errors
from src.metrics.diagnostics import (
    compute_action_margins,
    compute_boundary_pressure,
    compute_value_sensitivity,
    compute_advantage_perturbation,
)


def evaluate_adaptive_ranker(
    corrupted_grid: CorruptedMDP,
    strategy: str,
    rng: np.random.Generator,
) -> Dict[int, float]:
    """
    Evaluates sequential adaptive reranking:
    At each step k, recomputes the diagnostic metric using the current partially restored model,
    selects the single highest-priority remaining error component, and restores it.
    """
    true_mdp = corrupted_grid.true_mdp
    n_errors = len(corrupted_grid.errors)
    j_true = corrupted_grid.j_true_star
    j_corrupt = corrupted_grid.j_corrupt
    j_delta = j_true - j_corrupt

    results: Dict[int, float] = {0: 0.0}
    if abs(j_delta) < 1e-12:
        return {k: 1.0 for k in range(n_errors + 1)}

    restored_indices: List[int] = []
    remaining_indices = list(range(n_errors))

    # Precompute true MDP values
    V_star, Q_star, pi_star = value_iteration(true_mdp)
    m_true = compute_action_margins(Q_star, pi_star)

    for k in range(1, n_errors + 1):
        # Current partially restored model
        current_corrupted_mdp = corrupted_grid.restore_subset(restored_indices)
        V_curr, Q_curr, pi_curr = value_iteration(current_corrupted_mdp)
        d_s_curr, d_sa_curr = compute_occupancy(current_corrupted_mdp, pi_curr)
        m_curr = compute_action_margins(Q_curr, pi_curr)
        B_curr = compute_boundary_pressure(m_true, m_curr)

        candidate_scores = []
        for e_idx in remaining_indices:
            e = corrupted_grid.errors[e_idx]
            opt_a = int(pi_star[e.state])
            comp_actions = [a_c for a_c in range(true_mdp.num_actions) if a_c != opt_a]
            gap = min(m_true[e.state, a_c] for a_c in comp_actions)
            occ_val = float(d_sa_curr[e.state, e.action])
            b_val = float(B_curr[e.state, e.action])
            g_s, g_a = compute_value_sensitivity(e.corrupt_p - e.true_p, V_star)

            if strategy == "dyn_boundary_pressure":
                score = b_val
            elif strategy == "dyn_occ_boundary_pressure":
                score = occ_val * b_val
            elif strategy == "dyn_value_sensitivity_abs":
                score = g_a
            elif strategy == "dyn_occ_value_sensitivity_abs":
                score = occ_val * g_a
            elif strategy == "dyn_prediction_error":
                score = e.error_l1
            elif strategy == "dyn_occ_prediction_error":
                score = occ_val * e.error_l1
            elif strategy == "dyn_greedy_oracle":
                # Evaluate marginal return of restoring e_idx
                test_subset = restored_indices + [e_idx]
                test_mdp = corrupted_grid.restore_subset(test_subset)
                _, _, test_pi = value_iteration(test_mdp)
                test_j = expected_discounted_return(true_mdp, test_pi)
                score = test_j
            elif strategy == "random":
                score = rng.uniform()
            else:
                raise ValueError(f"Unknown strategy {strategy}")

            candidate_scores.append((score, e_idx))

        # Select candidate with maximum score
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        best_e_idx = candidate_scores[0][1]

        restored_indices.append(best_e_idx)
        remaining_indices.remove(best_e_idx)

        # Evaluate resulting policy
        new_restored_mdp = corrupted_grid.restore_subset(restored_indices)
        _, _, pi_k = value_iteration(new_restored_mdp)
        j_k = expected_discounted_return(true_mdp, pi_k)
        rec = float(np.clip((j_k - j_corrupt) / j_delta, 0.0, 1.0))
        results[k] = rec

    return results


def run_adaptive_benchmark(
    num_trials: int = 50,
    base_seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(base_seed)

    strategies = [
        "dyn_boundary_pressure",
        "dyn_occ_boundary_pressure",
        "dyn_value_sensitivity_abs",
        "dyn_occ_value_sensitivity_abs",
        "dyn_prediction_error",
        "dyn_occ_prediction_error",
        "dyn_greedy_oracle",
        "random",
    ]

    records = []
    trial_auc = []

    print(f"Running Adaptive Repair Benchmark over {num_trials} trials...")

    for trial_idx in range(num_trials):
        seed = base_seed + trial_idx
        t_rng = np.random.default_rng(seed)

        p_safe = float(t_rng.uniform(0.88, 0.94))
        p_risky = float(t_rng.uniform(0.65, 0.75))
        p_def = float(t_rng.uniform(0.80, 0.88))
        h_pen = float(t_rng.uniform(-2.5, -1.5))
        s_cost = float(t_rng.uniform(-0.08, -0.03))

        true_grid = make_stochastic_choice_gridworld(
            height=5,
            width=5,
            p_succ_safe=p_safe,
            p_succ_risky=p_risky,
            p_succ_default=p_def,
            hazard_penalty=h_pen,
            step_cost=s_cost,
            gamma=0.95,
            seed=seed,
        )

        corrupted_grid = inject_gridworld_multidistribution_errors(true_grid, num_errors=14, rng=t_rng)
        n_e = len(corrupted_grid.errors)

        auc_row = {"trial": trial_idx}
        for strat in strategies:
            curve = evaluate_adaptive_ranker(corrupted_grid, strat, t_rng)
            y_vals = [curve[k] for k in range(n_e + 1)]
            auc = float(np.sum(y_vals) / (n_e + 1))
            auc_row[strat] = auc

            for k in range(n_e + 1):
                records.append({
                    "trial": trial_idx,
                    "strategy": strat,
                    "budget_k": k,
                    "budget_fraction": k / n_e,
                    "recovery": curve[k],
                    "auc": auc,
                })

        trial_auc.append(auc_row)

    df_budget = pd.DataFrame(records)
    df_auc = pd.DataFrame(trial_auc)

    df_budget.to_csv(os.path.join(output_dir, "stage3_adaptive_benchmark_50seeds.csv"), index=False)
    df_auc.to_csv(os.path.join(output_dir, "stage3_adaptive_trial_auc_50seeds.csv"), index=False)

    # Statistical summary
    summary_table = []
    budget_fracs = [0.143, 0.286, 0.429, 0.571, 0.714, 1.0]
    for strat in strategies:
        grp = df_budget[df_budget["strategy"] == strat]
        row = {"Strategy": strat}
        for bf in budget_fracs:
            bf_grp = grp[np.isclose(grp["budget_fraction"], bf, atol=0.04)]
            row[f"Recovery@{int(bf*100)}%"] = float(bf_grp["recovery"].mean()) if not bf_grp.empty else 0.0
        row["AUC_Mean"] = float(df_auc[strat].mean())
        row["AUC_Std"] = float(df_auc[strat].std())
        summary_table.append(row)

    df_summary = pd.DataFrame(summary_table).sort_values("AUC_Mean", ascending=False)
    print("\n--- ADAPTIVE REPAIR BENCHMARK RESULTS ---")
    print(df_summary.to_string(index=False))

    # Paired Wilcoxon and Bootstrap CI between Dyn |G| and Dyn B
    auc_dyn_g = df_auc["dyn_value_sensitivity_abs"].to_numpy()
    auc_dyn_b = df_auc["dyn_boundary_pressure"].to_numpy()
    diff_dyn = auc_dyn_g - auc_dyn_b

    w_stat, w_pval = stats.wilcoxon(auc_dyn_g, auc_dyn_b)
    t_stat, t_pval = stats.ttest_rel(auc_dyn_g, auc_dyn_b)

    boot_diffs = [np.mean(rng.choice(diff_dyn, size=len(diff_dyn), replace=True)) for _ in range(2000)]
    ci_95 = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]
    hl_shift = float(np.median(diff_dyn))

    report = {
        "summary_table": summary_table,
        "dyn_g_vs_dyn_b": {
            "mean_diff_auc": float(np.mean(diff_dyn)),
            "hl_median_shift": hl_shift,
            "bootstrap_ci_95": ci_95,
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pval": float(w_pval),
            "ttest_stat": float(t_stat),
            "ttest_pval": float(t_pval),
        }
    }

    with open(os.path.join(output_dir, "stage3_adaptive_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- DYNAMIC |G| vs DYNAMIC B COMPARISON ---")
    print(f"Mean Delta AUC (Dyn |G| - Dyn B): {np.mean(diff_dyn):.4f}")
    print(f"Hodges-Lehmann Median Shift: {hl_shift:.4f}")
    print(f"95% Bootstrap CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"Wilcoxon p-value: {w_pval:.4f}")
    print(f"Paired t-test p-value: {t_pval:.4f}")

    return report


if __name__ == "__main__":
    run_adaptive_benchmark(num_trials=50, base_seed=42)
