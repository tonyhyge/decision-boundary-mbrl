"""
Stage 4 Data-Scaling & Estimation-Noise Sweep Pipeline.

Sweeps exploration dataset size |D| in {20, 40, 80, 160, 320} trajectories to evaluate:
  1. Does boundary-crossing classification AUROC improve with data?
  2. Does margin rank correlation rho improve with data?
  3. Does continuous incremental Delta R^2 recover, or does it remain structurally fragile?
  4. How does early-budget recovery Recovery@14%(B_hat) scale with model data?
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import value_iteration, compute_occupancy
from src.models.tabular_learned_model import (
    collect_gridworld_experience,
    LearnedWorldModel,
    evaluate_estimation_fidelity,
)
from src.metrics.diagnostics import (
    compute_action_margins,
    compute_boundary_pressure,
    compute_value_sensitivity,
    evaluate_incremental_r2,
)
from src.baselines.rankers import BaseRanker, EstimatedRanker
from src.correction.budget import BudgetEvaluator
from src.corruptions.injector import inject_gridworld_multidistribution_errors


def run_stage4_data_scaling_sweep(
    traj_counts: List[int] = [20, 40, 80, 160, 320],
    num_seeds: int = 15,
    base_seed: int = 242,
    output_dir: str = "results",
    figure_dir: str = "figures",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 4: DATA-SCALING & ESTIMATION-NOISE SWEEP (|D| in {20..320} trajs)      ")
    print("================================================================================")

    records: List[Dict] = []

    for n_trajs in traj_counts:
        print(f"\nEvaluating dataset size |D| = {n_trajs} trajectories across {num_seeds} seeds...")
        for seed_idx in range(num_seeds):
            seed = base_seed + seed_idx * 10 + n_trajs
            rng = np.random.default_rng(seed)

            # Ground truth MDP
            true_grid = make_stochastic_choice_gridworld(height=5, width=5, seed=seed)
            V_star, Q_star, pi_star = value_iteration(true_grid)
            _, d_sa = compute_occupancy(true_grid, pi_star)
            m_true = compute_action_margins(Q_star, pi_star)

            # Collect experience with specified trajectory budget
            dataset = collect_gridworld_experience(true_grid, num_trajectories=n_trajs, max_steps=40, seed=seed)

            # Train model
            world_model = LearnedWorldModel(num_states=25, num_actions=4, gamma=0.95)
            epochs = min(150, max(60, int(3000 / (len(dataset) + 1))))
            losses = world_model.fit(dataset, epochs=epochs, lr=0.01, seed=seed)
            learned_mdp = world_model.create_learned_mdp(true_grid)

            V_hat, Q_hat, pi_hat = value_iteration(learned_mdp)
            m_hat = compute_action_margins(Q_hat, pi_hat)

            # G4-A Fidelity
            fidelity = evaluate_estimation_fidelity(true_grid, learned_mdp)

            # Corruptions
            corrupted_grid = inject_gridworld_multidistribution_errors(true_grid, num_errors=14, rng=rng)
            p_hat_corrupt = learned_mdp.transitions.copy()
            for e in corrupted_grid.errors:
                p_hat_corrupt[e.state, e.action, :] = e.corrupt_p

            corrupted_learned_mdp = TabularMDP(25, 4, p_hat_corrupt, learned_mdp.rewards, learned_mdp.gamma, learned_mdp.initial_dist)
            _, Q_hat_c, pi_hat_c = value_iteration(corrupted_learned_mdp)
            m_hat_c = compute_action_margins(Q_hat_c, pi_hat_c)
            B_hat_all = compute_boundary_pressure(m_hat, m_hat_c)

            err_recs = []
            for e_idx, e in enumerate(corrupted_grid.errors):
                c_val, _, _ = corrupted_grid.compute_counterfactual_correction_value(e_idx)
                p_hat_s_a = learned_mdp.transitions[e.state, e.action, :]
                delta_p_hat = e.corrupt_p - p_hat_s_a
                _, g_a_hat = compute_value_sensitivity(delta_p_hat, V_hat)
                b_hat_val = float(B_hat_all[e.state, e.action])
                occ_hat = float(d_sa[e.state, e.action])

                err_recs.append({
                    "error_l1_hat": 0.5 * float(np.sum(np.abs(delta_p_hat))),
                    "value_sensitivity_abs_hat": g_a_hat,
                    "boundary_pressure_hat": b_hat_val,
                    "occ_boundary_pressure_hat": occ_hat * b_hat_val,
                    "correction_value": max(0.0, c_val),
                })

            df_errors = pd.DataFrame(err_recs)

            # G4-B Incremental R^2
            r2_stats = evaluate_incremental_r2(
                df=df_errors,
                target_col="correction_value",
                control_cols=["error_l1_hat", "value_sensitivity_abs_hat"],
                proposed_col="boundary_pressure_hat",
            )

            # G4-C Budget Recovery@14% (K=2/14)
            evaluator = BudgetEvaluator(corrupted_grid, df_errors)
            rk_b = EstimatedRanker("B_hat", "boundary_pressure_hat", ascending=False)
            rk_e = EstimatedRanker("E_hat", "error_l1_hat", ascending=False)
            rk_g = EstimatedRanker("G_hat", "value_sensitivity_abs_hat", ascending=False)

            curve_b = evaluator.evaluate_ranker(rk_b, rng)
            curve_e = evaluator.evaluate_ranker(rk_e, rng)
            curve_g = evaluator.evaluate_ranker(rk_g, rng)

            rec14_b = curve_b[2] if len(curve_b) > 2 else 0.0
            rec14_e = curve_e[2] if len(curve_e) > 2 else 0.0
            rec14_g = curve_g[2] if len(curve_g) > 2 else 0.0

            records.append({
                "num_trajectories": n_trajs,
                "seed": seed,
                "margin_mae": fidelity["margin_mae"],
                "crossing_auroc": fidelity["crossing_auroc"],
                "margin_rank_rho": fidelity["boundary_rank_correlation"],
                "action_agreement": fidelity["fraction_action_agreement"],
                "delta_r2_b_hat": r2_stats["delta_r2"],
                "rec14_b_hat": rec14_b,
                "rec14_e_hat": rec14_e,
                "rec14_g_hat": rec14_g,
            })

    df_scaling = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "stage4_data_scaling_sweep.csv")
    df_scaling.to_csv(csv_path, index=False)
    print(f"\n-> Saved Data-Scaling Sweep dataset to {csv_path}")

    # Summary table
    grouped = df_scaling.groupby("num_trajectories").agg({
        "crossing_auroc": ["mean", "std"],
        "margin_rank_rho": ["mean", "std"],
        "action_agreement": ["mean", "std"],
        "delta_r2_b_hat": ["mean", "std"],
        "rec14_b_hat": ["mean", "std"],
        "rec14_e_hat": ["mean", "std"],
    }).reset_index()

    print("\n--- STAGE 4 DATA-SCALING SUMMARY TABLE ---")
    print(grouped.to_string(index=False))

    # Generate Publication Figure
    generate_scaling_figures(df_scaling, figure_dir)

    return df_scaling


def generate_scaling_figures(df_scaling: pd.DataFrame, figure_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    
    trajs = np.sort(df_scaling["num_trajectories"].unique())

    # (a) Crossing AUROC vs |D|
    auroc_means = [df_scaling[df_scaling["num_trajectories"] == t]["crossing_auroc"].mean() for t in trajs]
    auroc_stds = [df_scaling[df_scaling["num_trajectories"] == t]["crossing_auroc"].std() for t in trajs]
    axes[0].errorbar(trajs, auroc_means, yerr=auroc_stds, fmt="o-", color="#4C72B0", linewidth=2, capsize=4)
    axes[0].set_xlabel("Exploration Trajectories $|\\mathcal{D}|$")
    axes[0].set_ylabel("Crossing AUROC")
    axes[0].set_title("(a) Crossing Classification AUROC")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Margin Rank Correlation vs |D|
    rho_means = [df_scaling[df_scaling["num_trajectories"] == t]["margin_rank_rho"].mean() for t in trajs]
    rho_stds = [df_scaling[df_scaling["num_trajectories"] == t]["margin_rank_rho"].std() for t in trajs]
    axes[1].errorbar(trajs, rho_means, yerr=rho_stds, fmt="s-", color="#55A868", linewidth=2, capsize=4)
    axes[1].set_xlabel("Exploration Trajectories $|\\mathcal{D}|$")
    axes[1].set_ylabel("Rank Correlation $\\rho(\\hat{m}, m)$")
    axes[1].set_title("(b) Margin Rank Correlation")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # (c) Incremental Delta R^2 vs |D|
    r2_means = [df_scaling[df_scaling["num_trajectories"] == t]["delta_r2_b_hat"].mean() for t in trajs]
    r2_stds = [df_scaling[df_scaling["num_trajectories"] == t]["delta_r2_b_hat"].std() for t in trajs]
    axes[2].errorbar(trajs, r2_means, yerr=r2_stds, fmt="^-", color="#C44E52", linewidth=2, capsize=4)
    axes[2].set_xlabel("Exploration Trajectories $|\\mathcal{D}|$")
    axes[2].set_ylabel("Incremental $\\Delta R^2(\\hat{B})$")
    axes[2].set_title("(c) Continuous Explanatory $\\Delta R^2$")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # (d) Recovery@14% vs |D|
    rec_b_means = [df_scaling[df_scaling["num_trajectories"] == t]["rec14_b_hat"].mean() for t in trajs]
    rec_e_means = [df_scaling[df_scaling["num_trajectories"] == t]["rec14_e_hat"].mean() for t in trajs]
    axes[3].plot(trajs, rec_b_means, "o-", color="#d62728", linewidth=2, label="Estimated $\\hat{B}$")
    axes[3].plot(trajs, rec_e_means, "s--", color="#1f77b4", linewidth=2, label="Estimated $\\hat{E}_{L1}$")
    axes[3].set_xlabel("Exploration Trajectories $|\\mathcal{D}|$")
    axes[3].set_ylabel("Recovery@14% ($K=2$)")
    axes[3].set_title("(d) Scarce-Budget Return Recovery")
    axes[3].legend(loc="lower right", fontsize=9.5)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage4_data_scaling_curves.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Data-Scaling Figure to {fig_path}")


if __name__ == "__main__":
    run_stage4_data_scaling_sweep()
