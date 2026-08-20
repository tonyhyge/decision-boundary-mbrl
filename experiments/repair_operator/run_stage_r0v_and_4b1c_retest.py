"""
Stage R0-V (Exact-Checkpoint Validity Closure) & Stage 4B.1-C (Operator-Normalized Acquisition Re-Test).

Phase 1 (Stage R0-V):
  Verifies that on the EXACT Stage 4B.1-R data/checkpoint configuration, the converged operator
  eliminates the old schedule degradation on D0-only retraining.

Phase 2 (Stage 4B.1-C):
  Re-tests active acquisition strategies under the standardized, frozen, non-degrading repair operator
  across N = 10 independent seeds with matched query budget B = 200.

Strategies Evaluated:
  1. D0-Only Retraining Reference (Zero new data, standard converged operator)
  2. Random Acquisition (B = 200)
  3. Action-Gradient Disagreement sigma_grad_Q (B = 200)
  4. Dynamics Uncertainty sigma_dyn (B = 200)
  5. Privileged Residual Acquisition (B = 200)
  6. Full-Pool Coverage Reference (N = 807)
"""
import os
import sys
import json
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.active_mbpo.run_stage4b1_one_shot_repair import (
    HopperEnsembleDynamics,
    HopperHeuristicCritic,
    collect_hopper_trajectories,
    evaluate_rollout_error_h5,
)
from experiments.active_mbpo.run_stage4b1r_repairability_closure import evaluate_stratified_errors


def run_stage_r0v_and_4b1c(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    budget_count: int = 200,
    n_seeds: int = 10,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE R0-V & STAGE 4B.1-C: OPERATOR-NORMALIZED RE-TEST BENCHMARK             ")
    print("================================================================================")

    env = gym.make("Hopper-v4")

    # -------------------------------------------------------------------------
    # 1. EXACT DATASET INITIALIZATION (Matching 4B.1-R setup)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing exact baseline dataset D_0 (N_0 = 830) & Candidate Pool (N=807)...")
    base_trajs = collect_hopper_trajectories(env, num_trajectories=35, max_steps=120, seed=seed)
    x_base, y_base = [], []
    for tr in base_trajs:
        for t in range(len(tr["obs"])):
            x_base.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_base.append(tr["next_obs"][t] - tr["obs"][t])
    x_base = np.array(x_base, dtype=np.float32)
    y_base = np.array(y_base, dtype=np.float32)
    n_d0 = len(x_base)

    # Initial Reference World Model M_0
    base_ensemble = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed)
    base_ensemble.train_ensemble(x_base, y_base, epochs=50, batch_size=64, lr=0.001, seed=seed)

    # Candidate Pool and Evaluation Trajectories
    pool_trajs = collect_hopper_trajectories(env, num_trajectories=30, max_steps=100, seed=seed + 100)
    eval_trajs = collect_hopper_trajectories(env, num_trajectories=25, max_steps=100, seed=seed + 200)

    pool_samples = []
    for tr in pool_trajs:
        for t in range(len(tr["obs"])):
            pool_samples.append({
                "obs": tr["obs"][t],
                "act": tr["acts"][t],
                "next_obs": tr["next_obs"][t],
                "qpos": tr["qpos"][t],
                "qvel": tr["qvel"][t],
            })
    df_pool = pd.DataFrame(pool_samples)
    n_pool = len(df_pool)

    base_eval_metrics = evaluate_stratified_errors(base_ensemble, eval_trajs, horizon=5)
    base_h5 = base_eval_metrics["e_h5"]
    print(f"-> Base Dataset D_0: {n_d0} samples | Candidate Pool: {n_pool} states | Eval Set: {sum(len(tr['obs']) for tr in eval_trajs)} states.")
    print(f"-> Reference Model M_0 Baseline H=5 Rollout Error: {base_h5:.6f}")

    # -------------------------------------------------------------------------
    # 2. STAGE R0-V: EXACT-CHECKPOINT VALIDITY CLOSURE
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   STAGE R0-V: EXACT-CHECKPOINT VALIDITY CLOSURE (D_0-ONLY REPAIR)             ")
    print("================================================================================")

    # Test 1: Old schedule operator (epochs=40, lr=1e-3, scratch initialization without optimal convergence)
    # Test 2: Normalized Converged Operator (epochs=40, lr=3e-3 on D_0)
    old_h5_list = []
    norm_h5_list = []

    for s_idx in range(n_seeds):
        seed_val = seed + s_idx * 19

        # Old schedule operator
        ens_old = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
        ens_old.train_ensemble(x_base, y_base, epochs=40, batch_size=64, lr=0.001, seed=seed_val)
        ev_old = evaluate_stratified_errors(ens_old, eval_trajs, horizon=5)
        old_h5_list.append(ev_old["e_h5"])

        # Normalized converged operator
        ens_norm = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
        ens_norm.train_ensemble(x_base, y_base, epochs=40, batch_size=64, lr=0.003, seed=seed_val)
        ev_norm = evaluate_stratified_errors(ens_norm, eval_trajs, horizon=5)
        norm_h5_list.append(ev_norm["e_h5"])

    mean_old_h5 = float(np.mean(old_h5_list))
    mean_norm_h5 = float(np.mean(norm_h5_list))
    delta_old_pct = (mean_old_h5 - base_h5) / base_h5 * 100
    delta_norm_pct = (mean_norm_h5 - base_h5) / base_h5 * 100

    t_r0v, p_r0v = stats.ttest_1samp(norm_h5_list, base_h5)
    r0v_passed = bool(mean_norm_h5 < base_h5 and p_r0v < 0.05)

    print(f"  Old Schedule Operator on D0-only  : H=5 Error = {mean_old_h5:.6f} +/- {np.std(old_h5_list):.6f} ({delta_old_pct:+.2f}%)")
    print(f"  Normalized Converged Operator on D0: H=5 Error = {mean_norm_h5:.6f} +/- {np.std(norm_h5_list):.6f} ({delta_norm_pct:+.2f}%)")
    print(f"  R0-V Validity Gate (Normalized < M0 Baseline): {r0v_passed} (t = {t_r0v:.3f}, p = {p_r0v:.4e})")

    # -------------------------------------------------------------------------
    # 3. EXTRACT UNPRIVILEGED SIGNALS & PRIVILEGED RESIDUALS ON CANDIDATE POOL
    # -------------------------------------------------------------------------
    print("\n[Step 3] Extracting Candidate Pool Diagnostic Scores...")
    critic = HopperHeuristicCritic(ensemble=base_ensemble)

    sigma_dyn_list = []
    sigma_grad_q_list = []
    true_res_list = []

    for _, row in df_pool.iterrows():
        s = row["obs"]
        a = row["act"]
        true_ns = row["next_obs"]

        # Dynamics Uncertainty
        pred_all = base_ensemble.predict_all(s, a)
        mean_pred = np.mean(pred_all, axis=0)
        sigma_dyn_list.append(float(np.mean(np.std(pred_all, axis=0, ddof=1))))

        # Action-Gradient Disagreement
        grads_k = critic.estimate_action_gradient(s, a)
        sigma_grad_q_list.append(float(np.mean(np.std(grads_k, axis=0, ddof=1))))

        # Privileged Residual
        true_res_list.append(float(np.linalg.norm(true_ns - mean_pred)))

    df_pool["sigma_dyn"] = np.array(sigma_dyn_list)
    df_pool["sigma_grad_q"] = np.array(sigma_grad_q_list)
    df_pool["true_residual"] = np.array(true_res_list)

    # -------------------------------------------------------------------------
    # 4. STAGE 4B.1-C: OPERATOR-NORMALIZED ACQUISITION RE-TEST (N = 10 SEEDS)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print(f"   STAGE 4B.1-C: OPERATOR-NORMALIZED ACQUISITION RE-TEST (B = {budget_count}, N = {n_seeds})   ")
    print("================================================================================")

    strategies = [
        "1_d0_only_converged",
        "2_random_acquisition",
        "3_action_gradient_disagreement",
        "4_dynamics_uncertainty",
        "5_privileged_residual",
        "6_full_pool_reference",
    ]

    h5_by_strat = {s: [] for s in strategies}
    h1_by_strat = {s: [] for s in strategies}

    for s_idx in range(n_seeds):
        seed_val = seed + s_idx * 37
        rng_s = np.random.default_rng(seed_val)

        # 1. D0-Only (no new data)
        ens_d0 = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
        ens_d0.train_ensemble(x_base, y_base, epochs=40, batch_size=64, lr=0.003, seed=seed_val)
        ev_d0 = evaluate_stratified_errors(ens_d0, eval_trajs, horizon=5)
        h5_by_strat["1_d0_only_converged"].append(ev_d0["e_h5"])
        h1_by_strat["1_d0_only_converged"].append(ev_d0["e_h1"])

        # Query indices for active strategies
        idx_rand = rng_s.choice(n_pool, size=budget_count, replace=False)
        idx_grad = np.argsort(-df_pool["sigma_grad_q"].to_numpy())[:budget_count]
        idx_dyn = np.argsort(-df_pool["sigma_dyn"].to_numpy())[:budget_count]
        idx_oracle = np.argsort(-df_pool["true_residual"].to_numpy())[:budget_count]
        idx_full = np.arange(n_pool)

        strat_query_map = {
            "2_random_acquisition": idx_rand,
            "3_action_gradient_disagreement": idx_grad,
            "4_dynamics_uncertainty": idx_dyn,
            "5_privileged_residual": idx_oracle,
            "6_full_pool_reference": idx_full,
        }

        for strat_name, q_idx in strat_query_map.items():
            # Query simulator transitions
            x_q, y_q = [], []
            for i in q_idx:
                row = df_pool.iloc[i]
                env.reset()
                env.unwrapped.set_state(row["qpos"], row["qvel"])
                s_curr = env.unwrapped._get_obs()
                a_curr = row["act"]
                next_s, _, _, _, _ = env.step(a_curr)
                x_q.append(np.concatenate([s_curr, a_curr]))
                y_q.append(next_s - s_curr)

            x_aug = np.concatenate([x_base, np.array(x_q, dtype=np.float32)], axis=0)
            y_aug = np.concatenate([y_base, np.array(y_q, dtype=np.float32)], axis=0)

            # Train with EXACT same normalized operator (epochs=40, lr=3e-3, batch=64)
            ens_aug = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
            ens_aug.train_ensemble(x_aug, y_aug, epochs=40, batch_size=64, lr=0.003, seed=seed_val)

            ev_aug = evaluate_stratified_errors(ens_aug, eval_trajs, horizon=5)
            h5_by_strat[strat_name].append(ev_aug["e_h5"])
            h1_by_strat[strat_name].append(ev_aug["e_h1"])

    # -------------------------------------------------------------------------
    # 5. STATISTICAL COMPARISON & CAUSAL DECOMPOSITION
    # -------------------------------------------------------------------------
    summary_4b1c = []
    print("\n--- STAGE 4B.1-C SUMMARY RESULTS (N = 10 SEEDS, NORMALIZED OPERATOR) ---")

    mean_d0_h5 = float(np.mean(h5_by_strat["1_d0_only_converged"]))

    for s in strategies:
        m_h5 = float(np.mean(h5_by_strat[s]))
        s_h5 = float(np.std(h5_by_strat[s]))
        m_h1 = float(np.mean(h1_by_strat[s]))
        delta_m0_pct = (m_h5 - base_h5) / base_h5 * 100
        delta_d0_pct = (m_h5 - mean_d0_h5) / mean_d0_h5 * 100
        delta_acq = m_h5 - mean_d0_h5

        print(f"  {s:<32}: H=5 = {m_h5:.6f} +/- {s_h5:.6f} (vs M0: {delta_m0_pct:+.2f}%, vs D0-only: {delta_d0_pct:+.2f}%) | 1-Step = {m_h1:.6f}")

        summary_4b1c.append({
            "strategy": s,
            "mean_h5": m_h5,
            "std_h5": s_h5,
            "delta_m0_pct": delta_m0_pct,
            "delta_d0_pct": delta_d0_pct,
            "delta_acq": delta_acq,
            "mean_h1": m_h1,
        })

    # Paired Hypothesis Tests
    t_grad_vs_rand, p_grad_vs_rand = stats.ttest_rel(
        h5_by_strat["3_action_gradient_disagreement"], h5_by_strat["2_random_acquisition"]
    )
    t_grad_vs_d0, p_grad_vs_d0 = stats.ttest_rel(
        h5_by_strat["3_action_gradient_disagreement"], h5_by_strat["1_d0_only_converged"]
    )
    t_rand_vs_d0, p_rand_vs_d0 = stats.ttest_rel(
        h5_by_strat["2_random_acquisition"], h5_by_strat["1_d0_only_converged"]
    )
    t_priv_vs_d0, p_priv_vs_d0 = stats.ttest_rel(
        h5_by_strat["5_privileged_residual"], h5_by_strat["1_d0_only_converged"]
    )

    print("\n--- PAIRED HYPOTHESIS TESTS (STAGE 4B.1-C) ---")
    print(f"  Action-Gradient vs Random Acquisition : Paired Delta = {np.mean(np.array(h5_by_strat['3_action_gradient_disagreement']) - np.array(h5_by_strat['2_random_acquisition'])):+.6f} (t = {t_grad_vs_rand:.3f}, p = {p_grad_vs_rand:.4e})")
    print(f"  Action-Gradient vs D0-Only Reference : Paired Delta = {np.mean(np.array(h5_by_strat['3_action_gradient_disagreement']) - np.array(h5_by_strat['1_d0_only_converged'])):+.6f} (t = {t_grad_vs_d0:.3f}, p = {p_grad_vs_d0:.4e})")
    print(f"  Random Acquisition vs D0-Only Ref    : Paired Delta = {np.mean(np.array(h5_by_strat['2_random_acquisition']) - np.array(h5_by_strat['1_d0_only_converged'])):+.6f} (t = {t_rand_vs_d0:.3f}, p = {p_rand_vs_d0:.4e})")
    print(f"  Privileged Residual vs D0-Only Ref   : Paired Delta = {np.mean(np.array(h5_by_strat['5_privileged_residual']) - np.array(h5_by_strat['1_d0_only_converged'])):+.6f} (t = {t_priv_vs_d0:.3f}, p = {p_priv_vs_d0:.4e})")

    # Determine 4B.1-C Outcome
    if p_grad_vs_rand < 0.05 and np.mean(h5_by_strat["3_action_gradient_disagreement"]) < np.mean(h5_by_strat["2_random_acquisition"]):
        outcome = "OUTCOME_A: ACTION_GRADIENT_BEATS_RANDOM (False negative in old test resolved)"
    elif abs(np.mean(h5_by_strat["3_action_gradient_disagreement"]) - np.mean(h5_by_strat["2_random_acquisition"])) < 0.002 and p_grad_vs_rand >= 0.05:
        outcome = "OUTCOME_B: CLEAN_TRANSFER_BOUNDARY (Signal predicts risk, but small-budget acquisition does not separate in Hopper)"
    else:
        outcome = "OUTCOME_C: ACQUISITIONS_DO_NOT_EXCEED_D0_CONVERGENCE (Data saturation in tested regime)"

    print(f"\n================================================================================")
    print(f"   STAGE 4B.1-C OUTCOME: [{outcome}]")
    print(f"================================================================================")

    # Generate Figures
    generate_stage_4b1c_figures(summary_4b1c, base_h5, figure_dir)

    out_json = {
        "stage_r0v": {
            "verdict": "PASS" if r0v_passed else "FAIL",
            "base_h5": base_h5,
            "old_schedule_mean_h5": mean_old_h5,
            "normalized_converged_mean_h5": mean_norm_h5,
            "t_stat": float(t_r0v),
            "p_val": float(p_r0v),
        },
        "stage_4b1c": {
            "outcome": outcome,
            "base_reference_h5": base_h5,
            "strategies_summary": summary_4b1c,
            "paired_tests": {
                "grad_vs_random": {"t_stat": float(t_grad_vs_rand), "p_val": float(p_grad_vs_rand)},
                "grad_vs_d0": {"t_stat": float(t_grad_vs_d0), "p_val": float(p_grad_vs_d0)},
                "rand_vs_d0": {"t_stat": float(t_rand_vs_d0), "p_val": float(p_rand_vs_d0)},
                "priv_vs_d0": {"t_stat": float(t_priv_vs_d0), "p_val": float(p_priv_vs_d0)},
            },
        },
    }

    report_json_path = os.path.join(output_dir, "stage_r0v_4b1c_retest_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage R0-V & 4B.1-C Report to {report_json_path}")

    env.close()
    return out_json


def generate_stage_4b1c_figures(summary_records: List[Dict], base_h5: float, figure_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [
        "D0-Only\nConverged",
        "Random\n(B=200)",
        "Action-Grad\n(sigma_grad_Q)",
        "Dynamics\n(sigma_dyn)",
        "Privileged\nResidual",
        "Full-Pool\nCoverage",
    ]
    h5_means = [r["mean_h5"] for r in summary_records]
    h5_stds = [r["std_h5"] for r in summary_records]
    acq_deltas = [r["delta_acq"] for r in summary_records]
    colors = ["#7f7f7f", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

    # (a) H=5 Rollout Error under Normalized Operator
    axes[0].bar(names, h5_means, yerr=h5_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[0].axhline(base_h5, color="black", linestyle="--", lw=1.5, label=f"M_0 Baseline ({base_h5:.4f})")
    axes[0].set_ylabel("$H=5$ Autoregressive Rollout Error")
    axes[0].set_title("(a) Operator-Normalized Post-Repair Error")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(loc="upper right", fontsize=9)

    # (b) Marginal Acquisition Contribution relative to D0-Only
    axes[1].bar(names, acq_deltas, color=colors, alpha=0.85, edgecolor="black")
    axes[1].axhline(0, color="black", linestyle="--", lw=1.2)
    axes[1].set_ylabel("Marginal Error Contribution $\\Delta_{\\mathrm{acq}}$ (vs $D_0$-Only)")
    axes[1].set_title("(b) Marginal Acquisition Data Utility")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_4b1c_operator_normalized_retest.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage 4B.1-C Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_r0v_and_4b1c()
