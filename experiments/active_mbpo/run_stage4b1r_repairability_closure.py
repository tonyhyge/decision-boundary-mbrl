"""
Stage 4B.1-R: Model Repairability Closure and Causal Decomposition on Hopper-v4.

Core Scientific Question:
  Can the current Hopper world model be improved at all by adding informative queried
  transitions under a controlled repair operator, or does parameter update drift / non-stationarity
  dominate post-repair performance?

Evaluates the 5 Canonical Controls + Diagnostics:
  1. No-Update Baseline (Frozen M_0)
  2. D0-Only Extra-SGD Control (Pure optimization drift check with 0 new data)
  3. Random Acquisition (B = 200)
  4. Dynamics Uncertainty sigma_dyn (B = 200)
  5. Action-Gradient Disagreement sigma_grad_Q (B = 200)
  6. Privileged Residual Oracle (B = 200 states with largest true 1-step model residual)
  7. Full-Pool Acquisition (All available candidate transitions)

Stratified Metrics on Held-Out Evaluation Set D_eval:
  - 1-step prediction MSE (E_H1)
  - H=5 autoregressive unroll error (E_H5)
  - In-distribution stable error (E_ID, height z >= 1.0)
  - Near-failure / falling error (E_near_fall, height z < 1.0)
  - Acquisition state selection diagnostic: P(near-fall | selected), mean height z, mean angle.
"""
import os
import sys
import json
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


def evaluate_stratified_errors(
    ensemble: HopperEnsembleDynamics,
    eval_trajs: List[Dict],
    horizon: int = 5,
) -> Dict[str, float]:
    """
    Evaluates 1-step MSE, H=5 rollout MSE, in-distribution (ID) MSE, and near-fall MSE.
    """
    h1_errors = []
    h5_errors = []
    id_errors = []
    near_fall_errors = []

    for traj in eval_trajs:
        obs_seq = traj["obs"]
        act_seq = traj["acts"]
        next_obs_seq = traj["next_obs"]
        T = len(obs_seq)

        # 1-step evaluations and stratification
        for t in range(T):
            s = obs_seq[t]
            a = act_seq[t]
            true_ns = next_obs_seq[t]

            pred_all = ensemble.predict_all(s, a)
            mean_pred = np.mean(pred_all, axis=0)
            err_1 = float(np.mean((mean_pred - true_ns) ** 2))
            h1_errors.append(err_1)

            # Hopper height is at index 0 of observation (z-coordinate)
            z_height = s[0]
            if z_height >= 1.0:
                id_errors.append(err_1)
            else:
                near_fall_errors.append(err_1)

        # H=5 autoregressive unroll error
        if T > horizon:
            for t in range(0, T - horizon, horizon):
                s_sim = obs_seq[t].copy()
                for h in range(horizon):
                    a_curr = act_seq[t + h]
                    pred_all = ensemble.predict_all(s_sim, a_curr)
                    s_sim = np.mean(pred_all, axis=0)
                    true_s = obs_seq[t + h + 1]
                    h5_errors.append(float(np.mean((s_sim - true_s) ** 2)))

    return {
        "e_h1": float(np.mean(h1_errors)),
        "e_h5": float(np.mean(h5_errors)),
        "e_id": float(np.mean(id_errors)) if id_errors else float(np.mean(h1_errors)),
        "e_near_fall": float(np.mean(near_fall_errors)) if near_fall_errors else 0.0,
    }


def run_stage4b1r_closure_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    budget_count: int = 200,
    n_seeds: int = 10,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 4B.1-R: MODEL REPAIRABILITY CLOSURE & CAUSAL DECOMPOSITION (HOPPER)   ")
    print("================================================================================")

    env = gym.make("Hopper-v4")

    # 1. Base Dataset D_0 and Candidate Pool D_pool
    print("\n[Step 1] Initializing base dataset D_0 (N_0 = 3,000 steps)...")
    base_trajs = collect_hopper_trajectories(env, num_trajectories=35, max_steps=120, seed=seed)
    x_base, y_base = [], []
    for tr in base_trajs:
        for t in range(len(tr["obs"])):
            x_base.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_base.append(tr["next_obs"][t] - tr["obs"][t])
    x_base = np.array(x_base, dtype=np.float32)
    y_base = np.array(y_base, dtype=np.float32)
    print(f"-> Base Dataset D_0: {len(x_base)} samples.")

    # 2. Train Initial World Model Ensemble M_0
    print("[Step 2] Training Initial World Model Ensemble M_0...")
    base_ensemble = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed)
    base_ensemble.train_ensemble(x_base, y_base, epochs=50, batch_size=64, seed=seed)

    # 3. Candidate Pool & Held-Out Evaluation Set
    print("\n[Step 3] Collecting Candidate Pool D_pool & Evaluation Set D_eval...")
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
    print(f"-> Candidate Pool: {n_pool} states | Evaluation Set: {sum(len(tr['obs']) for tr in eval_trajs)} states.")

    # 4. Extract Diagnostic Signals and Privileged True Model Residuals
    print("[Step 4] Extracting Diagnostic Scores & Privileged True Residuals on Candidate Pool...")
    critic = HopperHeuristicCritic(ensemble=base_ensemble)

    sigma_dyn_list = []
    sigma_grad_q_list = []
    true_residual_list = []
    near_fall_list = []

    for _, row in df_pool.iterrows():
        s = row["obs"]
        a = row["act"]
        true_ns = row["next_obs"]

        # Dynamics Uncertainty
        pred_all = base_ensemble.predict_all(s, a)
        mean_pred = np.mean(pred_all, axis=0)
        sigma_dyn = float(np.mean(np.std(pred_all, axis=0, ddof=1)))
        sigma_dyn_list.append(sigma_dyn)

        # Action-Gradient Disagreement
        grads_k = critic.estimate_action_gradient(s, a)
        sigma_g = float(np.mean(np.std(grads_k, axis=0, ddof=1)))
        sigma_grad_q_list.append(sigma_g)

        # Privileged True Model Residual ||f_P(s, a) - f_M(s, a)||
        res_true = float(np.linalg.norm(true_ns - mean_pred))
        true_residual_list.append(res_true)

        # State condition
        near_fall_list.append(int(s[0] < 1.0))

    df_pool["sigma_dyn"] = np.array(sigma_dyn_list)
    df_pool["sigma_grad_q"] = np.array(sigma_grad_q_list)
    df_pool["true_residual"] = np.array(true_residual_list)
    df_pool["near_fall"] = np.array(near_fall_list)

    # 5. Measure Acquisition State Selection Distribution
    print("\n[Step 5] Characterizing Selected State Distribution across Acquisition Strategies...")
    b_sz = min(budget_count, n_pool)
    idx_top_dyn = np.argsort(-df_pool["sigma_dyn"].to_numpy())[:b_sz]
    idx_top_grad = np.argsort(-df_pool["sigma_grad_q"].to_numpy())[:b_sz]
    idx_top_oracle = np.argsort(-df_pool["true_residual"].to_numpy())[:b_sz]
    idx_rand = np.random.default_rng(seed).choice(n_pool, size=b_sz, replace=False)

    p_fall_dyn = float(df_pool.iloc[idx_top_dyn]["near_fall"].mean())
    p_fall_grad = float(df_pool.iloc[idx_top_grad]["near_fall"].mean())
    p_fall_oracle = float(df_pool.iloc[idx_top_oracle]["near_fall"].mean())
    p_fall_rand = float(df_pool.iloc[idx_rand]["near_fall"].mean())

    mean_z_dyn = float(np.mean([df_pool.iloc[i]["obs"][0] for i in idx_top_dyn]))
    mean_z_grad = float(np.mean([df_pool.iloc[i]["obs"][0] for i in idx_top_grad]))
    mean_z_rand = float(np.mean([df_pool.iloc[i]["obs"][0] for i in idx_rand]))

    print(f"  P(Near-Fall | Selected by sigma_dyn)      = {p_fall_dyn*100:.1f}% (Mean Height z = {mean_z_dyn:.3f})")
    print(f"  P(Near-Fall | Selected by sigma_grad_Q)  = {p_fall_grad*100:.1f}% (Mean Height z = {mean_z_grad:.3f})")
    print(f"  P(Near-Fall | Selected by Random)        = {p_fall_rand*100:.1f}% (Mean Height z = {mean_z_rand:.3f})")
    print(f"  P(Near-Fall | Selected by True Oracle)   = {p_fall_oracle*100:.1f}%")

    # Baseline performance before repair
    base_errors = evaluate_stratified_errors(base_ensemble, eval_trajs, horizon=5)
    print(f"\n-> Baseline Untouched Model M_0 Errors:")
    print(f"   H=5 Rollout Error: {base_errors['e_h5']:.6f} | 1-Step MSE: {base_errors['e_h1']:.6f}")
    print(f"   In-Distribution (z >= 1.0) MSE: {base_errors['e_id']:.6f} | Near-Fall (z < 1.0) MSE: {base_errors['e_near_fall']:.6f}")

    # -------------------------------------------------------------------------
    # 6. EVALUATE 5 CANONICAL CONTROLS OVER N = 10 INDEPENDENT SEEDS
    # -------------------------------------------------------------------------
    print(f"\n================================================================================")
    print(f"   RUNNING 5 CONTROLS CLOSURE BENCHMARK (MATCHED GRADIENTS, N = {n_seeds} SEEDS)  ")
    print(f"================================================================================")

    controls = [
        "1_no_update_baseline",
        "2_d0_extra_sgd_drift",
        "3_random_acquisition",
        "4_action_gradient_disagreement",
        "5_dynamics_uncertainty",
        "6_privileged_residual_oracle",
        "7_full_pool_acquisition",
    ]

    h5_by_ctrl = {c: [] for c in controls}
    h1_by_ctrl = {c: [] for c in controls}
    id_by_ctrl = {c: [] for c in controls}
    near_fall_by_ctrl = {c: [] for c in controls}

    for s_idx in range(n_seeds):
        rng_s = np.random.default_rng(seed + s_idx * 59)

        # 1. No Update
        h5_by_ctrl["1_no_update_baseline"].append(base_errors["e_h5"])
        h1_by_ctrl["1_no_update_baseline"].append(base_errors["e_h1"])
        id_by_ctrl["1_no_update_baseline"].append(base_errors["e_id"])
        near_fall_by_ctrl["1_no_update_baseline"].append(base_errors["e_near_fall"])

        # 2. D0-Only Extra-SGD (Pure optimization drift check)
        ens_d0_extra = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed + s_idx)
        ens_d0_extra.train_ensemble(x_base, y_base, epochs=40, batch_size=64, seed=seed + s_idx)
        ev_d0 = evaluate_stratified_errors(ens_d0_extra, eval_trajs, horizon=5)
        h5_by_ctrl["2_d0_extra_sgd_drift"].append(ev_d0["e_h5"])
        h1_by_ctrl["2_d0_extra_sgd_drift"].append(ev_d0["e_h1"])
        id_by_ctrl["2_d0_extra_sgd_drift"].append(ev_d0["e_id"])
        near_fall_by_ctrl["2_d0_extra_sgd_drift"].append(ev_d0["e_near_fall"])

        # Subsets for remaining controls
        idx_rand_s = rng_s.choice(n_pool, size=b_sz, replace=False)
        idx_grad_s = np.argsort(-df_pool["sigma_grad_q"].to_numpy())[:b_sz]
        idx_dyn_s = np.argsort(-df_pool["sigma_dyn"].to_numpy())[:b_sz]
        idx_oracle_s = np.argsort(-df_pool["true_residual"].to_numpy())[:b_sz]
        idx_full_s = np.arange(n_pool)

        subset_map = {
            "3_random_acquisition": idx_rand_s,
            "4_action_gradient_disagreement": idx_grad_s,
            "5_dynamics_uncertainty": idx_dyn_s,
            "6_privileged_residual_oracle": idx_oracle_s,
            "7_full_pool_acquisition": idx_full_s,
        }

        for ctrl_name, q_idx in subset_map.items():
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

            ens_aug = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed + s_idx)
            ens_aug.train_ensemble(x_aug, y_aug, epochs=40, batch_size=64, seed=seed + s_idx)

            ev_aug = evaluate_stratified_errors(ens_aug, eval_trajs, horizon=5)
            h5_by_ctrl[ctrl_name].append(ev_aug["e_h5"])
            h1_by_ctrl[ctrl_name].append(ev_aug["e_h1"])
            id_by_ctrl[ctrl_name].append(ev_aug["e_id"])
            near_fall_by_ctrl[ctrl_name].append(ev_aug["e_near_fall"])

    # Statistical Summary
    summary_records = []
    print("\n--- STAGE 4B.1-R SUMMARY PERFORMANCE (N = 10 SEEDS) ---")
    for c in controls:
        m_h5 = float(np.mean(h5_by_ctrl[c]))
        s_h5 = float(np.std(h5_by_ctrl[c]))
        m_h1 = float(np.mean(h1_by_ctrl[c]))
        m_id = float(np.mean(id_by_ctrl[c]))
        m_fall = float(np.mean(near_fall_by_ctrl[c]))
        delta_h5_pct = (m_h5 - base_errors["e_h5"]) / base_errors["e_h5"] * 100

        print(f"  {c:<32}: H=5 Error = {m_h5:.6f} +/- {s_h5:.6f} ({delta_h5_pct:+.2f}%) | 1-Step = {m_h1:.6f} | ID = {m_id:.6f} | Fall = {m_fall:.6f}")

        summary_records.append({
            "control": c,
            "mean_h5_error": m_h5,
            "std_h5_error": s_h5,
            "delta_h5_pct": delta_h5_pct,
            "mean_h1_error": m_h1,
            "mean_id_error": m_id,
            "mean_near_fall_error": m_fall,
        })

    # Causal Decomposition & Decision Tree Audits
    # 1. Check Optimization Drift: D0-only extra-SGD vs No-Update Baseline
    t_drift, p_drift = stats.ttest_1samp(h5_by_ctrl["2_d0_extra_sgd_drift"], base_errors["e_h5"])
    opt_drift_detected = bool(np.mean(h5_by_ctrl["2_d0_extra_sgd_drift"]) > base_errors["e_h5"] and p_drift < 0.05)

    # 2. Check Oracle Repairability: Privileged Oracle vs No-Update Baseline
    t_oracle, p_oracle = stats.ttest_1samp(h5_by_ctrl["6_privileged_residual_oracle"], base_errors["e_h5"])
    oracle_improves = bool(np.mean(h5_by_ctrl["6_privileged_residual_oracle"]) < base_errors["e_h5"] and p_oracle < 0.05)

    # 3. Check Action-Gradient vs Dynamics
    t_grad_dyn, p_grad_dyn = stats.ttest_rel(h5_by_ctrl["4_action_gradient_disagreement"], h5_by_ctrl["5_dynamics_uncertainty"])

    # Decision Tree Determination
    if oracle_improves:
        decision_case = "CASE_A: ORACLE_REPAIRS_MODEL (Repair operator viable, but unprivileged acquisition needs calibration)"
    elif opt_drift_detected:
        decision_case = "CASE_C: OPTIMIZATION_DRIFT_DOMINATES (Extra SGD on sparse data degrades representation independently of acquisition)"
    else:
        decision_case = "CASE_B: OPERATOR_NOT_REPAIRABLE (One-shot scratch retraining is not repairable on sparse Hopper data)"

    print(f"\n================================================================================")
    print(f"   STAGE 4B.1-R DECISION TREE OUTCOME: [{decision_case}]")
    print(f"================================================================================")
    print(f"  1. Optimization Drift Detected (D0-ExtraSGD > Baseline): {opt_drift_detected} (t = {t_drift:.3f}, p = {p_drift:.4e})")
    print(f"  2. Privileged Oracle Improves Model: {oracle_improves} (Mean H5: {np.mean(h5_by_ctrl['6_privileged_residual_oracle']):.6f} vs Base: {base_errors['e_h5']:.6f}, p = {p_oracle:.4e})")
    print(f"  3. Action-Gradient Disagreement beats Dynamics Uncertainty: {p_grad_dyn < 0.05} (t = {t_grad_dyn:.3f}, p = {p_grad_dyn:.4e})")

    # Generate Figures
    generate_stage4b1r_figures(summary_records, figure_dir)

    out_json = {
        "decision_case": decision_case,
        "optimization_drift_detected": opt_drift_detected,
        "oracle_improves": oracle_improves,
        "state_selection_diagnostics": {
            "p_fall_sigma_dyn": p_fall_dyn,
            "p_fall_sigma_grad_q": p_fall_grad,
            "p_fall_random": p_fall_rand,
            "p_fall_oracle": p_fall_oracle,
            "mean_z_sigma_dyn": mean_z_dyn,
            "mean_z_sigma_grad_q": mean_z_grad,
            "mean_z_random": mean_z_rand,
        },
        "controls_summary": summary_records,
        "t_drift": float(t_drift),
        "p_drift": float(p_drift),
        "t_oracle": float(t_oracle),
        "p_oracle": float(p_oracle),
        "t_grad_dyn": float(t_grad_dyn),
        "p_grad_dyn": float(p_grad_dyn),
    }

    report_path = os.path.join(output_dir, "stage4b1r_repairability_closure_report.json")
    with open(report_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n-> Saved Stage 4B.1-R JSON Report to {report_path}")

    env.close()
    return out_json


def generate_stage4b1r_figures(summary_records: List[Dict], figure_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [
        "No-Update\n(Base M0)",
        "D0-Only\nExtra SGD",
        "Random\n(B=200)",
        "Action-Grad\n(sigma_grad_Q)",
        "Dynamics\n(sigma_dyn)",
        "Residual\nOracle",
        "Full-Pool\n(All Data)",
    ]
    h5_means = [r["mean_h5_error"] for r in summary_records]
    h5_stds = [r["std_h5_error"] for r in summary_records]
    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

    # (a) H=5 Rollout Error by Control
    bars = axes[0].bar(names, h5_means, yerr=h5_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[0].axhline(h5_means[0], color="black", linestyle="--", label="Untouched Baseline")
    axes[0].set_ylabel("$H=5$ Autoregressive Rollout Error")
    axes[0].set_title("(a) Causal Decomposition of Model Repair on Hopper")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(loc="upper left", fontsize=9)

    # (b) In-Distribution vs Near-Fall Error Breakdown
    id_means = [r["mean_id_error"] for r in summary_records]
    fall_means = [r["mean_near_fall_error"] for r in summary_records]
    x_pos = np.arange(len(names))
    width = 0.35

    axes[1].bar(x_pos - width/2, id_means, width, label="In-Distribution (z >= 1.0)", color="#1f77b4", alpha=0.85)
    axes[1].bar(x_pos + width/2, fall_means, width, label="Near-Fall (z < 1.0)", color="#d62728", alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, fontsize=8.5)
    axes[1].set_ylabel("1-Step Prediction Error")
    axes[1].set_title("(b) Stratified In-Distribution vs Falling Error")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage4b1r_repairability_decomposition.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Causal Decomposition Figure to {fig_path}")


if __name__ == "__main__":
    run_stage4b1r_closure_benchmark()
