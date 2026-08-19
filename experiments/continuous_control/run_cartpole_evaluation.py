"""
Stage 5: Discrete-Action Continuous-State Control External Validity (CartPole).

Evaluates whether the decision-boundary phenomenon generalizes to a continuous-state domain:
  - State: s in R^4 (continuous cart position, velocity, pole angle, angular velocity).
  - Action: a in {0, 1} (discrete push left, push right).
  - Competent Value Function: LQR Lyapunov Q(s, a).
  - Margin: m(s) = |Q(s, 0) - Q(s, 1)|.
  - Test: Does boundary proximity (small margin m) predict action flips and control damage
    when transition predictive error E = ||s_hat - s_true||^2 is matched?
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.cartpole_continuous import (
    CartPoleDynamics,
    CompetentCartPoleValueFunction,
    CartPoleNeuralDynamics,
)
from src.metrics.diagnostics import evaluate_incremental_r2


def collect_cartpole_dataset(
    dynamics: CartPoleDynamics,
    val_fn: CompetentCartPoleValueFunction,
    num_episodes: int = 60,
    max_steps: int = 100,
    epsilon: float = 0.35,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []

    for _ in range(num_episodes):
        # Initial state: small perturbation around upright equilibrium
        s = rng.uniform(low=[-0.1, -0.1, -0.05, -0.05], high=[0.1, 0.1, 0.05, 0.05])
        for _ in range(max_steps):
            if rng.uniform() < epsilon:
                a = int(rng.choice(2))
            else:
                a = val_fn.get_optimal_action(s)

            next_s, _, done = dynamics.step(s, a)
            delta = next_s - s
            sa = np.concatenate([s, [1.0 if a == 1 else -1.0]])
            x_list.append(sa)
            y_list.append(delta)

            s = next_s
            if done:
                break

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_cartpole_dynamics(
    x_data: np.ndarray,
    y_data: np.ndarray,
    epochs: int = 80,
    lr: float = 0.005,
    seed: int = 42,
) -> CartPoleNeuralDynamics:
    torch.manual_seed(seed)
    model = CartPoleNeuralDynamics(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    n = len(x_data)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for b in range(0, n, 64):
            idx = perm[b : b + 64]
            bx = x_tensor[idx]
            by = y_tensor[idx]

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    return model


def run_stage5_cartpole_benchmark(
    num_seeds: int = 20,
    base_seed: int = 542,
    output_dir: str = "results",
    figure_dir: str = "figures",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 5: DISCRETE-ACTION CONTINUOUS-STATE EXTERNAL VALIDITY (CartPole)       ")
    print("================================================================================")

    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)

    all_test_records: List[Dict] = []
    matched_diffs: List[float] = []

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        rng = np.random.default_rng(seed)

        # 1. Collect training data and train neural dynamics
        x_train, y_train = collect_cartpole_dataset(dynamics, val_fn, num_episodes=50, seed=seed)
        model = train_cartpole_dynamics(x_train, y_train, epochs=75, seed=seed)

        # 2. Sample 200 diverse test states
        for _ in range(200):
            # Sample across wide range of pole angles and angular velocities
            s = rng.uniform(low=[-1.0, -1.0, -0.15, -0.8], high=[1.0, 1.0, 0.15, 0.8])
            opt_a = val_fn.get_optimal_action(s)
            a = int(rng.choice(2))

            # Ground truth step
            true_next_s, _, true_done = dynamics.step(s, a)
            if true_done:
                continue

            # Model predicted step
            pred_next_s = model.predict_next_state(s, a)

            # Metrics
            mse_error = float(np.sum((pred_next_s - true_next_s) ** 2))
            l1_error = float(np.sum(np.abs(pred_next_s - true_next_s)))

            # Margin at next state under true value function
            m_next = val_fn.compute_action_margin(true_next_s)
            boundary_proximity = 1.0 / (m_next + 0.1)

            # Planned optimal action at next state under true vs model predicted state
            true_next_opt_a = val_fn.get_optimal_action(true_next_s)
            model_next_opt_a = val_fn.get_optimal_action(pred_next_s)
            action_flip = int(true_next_opt_a != model_next_opt_a)

            # Correct local decision regret estimand:
            best_q = val_fn.evaluate_q(true_next_s, true_next_opt_a)
            chosen_q = val_fn.evaluate_q(true_next_s, model_next_opt_a)
            control_damage = float(max(0.0, best_q - chosen_q))

            all_test_records.append({
                "seed": seed,
                "model_id": seed_idx,
                "x": float(true_next_s[0]),
                "x_dot": float(true_next_s[1]),
                "theta": float(true_next_s[2]),
                "theta_dot": float(true_next_s[3]),
                "state_norm": float(np.linalg.norm(true_next_s)),
                "error_mse": mse_error,
                "error_l1": l1_error,
                "margin_next": m_next,
                "boundary_proximity": boundary_proximity,
                "action_flip": action_flip,
                "control_damage": control_damage,
            })

    df_cartpole = pd.DataFrame(all_test_records)
    tau_margin = float(df_cartpole["margin_next"].quantile(0.25))
    df_cartpole["z_near"] = (df_cartpole["margin_next"] <= tau_margin).astype(int)

    csv_path = os.path.join(output_dir, "stage5_cartpole_dataset.csv")
    df_cartpole.to_csv(csv_path, index=False)
    print(f"\n-> Collected {len(df_cartpole)} CartPole test transitions. Saved to {csv_path}")

    # -------------------------------------------------------------------------
    # 1. MATCHED-PAIR ANALYSIS (Holding Transition Error Constant)
    # -------------------------------------------------------------------------
    matched_pairs = []
    # Bin by transition error deciles
    df_cartpole["error_bin"] = pd.qcut(df_cartpole["error_mse"], q=10, labels=False)
    for b_id, grp in df_cartpole.groupby("error_bin"):
        med_prox = grp["boundary_proximity"].median()
        near_boundary = grp[grp["boundary_proximity"] >= med_prox]
        far_boundary = grp[grp["boundary_proximity"] < med_prox]
        if len(near_boundary) > 5 and len(far_boundary) > 5:
            d_near = near_boundary["control_damage"].mean()
            d_far = far_boundary["control_damage"].mean()
            diff = d_near - d_far
            matched_pairs.append({
                "bin": b_id,
                "mean_error": grp["error_mse"].mean(),
                "damage_near_boundary": d_near,
                "damage_far_boundary": d_far,
                "diff": diff,
                "flip_rate_near": near_boundary["action_flip"].mean(),
                "flip_rate_far": far_boundary["action_flip"].mean(),
            })

    df_matched = pd.DataFrame(matched_pairs)
    mean_matched_diff = float(df_matched["diff"].mean())

    # Model-level differences for clustered bootstrap
    model_diffs = []
    for m_id, grp in df_cartpole.groupby("model_id"):
        med_prox = grp["boundary_proximity"].median()
        near = grp[grp["boundary_proximity"] >= med_prox]
        far = grp[grp["boundary_proximity"] < med_prox]
        model_diffs.append(near["control_damage"].mean() - far["control_damage"].mean())
    model_diffs = np.array(model_diffs)

    rng_b = np.random.default_rng(42)
    boot_model_means = []
    unique_models = df_cartpole["model_id"].unique()
    for _ in range(2000):
        sampled_models = rng_b.choice(unique_models, size=len(unique_models), replace=True)
        b_diffs = []
        for m in sampled_models:
            grp = df_cartpole[df_cartpole["model_id"] == m]
            med_p = grp["boundary_proximity"].median()
            b_diffs.append(grp[grp["boundary_proximity"] >= med_p]["control_damage"].mean() - grp[grp["boundary_proximity"] < med_p]["control_damage"].mean())
        boot_model_means.append(np.mean(b_diffs))

    bci_matched = [float(np.percentile(boot_model_means, 2.5)), float(np.percentile(boot_model_means, 97.5))]
    boot_mean = float(np.mean(boot_model_means))
    sem_model = float(np.std(model_diffs, ddof=1) / np.sqrt(len(model_diffs)))
    t_stat_model = float(np.mean(model_diffs) / sem_model)
    p_val_model = float(2 * (1 - stats.t.cdf(abs(t_stat_model), df=len(model_diffs)-1)))

    print("\n--- STAGE 5 MATCHED-ERROR BOUNDARY COMPARISON ---")
    print(f"   Matched Damage Difference (Near - Far): {mean_matched_diff:+.5f}")
    print(f"   Model-Level Mean: {np.mean(model_diffs):+.5f}, SEM: {sem_model:.5f}, t(19) = {t_stat_model:.3f}, p = {p_val_model:.4e}")
    print(f"   Cluster-Bootstrap Mean: {boot_mean:+.5f}, 95% BCI: [{bci_matched[0]:.5f}, {bci_matched[1]:.5f}]")
    print(f"   Mean Action Flip Rate: Near-Boundary = {df_matched['flip_rate_near'].mean()*100:.1f}% vs Far-Boundary = {df_matched['flip_rate_far'].mean()*100:.1f}%")

    # -------------------------------------------------------------------------
    # 2. INCREMENTAL REGRESSION ON CARTPOLE
    # -------------------------------------------------------------------------
    from src.metrics.diagnostics import evaluate_cluster_robust_ols

    res_base = evaluate_cluster_robust_ols(df_cartpole, target_col="control_damage", feature_cols=["error_mse", "error_l1"], cluster_col="model_id")
    res_full_near = evaluate_cluster_robust_ols(df_cartpole, target_col="control_damage", feature_cols=["error_mse", "error_l1", "z_near"], cluster_col="model_id")
    res_full_prox = evaluate_cluster_robust_ols(df_cartpole, target_col="control_damage", feature_cols=["error_mse", "error_l1", "boundary_proximity"], cluster_col="model_id")
    res_cov_adj = evaluate_cluster_robust_ols(df_cartpole, target_col="control_damage", feature_cols=["error_mse", "error_l1", "x", "x_dot", "theta", "theta_dot", "z_near"], cluster_col="model_id")

    delta_r2_near = float(res_full_near["r2"] - res_base["r2"])
    delta_r2_prox = float(res_full_prox["r2"] - res_base["r2"])

    # Seed-level Delta R^2
    seed_delta_r2 = []
    for m_id, grp in df_cartpole.groupby("model_id"):
        r_b = evaluate_incremental_r2(grp, target_col="control_damage", control_cols=["error_mse", "error_l1"], proposed_col="z_near")
        seed_delta_r2.append(r_b["delta_r2"])
    seed_delta_r2 = np.array(seed_delta_r2)
    seed_mean_delta_r2 = float(np.mean(seed_delta_r2))
    seed_bci_delta_r2 = [float(np.percentile(seed_delta_r2, 2.5)), float(np.percentile(seed_delta_r2, 97.5))]

    print("\n--- STAGE 5 INCREMENTAL REGRESSION RESULTS ---")
    print(f"   Baseline R^2 (MSE + L1): {res_base['r2']:.4f}")
    print(f"   Full Model R^2 (+ z_near): {res_full_near['r2']:.4f}, Delta R^2 = +{delta_r2_near:.4f}")
    print(f"   Full Model R^2 (+ boundary_proximity): {res_full_prox['r2']:.4f}, Delta R^2 = +{delta_r2_prox:.4f}")
    print(f"   Covariate-Adjusted Model R^2: {res_cov_adj['r2']:.4f}")
    print(f"   Seed-Level Mean Delta R^2: {seed_mean_delta_r2:.4f}, 95% BCI: [{seed_bci_delta_r2[0]:.4f}, {seed_bci_delta_r2[1]:.4f}]")

    # Action flip classification AUROC
    y_flips = df_cartpole["action_flip"].to_numpy()
    scores_prox = df_cartpole["boundary_proximity"].to_numpy()
    u_stat, _ = stats.mannwhitneyu(scores_prox[y_flips == 1], scores_prox[y_flips == 0], alternative="greater")
    n1 = np.sum(y_flips == 1)
    n0 = np.sum(y_flips == 0)
    auroc_flip = float(u_stat / (n1 * n0))
    print(f"   Boundary Proximity Predicting Action Flips AUROC: {auroc_flip:.4f}")

    # Generate Figure
    generate_cartpole_figures(df_cartpole, df_matched, figure_dir)

    summary_dict = {
        "mean_matched_diff": mean_matched_diff,
        "model_level_mean": float(np.mean(model_diffs)),
        "sem_model": sem_model,
        "t_stat_model": t_stat_model,
        "p_val_model": p_val_model,
        "cluster_bootstrap_mean": boot_mean,
        "matched_bci_95": bci_matched,
        "flip_rate_near": float(df_matched["flip_rate_near"].mean()),
        "flip_rate_far": float(df_matched["flip_rate_far"].mean()),
        "r2_base": float(res_base["r2"]),
        "r2_full_near": float(res_full_near["r2"]),
        "delta_r2_near": delta_r2_near,
        "coef_near": res_full_near["coefficients"],
        "r2_full_prox": float(res_full_prox["r2"]),
        "delta_r2_prox": delta_r2_prox,
        "coef_prox": res_full_prox["coefficients"],
        "r2_cov_adj": float(res_cov_adj["r2"]),
        "coef_cov_adj": res_cov_adj["coefficients"],
        "seed_mean_delta_r2": seed_mean_delta_r2,
        "seed_bci_delta_r2": seed_bci_delta_r2,
        "auroc_flip": auroc_flip,
    }

    with open(os.path.join(output_dir, "stage5_cartpole_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    return summary_dict


def generate_cartpole_figures(df_cartpole: pd.DataFrame, df_matched: pd.DataFrame, figure_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Action Flip Rate by Error Decile
    axes[0].bar(df_matched["bin"] - 0.15, df_matched["flip_rate_near"] * 100, width=0.3, color="#d62728", label="Near-Boundary (Small Margin)")
    axes[0].bar(df_matched["bin"] + 0.15, df_matched["flip_rate_far"] * 100, width=0.3, color="#1f77b4", label="Far-Boundary (Large Margin)")
    axes[0].set_xlabel("Transition Error Decile")
    axes[0].set_ylabel("Action Flip Rate (%)")
    axes[0].set_title("(a) Matched-Error Action Flip Rate")
    axes[0].legend(loc="upper left", fontsize=9.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Control Damage by Boundary Proximity
    axes[1].scatter(df_cartpole["boundary_proximity"], df_cartpole["control_damage"], alpha=0.25, color="#4C72B0", s=15)
    axes[1].set_xlabel("Boundary Proximity $1 / (m(s') + 0.1)$")
    axes[1].set_ylabel("Local Decision Regret $D_{\\mathrm{reg}}$")
    axes[1].set_title("(b) Continuous Margin Regret Relation")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage5_cartpole_external_validity.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved CartPole External Validity Figure to {fig_path}")


if __name__ == "__main__":
    run_stage5_cartpole_benchmark()
