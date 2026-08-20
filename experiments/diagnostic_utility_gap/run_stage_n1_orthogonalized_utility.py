"""
Stage N1: Orthogonalized Set-Utility Mechanism Benchmark on Hopper-v4.

Core Scientific Question:
  Why does a pointwise decision-risk signal fail as a batch-level training criterion?
  Does decision relevance provide conditional training utility only when batch coverage
  is preserved, or is it fundamentally decoupled from neural world-model updates?

Experimental Design (Orthogonalized 2x2 Factorial):
  - Fixed Compute: T_steps = 650 gradient updates across all conditions.
  - Two-Stream Replay Sampler: Every batch draws exactly 51 samples from D0 (80%)
    and 13 samples from S (20%), eliminating dataset-size dilution confounds.
  - Normalized (s, a) State-Action Space Clustering:
    1. High Risk / High Coverage (HR/HC): Top risk distributed across 10 spatial clusters.
    2. High Risk / Low Coverage  (HR/LC): Top risk concentrated in 2 tight spatial clusters.
    3. Low Risk  / High Coverage (LR/HC): Low risk distributed across 10 spatial clusters.
    4. Low Risk  / Low Coverage  (LR/LC): Low risk concentrated in 2 tight spatial clusters.

Statistical Model:
  U(S) = beta_0 + beta_R * R + beta_C * C + beta_RC * (R * C) + epsilon
  where U(S) = -(H5(theta_{D0 + S}) - H5(theta_{D0})).
"""
import os
import sys
import json
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
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
)
from experiments.active_mbpo.run_stage4b1r_repairability_closure import evaluate_stratified_errors


def train_ensemble_two_stream(
    ensemble: HopperEnsembleDynamics,
    x_d0: np.ndarray,
    y_d0: np.ndarray,
    x_s: Optional[np.ndarray],
    y_s: Optional[np.ndarray],
    total_steps: int = 650,
    batch_d0: int = 51,
    batch_s: int = 13,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> None:
    """Trains ensemble using strict two-stream replay: 80% D0 and 20% S in every mini-batch."""
    n_d0 = len(x_d0)
    has_subset = (x_s is not None) and (len(x_s) > 0)
    n_s = len(x_s) if has_subset else 0
    criterion = nn.MSELoss()

    for k, model in enumerate(ensemble.models):
        torch.manual_seed(seed + k * 100)
        rng = np.random.default_rng(seed + k * 100)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        for step in range(total_steps):
            # Draw from D0
            idx_d0 = rng.choice(n_d0, size=batch_d0 if has_subset else 64, replace=True)
            bx_d0 = x_d0[idx_d0]
            by_d0 = y_d0[idx_d0]

            if has_subset:
                # Draw from S
                idx_s = rng.choice(n_s, size=batch_s, replace=True)
                bx_s = x_s[idx_s]
                by_s = y_s[idx_s]
                bx = torch.tensor(np.concatenate([bx_d0, bx_s], axis=0), dtype=torch.float32)
                by = torch.tensor(np.concatenate([by_d0, by_s], axis=0), dtype=torch.float32)
            else:
                bx = torch.tensor(bx_d0, dtype=torch.float32)
                by = torch.tensor(by_d0, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()


def construct_orthogonalized_cells(
    x_pool: np.ndarray,
    scores: np.ndarray,
    budget: int = 200,
    n_clusters: int = 10,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Constructs 4 balanced cells in (s, a) space: HR/HC, HR/LC, LR/HC, LR/LC."""
    scaler = StandardScaler()
    x_pool_norm = scaler.fit_transform(x_pool)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(x_pool_norm)
    labels = kmeans.labels_

    # Sort into high-risk (top 50%) and low-risk (bottom 50%)
    median_score = np.median(scores)
    high_risk_mask = scores >= median_score
    low_risk_mask = scores < median_score

    # 1. High Risk / High Coverage (HR/HC): Top risk spread across all 10 clusters
    hr_hc = []
    per_c = budget // n_clusters
    for c in range(n_clusters):
        c_idx = np.where((labels == c) & high_risk_mask)[0]
        if len(c_idx) == 0:
            c_idx = np.where(labels == c)[0]
        top_local = c_idx[np.argsort(-scores[c_idx])[:per_c]]
        hr_hc.extend(top_local.tolist())
    if len(hr_hc) < budget:
        rem = list(set(np.where(high_risk_mask)[0]) - set(hr_hc))
        hr_hc.extend(np.array(rem)[np.argsort(-scores[rem])[:budget - len(hr_hc)]].tolist())
    idx_hr_hc = np.array(hr_hc[:budget], dtype=int)

    # 2. High Risk / Low Coverage (HR/LC): Top risk concentrated in 2 tight clusters
    # Pick the 2 clusters with the largest number of high-risk points
    c_counts_hr = [np.sum((labels == c) & high_risk_mask) for c in range(n_clusters)]
    top_2_clusters = np.argsort(-np.array(c_counts_hr))[:2]
    hr_lc = []
    for c in top_2_clusters:
        c_idx = np.where((labels == c) & high_risk_mask)[0]
        top_local = c_idx[np.argsort(-scores[c_idx])[:budget // 2]]
        hr_lc.extend(top_local.tolist())
    if len(hr_lc) < budget:
        rem_c = np.where(np.isin(labels, top_2_clusters))[0]
        rem = list(set(rem_c) - set(hr_lc))
        hr_lc.extend(np.array(rem)[np.argsort(-scores[rem])[:budget - len(hr_lc)]].tolist())
    idx_hr_lc = np.array(hr_lc[:budget], dtype=int)

    # 3. Low Risk / High Coverage (LR/HC): Low risk spread across all 10 clusters
    lr_hc = []
    for c in range(n_clusters):
        c_idx = np.where((labels == c) & low_risk_mask)[0]
        if len(c_idx) == 0:
            c_idx = np.where(labels == c)[0]
        bot_local = c_idx[np.argsort(scores[c_idx])[:per_c]]
        lr_hc.extend(bot_local.tolist())
    if len(lr_hc) < budget:
        rem = list(set(np.where(low_risk_mask)[0]) - set(lr_hc))
        lr_hc.extend(np.array(rem)[np.argsort(scores[rem])[:budget - len(lr_hc)]].tolist())
    idx_lr_hc = np.array(lr_hc[:budget], dtype=int)

    # 4. Low Risk / Low Coverage (LR/LC): Low risk concentrated in 2 tight clusters
    c_counts_lr = [np.sum((labels == c) & low_risk_mask) for c in range(n_clusters)]
    top_2_lr_clusters = np.argsort(-np.array(c_counts_lr))[:2]
    lr_lc = []
    for c in top_2_lr_clusters:
        c_idx = np.where((labels == c) & low_risk_mask)[0]
        bot_local = c_idx[np.argsort(scores[c_idx])[:budget // 2]]
        lr_lc.extend(bot_local.tolist())
    if len(lr_lc) < budget:
        rem_c = np.where(np.isin(labels, top_2_lr_clusters))[0]
        rem = list(set(rem_c) - set(lr_lc))
        lr_lc.extend(np.array(rem)[np.argsort(scores[rem])[:budget - len(lr_lc)]].tolist())
    idx_lr_lc = np.array(lr_lc[:budget], dtype=int)

    return {
        "HR_HC": idx_hr_hc,
        "HR_LC": idx_hr_lc,
        "LR_HC": idx_lr_hc,
        "LR_LC": idx_lr_lc,
    }


def compute_normalized_cluster_entropy(x_pool: np.ndarray, x_subset: np.ndarray, n_clusters: int = 10) -> float:
    scaler = StandardScaler()
    x_pool_norm = scaler.fit_transform(x_pool)
    x_sub_norm = scaler.transform(x_subset)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(x_pool_norm)
    labels = kmeans.predict(x_sub_norm)
    counts = np.bincount(labels, minlength=n_clusters)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(n_clusters))


def run_stage_n1_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    n_seeds: int = 10,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE N1: ORTHOGONALIZED SET-UTILITY MECHANISM BENCHMARK (HOPPER-v4)        ")
    print("================================================================================")

    env = gym.make("Hopper-v4")

    # 1. Dataset Collection
    print("\n[Step 1] Initializing base dataset D_0 (N_0 = 830) & Candidate Pool (N=807)...")
    base_trajs = collect_hopper_trajectories(env, num_trajectories=35, max_steps=120, seed=seed)
    x_base, y_base = [], []
    for tr in base_trajs:
        for t in range(len(tr["obs"])):
            x_base.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_base.append(tr["next_obs"][t] - tr["obs"][t])
    x_base = np.array(x_base, dtype=np.float32)
    y_base = np.array(y_base, dtype=np.float32)

    # Base reference model M_0
    base_ensemble = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed)
    base_ensemble.train_ensemble(x_base, y_base, epochs=50, batch_size=64, lr=0.001, seed=seed)

    pool_trajs = collect_hopper_trajectories(env, num_trajectories=30, max_steps=100, seed=seed + 100)
    eval_trajs = collect_hopper_trajectories(env, num_trajectories=25, max_steps=100, seed=seed + 200)

    x_pool, y_pool = [], []
    for tr in pool_trajs:
        for t in range(len(tr["obs"])):
            x_pool.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_pool.append(tr["next_obs"][t] - tr["obs"][t])
    x_pool = np.array(x_pool, dtype=np.float32)
    y_pool = np.array(y_pool, dtype=np.float32)
    n_pool = len(x_pool)

    base_eval_metrics = evaluate_stratified_errors(base_ensemble, eval_trajs, horizon=5)
    base_h5 = base_eval_metrics["e_h5"]
    print(f"-> Base Dataset D_0: {len(x_base)} samples | Candidate Pool: {n_pool} states.")
    print(f"-> Reference Model M_0 Baseline H=5 Error: {base_h5:.6f}")

    # Extract diagnostic risk score (sigma_grad_Q)
    print("\n[Step 2] Extracting Candidate Pool Action-Gradient Risk Scores...")
    critic = HopperHeuristicCritic(ensemble=base_ensemble)
    sigma_grad_q_list = []
    for i in range(n_pool):
        s = x_pool[i][:11]
        a = x_pool[i][11:]
        grads_k = critic.estimate_action_gradient(s, a)
        sigma_grad_q_list.append(float(np.mean(np.std(grads_k, axis=0, ddof=1))))
    scores_pool = np.array(sigma_grad_q_list)

    # 2. Construct 4 Orthogonalized Cells
    print("\n[Step 3] Constructing 2x2 Orthogonalized Subsets in (s, a) Space...")
    cells = construct_orthogonalized_cells(x_pool, scores_pool, budget=200, n_clusters=10, seed=seed)

    cell_diagnostics = {}
    for c_name, idx_c in cells.items():
        sub_x = x_pool[idx_c]
        sub_scores = scores_pool[idx_c]
        entropy = compute_normalized_cluster_entropy(x_pool, sub_x, n_clusters=10)
        pdist = pairwise_distances(sub_x)
        np.fill_diagonal(pdist, np.nan)
        mean_pdist = float(np.nanmean(pdist))

        cell_diagnostics[c_name] = {
            "mean_risk": float(np.mean(sub_scores)),
            "std_risk": float(np.std(sub_scores)),
            "cluster_entropy": entropy,
            "mean_pdist": mean_pdist,
        }
        print(f"  Cell [{c_name:<5}]: Mean Risk = {cell_diagnostics[c_name]['mean_risk']:.4f} | Cluster Entropy = {entropy:.4f} | Mean Pairwise Dist = {mean_pdist:.4f}")

    # 3. Two-Stream Replay Benchmark (N = 10 Seeds per Cell + D0-Only Baseline)
    print("\n================================================================================")
    print("   RUNNING TWO-STREAM 2x2 FACTORIAL BENCHMARK (N = 10 SEEDS, T_steps = 650)     ")
    print("================================================================================")

    # D0-Only Baseline under Two-Stream / Matched Steps
    d0_h5_seeds = []
    for s_idx in range(n_seeds):
        seed_val = seed + s_idx * 13
        ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
        train_ensemble_two_stream(ens, x_base, y_base, None, None, total_steps=650, lr=0.003, seed=seed_val)
        ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)
        d0_h5_seeds.append(ev["e_h5"])
    mean_d0_h5 = float(np.mean(d0_h5_seeds))
    print(f"-> D0-Only Converged Baseline (Two-Stream, T=650): H=5 Error = {mean_d0_h5:.6f}")

    cell_results = {c: [] for c in cells}
    regression_records = []

    for c_name, idx_c in cells.items():
        is_high_risk = 1.0 if "HR" in c_name else 0.0
        is_high_cov = 1.0 if "HC" in c_name else 0.0

        x_s = x_pool[idx_c]
        y_s = y_pool[idx_c]

        for s_idx in range(n_seeds):
            seed_val = seed + s_idx * 23 + int(is_high_risk * 100) + int(is_high_cov * 10)
            ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
            train_ensemble_two_stream(
                ens, x_base, y_base, x_s, y_s, total_steps=650, batch_d0=51, batch_s=13, lr=0.003, seed=seed_val
            )
            ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)
            h5_err = ev["e_h5"]
            cell_results[c_name].append(h5_err)

            # Realized Utility U(S) = -(H5 - mean_d0_h5)
            u_s = -(h5_err - mean_d0_h5)

            regression_records.append({
                "cell": c_name,
                "risk": is_high_risk,
                "coverage": is_high_cov,
                "risk_x_cov": is_high_risk * is_high_cov,
                "h5_error": h5_err,
                "utility_u_s": u_s,
            })

    print("\n--- 2x2 FACTORIAL CELL RESULTS (N = 10 SEEDS) ---")
    summary_table = []
    for c_name in cells:
        m_h5 = float(np.mean(cell_results[c_name]))
        s_h5 = float(np.std(cell_results[c_name]))
        mean_u = float(-(m_h5 - mean_d0_h5))
        delta_m0_pct = (m_h5 - base_h5) / base_h5 * 100
        delta_d0_pct = (m_h5 - mean_d0_h5) / mean_d0_h5 * 100

        print(f"  Cell {c_name:<5}: H=5 Error = {m_h5:.6f} +/- {s_h5:.6f} (Utility U(S) = {mean_u:+.6f}, vs D0: {delta_d0_pct:+.2f}%)")
        summary_table.append({
            "cell": c_name,
            "mean_h5": m_h5,
            "std_h5": s_h5,
            "mean_utility": mean_u,
            "delta_d0_pct": delta_d0_pct,
            "diagnostics": cell_diagnostics[c_name],
        })

    # 4. OLS Factorial Regression Analysis
    df_reg = pd.DataFrame(regression_records)
    X = df_reg[["risk", "coverage", "risk_x_cov"]]
    X = sm.add_constant(X)
    y = df_reg["utility_u_s"]
    model = sm.OLS(y, X).fit()

    print("\n================================================================================")
    print("   STAGE N1 OLS FACTORIAL REGRESSION RESULTS                                   ")
    print("================================================================================")
    print(model.summary())

    beta_0 = float(model.params["const"])
    beta_R = float(model.params["risk"])
    beta_C = float(model.params["coverage"])
    beta_RC = float(model.params["risk_x_cov"])

    p_R = float(model.pvalues["risk"])
    p_C = float(model.pvalues["coverage"])
    p_RC = float(model.pvalues["risk_x_cov"])

    ci_R = [float(model.conf_int().loc["risk"][0]), float(model.conf_int().loc["risk"][1])]
    ci_C = [float(model.conf_int().loc["coverage"][0]), float(model.conf_int().loc["coverage"][1])]
    ci_RC = [float(model.conf_int().loc["risk_x_cov"][0]), float(model.conf_int().loc["risk_x_cov"][1])]

    # Determine Gate Outcome
    if p_RC < 0.05 and beta_RC > 0:
        verdict = "CONDITIONAL_UTILITY_CONFIRMED: Decision relevance is useful ONLY when combined with high coverage"
    elif p_C < 0.05 and beta_C > 0 and p_R >= 0.05 and p_RC >= 0.05:
        verdict = "COVERAGE_DOMINATES: State-action support coverage drives utility; risk score has no marginal contribution"
    elif p_R >= 0.05 and p_C >= 0.05 and p_RC >= 0.05:
        verdict = "DECOUPLED_ZERO_MARGINAL_UTILITY: Neither pointwise risk nor coverage separates utility at B=200 under two-stream replay"
    else:
        verdict = "COMPLEX_FACTORIAL_EFFECT"

    print(f"\n================================================================================")
    print(f"   STAGE N1 VERDICT: [{verdict}]")
    print(f"================================================================================")

    # Generate Figures
    generate_stage_n1_figures(summary_table, mean_d0_h5, df_reg, figure_dir)

    out_json = {
        "verdict": verdict,
        "base_reference_h5": base_h5,
        "d0_two_stream_h5": mean_d0_h5,
        "summary_table": summary_table,
        "regression_results": {
            "beta_0": beta_0,
            "beta_risk": beta_R,
            "p_risk": p_R,
            "ci_risk": ci_R,
            "beta_coverage": beta_C,
            "p_coverage": p_C,
            "ci_coverage": ci_C,
            "beta_risk_x_cov": beta_RC,
            "p_risk_x_cov": p_RC,
            "ci_risk_x_cov": ci_RC,
            "r_squared": float(model.rsquared),
            "f_pvalue": float(model.f_pvalue),
        },
    }

    report_json_path = os.path.join(output_dir, "stage_n1_orthogonalized_utility_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage N1 Report to {report_json_path}")

    env.close()
    return out_json


def generate_stage_n1_figures(
    summary_table: List[Dict],
    d0_h5: float,
    df_reg: pd.DataFrame,
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cells = [r["cell"] for r in summary_table]
    h5_means = [r["mean_h5"] for r in summary_table]
    h5_stds = [r["std_h5"] for r in summary_table]
    utilities = [r["mean_utility"] for r in summary_table]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    # (a) 2x2 Cell H=5 Rollout Error
    axes[0].bar(cells, h5_means, yerr=h5_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[0].axhline(d0_h5, color="black", linestyle="--", lw=1.5, label=f"D0-Only ({d0_h5:.4f})")
    axes[0].set_ylabel("$H=5$ Rollout Error")
    axes[0].set_title("(a) 2x2 Orthogonalized Factorial H=5 Error")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # (b) Interaction Plot (Coverage vs Realized Utility grouped by Risk)
    hr_data = df_reg[df_reg["risk"] == 1.0]
    lr_data = df_reg[df_reg["risk"] == 0.0]

    hr_u_lc = float(hr_data[hr_data["coverage"] == 0.0]["utility_u_s"].mean())
    hr_u_hc = float(hr_data[hr_data["coverage"] == 1.0]["utility_u_s"].mean())
    lr_u_lc = float(lr_data[lr_data["coverage"] == 0.0]["utility_u_s"].mean())
    lr_u_hc = float(lr_data[lr_data["coverage"] == 1.0]["utility_u_s"].mean())

    axes[1].plot([0, 1], [hr_u_lc, hr_u_hc], marker="o", label="High Risk", color="#d62728", lw=2)
    axes[1].plot([0, 1], [lr_u_lc, lr_u_hc], marker="s", label="Low Risk", color="#1f77b4", lw=2)
    axes[1].axhline(0, color="black", linestyle="--", lw=1.2)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Low Coverage (LC)", "High Coverage (HC)"])
    axes[1].set_ylabel("Realized Utility $U(S) = -\\Delta H=5$")
    axes[1].set_title("(b) Risk $\\times$ Coverage Interaction on Utility")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_n1_orthogonalized_utility.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage N1 Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_n1_benchmark()
