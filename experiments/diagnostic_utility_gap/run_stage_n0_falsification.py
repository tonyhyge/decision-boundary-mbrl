"""
Stage N0: Diagnostic-to-Utility Gap Falsification Benchmark on Hopper-v4.

Core Scientific Question:
  Why does a strong pointwise decision-risk signal (sigma_grad_Q) fail to produce
  useful world-model training subsets at B = 200 on Hopper-v4?

Three Competing Hypotheses:
  - H1 (Budget Insufficiency): B = 200 is below a non-linear information threshold.
      Evaluated across B in {50, 100, 200, 400, 600, 807} under STRICTLY MATCHED
      optimizer step counts (T_steps = 650 updates for all sizes).
  - H2 (Top-Risk Redundancy / Coverage Collapse): Top-risk queries collapse into a narrow,
      spatially redundant cluster. Evaluates Top-Risk vs Random vs Cluster-Stratified Risk
      vs Diversity-Only (k-Center) at B = 200, with spatial dispersion metrics.
  - H3 (Pointwise Risk != Training Utility Decoupling): Even under matched budget and
      random variations, subset aggregate risk mass does not predict realized post-update
      utility U(S) = -Delta H5. Evaluated over M = 25 random subsets.
"""
import os
import sys
import json
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans
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
    evaluate_rollout_error_h5,
)
from experiments.active_mbpo.run_stage4b1r_repairability_closure import evaluate_stratified_errors


def train_ensemble_matched_steps(
    ensemble: HopperEnsembleDynamics,
    x_data: np.ndarray,
    y_data: np.ndarray,
    total_steps: int = 650,
    batch_size: int = 64,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> None:
    """Trains ensemble for an EXACT fixed number of gradient steps regardless of dataset size."""
    n_samples = len(x_data)
    criterion = nn.MSELoss()

    for k, model in enumerate(ensemble.models):
        torch.manual_seed(seed + k * 100)
        rng = np.random.default_rng(seed + k * 100)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()

        for step in range(total_steps):
            # Sample batch with replacement
            idx = rng.choice(n_samples, size=min(batch_size, n_samples), replace=True)
            bx = torch.tensor(x_data[idx], dtype=torch.float32)
            by = torch.tensor(y_data[idx], dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()


def compute_subset_spatial_metrics(
    x_subset: np.ndarray,
    x_pool: np.ndarray,
    n_clusters: int = 10,
) -> Dict[str, float]:
    """Computes spatial dispersion, pairwise distance, and cluster entropy of a subset."""
    # 1. Mean Pairwise Euclidean Distance
    pdist = pairwise_distances(x_subset)
    np.fill_diagonal(pdist, np.nan)
    mean_pdist = float(np.nanmean(pdist))

    # 2. Cluster Entropy across Pool Clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(x_pool)
    cluster_labels = kmeans.predict(x_subset)
    counts = np.bincount(cluster_labels, minlength=n_clusters)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(n_clusters))
    normalized_entropy = entropy / max_entropy

    # 3. Coverage Radius (mean distance to nearest neighbor in subset from pool)
    dist_to_subset = pairwise_distances(x_pool, x_subset)
    min_dist_to_subset = np.min(dist_to_subset, axis=1)
    mean_coverage_radius = float(np.mean(min_dist_to_subset))

    return {
        "mean_pairwise_dist": mean_pdist,
        "cluster_entropy": normalized_entropy,
        "coverage_radius": mean_coverage_radius,
    }


def select_k_center_diversity(x_pool: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    """Greedy k-Center / Farthest-First-Traversal coreset selection for maximum spatial diversity."""
    rng = np.random.default_rng(seed)
    n_samples = len(x_pool)
    first_idx = rng.integers(0, n_samples)
    selected = [first_idx]

    min_dists = pairwise_distances(x_pool, x_pool[first_idx:first_idx+1]).flatten()

    for _ in range(1, budget):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = pairwise_distances(x_pool, x_pool[next_idx:next_idx+1]).flatten()
        min_dists = np.minimum(min_dists, new_dists)

    return np.array(selected, dtype=int)


def select_cluster_stratified_risk(
    x_pool: np.ndarray,
    scores: np.ndarray,
    budget: int,
    n_clusters: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Partitions state space into clusters and selects top-risk points within each cluster."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(x_pool)
    cluster_labels = kmeans.labels_

    per_cluster_budget = budget // n_clusters
    remainder = budget % n_clusters
    selected = []

    for c in range(n_clusters):
        c_indices = np.where(cluster_labels == c)[0]
        c_scores = scores[c_indices]
        n_take = per_cluster_budget + (1 if c < remainder else 0)
        top_local = c_indices[np.argsort(-c_scores)[:min(n_take, len(c_indices))]]
        selected.extend(top_local.tolist())

    # Fallback if some clusters had fewer samples
    if len(selected) < budget:
        remaining_pool = list(set(range(len(x_pool))) - set(selected))
        rem_scores = scores[remaining_pool]
        top_rem = np.array(remaining_pool)[np.argsort(-rem_scores)[:budget - len(selected)]]
        selected.extend(top_rem.tolist())

    return np.array(selected[:budget], dtype=int)


def run_stage_n0_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    n_seeds: int = 10,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE N0: DIAGNOSTIC-TO-UTILITY GAP FALSIFICATION BENCHMARK (HOPPER-v4)     ")
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
    n_d0 = len(x_base)

    # Initial Reference Model M_0
    base_ensemble = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed)
    base_ensemble.train_ensemble(x_base, y_base, epochs=50, batch_size=64, lr=0.001, seed=seed)

    pool_trajs = collect_hopper_trajectories(env, num_trajectories=30, max_steps=100, seed=seed + 100)
    eval_trajs = collect_hopper_trajectories(env, num_trajectories=25, max_steps=100, seed=seed + 200)

    pool_samples = []
    x_pool_arr, y_pool_arr = [], []
    for tr in pool_trajs:
        for t in range(len(tr["obs"])):
            sa = np.concatenate([tr["obs"][t], tr["acts"][t]])
            dy = tr["next_obs"][t] - tr["obs"][t]
            x_pool_arr.append(sa)
            y_pool_arr.append(dy)
            pool_samples.append({
                "obs": tr["obs"][t],
                "act": tr["acts"][t],
                "next_obs": tr["next_obs"][t],
                "qpos": tr["qpos"][t],
                "qvel": tr["qvel"][t],
            })
    df_pool = pd.DataFrame(pool_samples)
    x_pool_arr = np.array(x_pool_arr, dtype=np.float32)
    y_pool_arr = np.array(y_pool_arr, dtype=np.float32)
    n_pool = len(df_pool)

    base_eval_metrics = evaluate_stratified_errors(base_ensemble, eval_trajs, horizon=5)
    base_h5 = base_eval_metrics["e_h5"]
    print(f"-> Base Dataset D_0: {n_d0} samples | Candidate Pool: {n_pool} states | Eval Set: {sum(len(tr['obs']) for tr in eval_trajs)} states.")
    print(f"-> Reference Model M_0 Baseline H=5 Rollout Error: {base_h5:.6f}")

    # Extract diagnostic signals on pool
    print("\n[Step 2] Extracting Candidate Pool Action-Gradient and Dynamics Uncertainty...")
    critic = HopperHeuristicCritic(ensemble=base_ensemble)
    sigma_dyn_list, sigma_grad_q_list, true_res_list = [], [], []

    for i in range(n_pool):
        s = df_pool.iloc[i]["obs"]
        a = df_pool.iloc[i]["act"]
        true_ns = df_pool.iloc[i]["next_obs"]

        pred_all = base_ensemble.predict_all(s, a)
        mean_pred = np.mean(pred_all, axis=0)
        sigma_dyn_list.append(float(np.mean(np.std(pred_all, axis=0, ddof=1))))
        grads_k = critic.estimate_action_gradient(s, a)
        sigma_grad_q_list.append(float(np.mean(np.std(grads_k, axis=0, ddof=1))))
        true_res_list.append(float(np.linalg.norm(true_ns - mean_pred)))

    df_pool["sigma_dyn"] = np.array(sigma_dyn_list)
    df_pool["sigma_grad_q"] = np.array(sigma_grad_q_list)
    df_pool["true_residual"] = np.array(true_res_list)

    # -------------------------------------------------------------------------
    # PART 1: HYPOTHESIS H1 — BUDGET INSUFFICIENCY (STRICTLY MATCHED STEPS = 650)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PART 1: HYPOTHESIS H1 — BUDGET SWEEP (STRICTLY MATCHED T_steps = 650)       ")
    print("================================================================================")

    budget_levels = [50, 100, 200, 400, 600, 807]
    h1_results = []
    n_seeds_h1 = 5

    # D0-Only Matched Steps Baseline
    d0_h5_seeds = []
    for s_idx in range(n_seeds_h1):
        seed_val = seed + s_idx * 13
        ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
        train_ensemble_matched_steps(ens, x_base, y_base, total_steps=650, batch_size=64, lr=0.003, seed=seed_val)
        ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)
        d0_h5_seeds.append(ev["e_h5"])
    d0_matched_mean_h5 = float(np.mean(d0_h5_seeds))
    print(f"-> D0-Only Matched-Step Baseline (T=650): H=5 Error = {d0_matched_mean_h5:.6f}")

    for b in budget_levels:
        rand_h5, grad_h5, dyn_h5, oracle_h5 = [], [], [], []

        for s_idx in range(n_seeds_h1):
            seed_val = seed + s_idx * 17 + b
            rng_b = np.random.default_rng(seed_val)

            idx_rand = rng_b.choice(n_pool, size=min(b, n_pool), replace=False)
            idx_grad = np.argsort(-df_pool["sigma_grad_q"].to_numpy())[:min(b, n_pool)]
            idx_dyn = np.argsort(-df_pool["sigma_dyn"].to_numpy())[:min(b, n_pool)]
            idx_oracle = np.argsort(-df_pool["true_residual"].to_numpy())[:min(b, n_pool)]

            for strat_name, q_idx, store_list in [
                ("Random", idx_rand, rand_h5),
                ("Action-Grad", idx_grad, grad_h5),
                ("Dynamics", idx_dyn, dyn_h5),
                ("Privileged", idx_oracle, oracle_h5),
            ]:
                x_aug = np.concatenate([x_base, x_pool_arr[q_idx]], axis=0)
                y_aug = np.concatenate([y_base, y_pool_arr[q_idx]], axis=0)
                ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
                train_ensemble_matched_steps(ens, x_aug, y_aug, total_steps=650, batch_size=64, lr=0.003, seed=seed_val)
                ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)
                store_list.append(ev["e_h5"])

        print(f"  Budget B={b:>3}: Random = {np.mean(rand_h5):.6f} | Action-Grad = {np.mean(grad_h5):.6f} | Dyn = {np.mean(dyn_h5):.6f} | Privileged = {np.mean(oracle_h5):.6f}")
        h1_results.append({
            "budget": b,
            "mean_rand_h5": float(np.mean(rand_h5)),
            "mean_grad_h5": float(np.mean(grad_h5)),
            "mean_dyn_h5": float(np.mean(dyn_h5)),
            "mean_priv_h5": float(np.mean(oracle_h5)),
            "d0_baseline_h5": d0_matched_mean_h5,
        })

    # -------------------------------------------------------------------------
    # PART 2: HYPOTHESIS H2 — TOP-RISK REDUNDANCY & CLUSTER STRATIFICATION (B = 200)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PART 2: HYPOTHESIS H2 — COVERAGE COLLAPSE & STRATIFICATION (B = 200, N = 10)")
    print("================================================================================")

    h2_strategies = [
        "1_top_risk_action_grad",
        "2_random_uniform",
        "3_cluster_stratified_risk",
        "4_diversity_only_kcenter",
    ]

    h2_records = {s: [] for s in h2_strategies}
    spatial_metrics_map = {}

    idx_top_risk = np.argsort(-df_pool["sigma_grad_q"].to_numpy())[:200]
    idx_kcenter = select_k_center_diversity(x_pool_arr, budget=200, seed=seed)
    idx_stratified = select_cluster_stratified_risk(x_pool_arr, df_pool["sigma_grad_q"].to_numpy(), budget=200, n_clusters=10, seed=seed)

    spatial_metrics_map["1_top_risk_action_grad"] = compute_subset_spatial_metrics(x_pool_arr[idx_top_risk], x_pool_arr)
    spatial_metrics_map["3_cluster_stratified_risk"] = compute_subset_spatial_metrics(x_pool_arr[idx_stratified], x_pool_arr)
    spatial_metrics_map["4_diversity_only_kcenter"] = compute_subset_spatial_metrics(x_pool_arr[idx_kcenter], x_pool_arr)

    print(f"  Spatial Metrics:")
    print(f"    Top-Risk Action-Grad   : Mean Pairwise Dist = {spatial_metrics_map['1_top_risk_action_grad']['mean_pairwise_dist']:.4f} | Cluster Entropy = {spatial_metrics_map['1_top_risk_action_grad']['cluster_entropy']:.4f} | Coverage Rad = {spatial_metrics_map['1_top_risk_action_grad']['coverage_radius']:.4f}")
    print(f"    Stratified Action-Grad : Mean Pairwise Dist = {spatial_metrics_map['3_cluster_stratified_risk']['mean_pairwise_dist']:.4f} | Cluster Entropy = {spatial_metrics_map['3_cluster_stratified_risk']['cluster_entropy']:.4f} | Coverage Rad = {spatial_metrics_map['3_cluster_stratified_risk']['coverage_radius']:.4f}")
    print(f"    Diversity-Only (k-Center): Mean Pairwise Dist = {spatial_metrics_map['4_diversity_only_kcenter']['mean_pairwise_dist']:.4f} | Cluster Entropy = {spatial_metrics_map['4_diversity_only_kcenter']['cluster_entropy']:.4f} | Coverage Rad = {spatial_metrics_map['4_diversity_only_kcenter']['coverage_radius']:.4f}")

    for s_idx in range(n_seeds):
        seed_val = seed + s_idx * 29
        rng_h2 = np.random.default_rng(seed_val)

        idx_rand_s = rng_h2.choice(n_pool, size=200, replace=False)
        strat_map = {
            "1_top_risk_action_grad": idx_top_risk,
            "2_random_uniform": idx_rand_s,
            "3_cluster_stratified_risk": idx_stratified,
            "4_diversity_only_kcenter": idx_kcenter,
        }

        for s_name, q_idx in strat_map.items():
            x_aug = np.concatenate([x_base, x_pool_arr[q_idx]], axis=0)
            y_aug = np.concatenate([y_base, y_pool_arr[q_idx]], axis=0)
            ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
            train_ensemble_matched_steps(ens, x_aug, y_aug, total_steps=650, batch_size=64, lr=0.003, seed=seed_val)
            ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)
            h2_records[s_name].append(ev["e_h5"])

    print("\n--- STAGE H2 EMPIRICAL COMPARISON (N = 10 SEEDS, MATCHED STEPS) ---")
    for s_name in h2_strategies:
        m_h5 = float(np.mean(h2_records[s_name]))
        s_h5 = float(np.std(h2_records[s_name]))
        delta_m0_pct = (m_h5 - base_h5) / base_h5 * 100
        print(f"  {s_name:<30}: H=5 Error = {m_h5:.6f} +/- {s_h5:.6f} ({delta_m0_pct:+.2f}% vs M0)")

    t_strat_vs_top, p_strat_vs_top = stats.ttest_rel(
        h2_records["3_cluster_stratified_risk"], h2_records["1_top_risk_action_grad"]
    )
    t_strat_vs_rand, p_strat_vs_rand = stats.ttest_rel(
        h2_records["3_cluster_stratified_risk"], h2_records["2_random_uniform"]
    )
    print(f"\n  Stratified Risk vs Top-Risk     : Paired Delta = {np.mean(np.array(h2_records['3_cluster_stratified_risk']) - np.array(h2_records['1_top_risk_action_grad'])):+.6f} (t = {t_strat_vs_top:.3f}, p = {p_strat_vs_top:.4e})")
    print(f"  Stratified Risk vs Random Uniform: Paired Delta = {np.mean(np.array(h2_records['3_cluster_stratified_risk']) - np.array(h2_records['2_random_uniform'])):+.6f} (t = {t_strat_vs_rand:.3f}, p = {p_strat_vs_rand:.4e})")

    # -------------------------------------------------------------------------
    # PART 3: HYPOTHESIS H3 — SUBSET-LEVEL UTILITY PREDICTION (M = 25 SUBSETS)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PART 3: HYPOTHESIS H3 — SUBSET RISK MASS VS REALIZED UTILITY (M = 25)       ")
    print("================================================================================")

    m_subsets = 25
    h3_records = []

    for j in range(m_subsets):
        rng_j = np.random.default_rng(seed + j * 47)
        # Sample subset of size 200 with varying temperature on risk to create spectrum of risk mass
        temp = float(rng_j.uniform(0.1, 5.0))
        logits = df_pool["sigma_grad_q"].to_numpy() / temp
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        idx_j = rng_j.choice(n_pool, size=200, replace=False, p=probs)

        sub_grad_mass = float(np.mean(df_pool.iloc[idx_j]["sigma_grad_q"]))
        sub_dyn_mass = float(np.mean(df_pool.iloc[idx_j]["sigma_dyn"]))
        sub_res_mass = float(np.mean(df_pool.iloc[idx_j]["true_residual"]))
        sub_spatial = compute_subset_spatial_metrics(x_pool_arr[idx_j], x_pool_arr)

        # Train model on D0 U S_j with matched steps
        x_aug = np.concatenate([x_base, x_pool_arr[idx_j]], axis=0)
        y_aug = np.concatenate([y_base, y_pool_arr[idx_j]], axis=0)
        ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed + j)
        train_ensemble_matched_steps(ens, x_aug, y_aug, total_steps=650, batch_size=64, lr=0.003, seed=seed + j)
        ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)

        # Utility U(S) = -(H5 - d0_matched_baseline)
        utility = float(-(ev["e_h5"] - d0_matched_mean_h5))

        h3_records.append({
            "subset_id": j,
            "mean_risk_mass": sub_grad_mass,
            "mean_dyn_mass": sub_dyn_mass,
            "mean_res_mass": sub_res_mass,
            "mean_pairwise_dist": sub_spatial["mean_pairwise_dist"],
            "cluster_entropy": sub_spatial["cluster_entropy"],
            "coverage_radius": sub_spatial["coverage_radius"],
            "h5_error": ev["e_h5"],
            "utility_u_s": utility,
        })

    df_h3 = pd.DataFrame(h3_records)

    # Compute correlation with utility
    corr_risk_u = stats.pearsonr(df_h3["mean_risk_mass"], df_h3["utility_u_s"])
    corr_dyn_u = stats.pearsonr(df_h3["mean_dyn_mass"], df_h3["utility_u_s"])
    corr_res_u = stats.pearsonr(df_h3["mean_res_mass"], df_h3["utility_u_s"])
    corr_entropy_u = stats.pearsonr(df_h3["cluster_entropy"], df_h3["utility_u_s"])
    corr_dist_u = stats.pearsonr(df_h3["mean_pairwise_dist"], df_h3["utility_u_s"])

    print(f"  Correlations with Realized Training Utility U(S):")
    print(f"    Subset Mean Action-Gradient Risk : r = {corr_risk_u.statistic:+.4f} (p = {corr_risk_u.pvalue:.4e})")
    print(f"    Subset Mean Dynamics Uncertainty : r = {corr_dyn_u.statistic:+.4f} (p = {corr_dyn_u.pvalue:.4e})")
    print(f"    Subset Mean True Residual        : r = {corr_res_u.statistic:+.4f} (p = {corr_res_u.pvalue:.4e})")
    print(f"    Subset Spatial Cluster Entropy   : r = {corr_entropy_u.statistic:+.4f} (p = {corr_entropy_u.pvalue:.4e})")
    print(f"    Subset Mean Pairwise Distance    : r = {corr_dist_u.statistic:+.4f} (p = {corr_dist_u.pvalue:.4e})")

    # -------------------------------------------------------------------------
    # 5. SYNTHESIS & STAGE N0 VERDICT
    # -------------------------------------------------------------------------
    if corr_risk_u.pvalue >= 0.05 or corr_risk_u.statistic <= 0.1:
        if corr_entropy_u.statistic > 0.3 and corr_entropy_u.pvalue < 0.05:
            verdict = "H2_SUPPORTED: COVERAGE_ENTROPY_DOMINATES_POINTWISE_RISK"
        else:
            verdict = "H3_SUPPORTED: POINTWISE_RISK_DECOUPLED_FROM_SET_UTILITY"
    else:
        verdict = "H1_SUPPORTED: BUDGET_THRESHOLD_EFFECT"

    print(f"\n================================================================================")
    print(f"   STAGE N0 VERDICT: [{verdict}]")
    print(f"================================================================================")

    # Generate Figures
    generate_stage_n0_figures(h1_results, h2_records, df_h3, figure_dir)

    out_json = {
        "verdict": verdict,
        "base_reference_h5": base_h5,
        "d0_matched_mean_h5": d0_matched_mean_h5,
        "h1_budget_sweep": h1_results,
        "h2_spatial_stratification": {
            "strategies": {s: {"mean_h5": float(np.mean(h2_records[s])), "std_h5": float(np.std(h2_records[s]))} for s in h2_strategies},
            "spatial_metrics": spatial_metrics_map,
            "paired_strat_vs_top": {"t_stat": float(t_strat_vs_top), "p_val": float(p_strat_vs_top)},
            "paired_strat_vs_rand": {"t_stat": float(t_strat_vs_rand), "p_val": float(p_strat_vs_rand)},
        },
        "h3_utility_correlations": {
            "risk_vs_utility": {"r": float(corr_risk_u.statistic), "p": float(corr_risk_u.pvalue)},
            "dyn_vs_utility": {"r": float(corr_dyn_u.statistic), "p": float(corr_dyn_u.pvalue)},
            "res_vs_utility": {"r": float(corr_res_u.statistic), "p": float(corr_res_u.pvalue)},
            "entropy_vs_utility": {"r": float(corr_entropy_u.statistic), "p": float(corr_entropy_u.pvalue)},
            "pairwise_dist_vs_utility": {"r": float(corr_dist_u.statistic), "p": float(corr_dist_u.pvalue)},
        },
    }

    report_json_path = os.path.join(output_dir, "stage_n0_gap_falsification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage N0 Report to {report_json_path}")

    env.close()
    return out_json


def generate_stage_n0_figures(
    h1_records: List[Dict],
    h2_records: Dict[str, List[float]],
    df_h3: pd.DataFrame,
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Part 1: Budget Sweep Curve under Matched Steps
    df_b = pd.DataFrame(h1_records)
    axes[0].plot(df_b["budget"], df_b["mean_rand_h5"], marker="o", label="Random", color="#1f77b4", lw=2)
    axes[0].plot(df_b["budget"], df_b["mean_grad_h5"], marker="s", label="Action-Grad", color="#d62728", lw=2)
    axes[0].plot(df_b["budget"], df_b["mean_dyn_h5"], marker="^", label="Dynamics", color="#ff7f0e", lw=2)
    axes[0].plot(df_b["budget"], df_b["mean_priv_h5"], marker="d", label="Privileged", color="#2ca02c", lw=2)
    axes[0].axhline(df_b["d0_baseline_h5"].iloc[0], color="black", linestyle="--", label="D0-Only (T=650)")
    axes[0].set_xlabel("Query Budget $B$")
    axes[0].set_ylabel("$H=5$ Rollout Error")
    axes[0].set_title("(a) Matched-Step Budget Curve (H1)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Part 2: Spatial Redundancy vs Cluster-Stratification (B=200)
    names = ["Top-Risk\n(Action-Grad)", "Random\nUniform", "Stratified\nRisk", "Diversity\n(k-Center)"]
    means = [np.mean(h2_records[k]) for k in ["1_top_risk_action_grad", "2_random_uniform", "3_cluster_stratified_risk", "4_diversity_only_kcenter"]]
    stds = [np.std(h2_records[k]) for k in ["1_top_risk_action_grad", "2_random_uniform", "3_cluster_stratified_risk", "4_diversity_only_kcenter"]]
    colors = ["#d62728", "#1f77b4", "#9467bd", "#2ca02c"]

    axes[1].bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_ylabel("$H=5$ Rollout Error")
    axes[1].set_title("(b) Spatial Redundancy Intervention (H2)")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    # (c) Part 3: Subset Risk Mass vs Realized Utility U(S)
    axes[2].scatter(df_h3["mean_risk_mass"], df_h3["utility_u_s"], color="#d62728", alpha=0.85, s=50, edgecolors="black")
    m, b = np.polyfit(df_h3["mean_risk_mass"], df_h3["utility_u_s"], 1)
    x_vals = np.linspace(df_h3["mean_risk_mass"].min(), df_h3["mean_risk_mass"].max(), 50)
    axes[2].plot(x_vals, m * x_vals + b, color="black", linestyle="--", lw=1.5)
    axes[2].axhline(0, color="gray", linestyle=":")
    axes[2].set_xlabel("Subset Mean Risk Mass $\\bar{r}(S)$")
    axes[2].set_ylabel("Realized Utility $U(S) = -\\Delta H=5$")
    axes[2].set_title("(c) Risk Mass vs Training Utility (H3)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_n0_gap_falsification.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage N0 Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_n0_benchmark()
