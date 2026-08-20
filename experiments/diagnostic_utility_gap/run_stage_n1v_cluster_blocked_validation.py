"""
Stage N1-V: Cluster-Blocked Interaction Validation & Evaluation Occupancy Audit.

Core Scientific Question:
  Does high decision-risk data retain positive marginal training utility when
  cluster identity, state-action support, and local density are strictly held constant?
  Or was the HR/LC advantage in Stage N1 merely an artifact of selecting regions
  with higher evaluation occupancy / baseline error?

Sub-Experiments:
  1. Cluster-Blocked Risk Comparison across K = 5 Distinct Cluster Pairs:
     Within the EXACT SAME cluster pair P_k = (c_i, c_j), compares:
       HR/LC_{P_k} (Top risk in P_k) vs LR/LC_{P_k} (Bottom risk in P_k)
     across N = 5 independent seeds per pair (50 total model runs + references).
  2. Evaluation Occupancy & Baseline Error Audit across all 10 clusters:
     Measures evaluation weight w_c = P_eval[(s, a) in c], baseline model error E_c,
     and mean risk bar{r}_c to test for regional confounding.
  3. Pre-Registered Contrast Statistics from Stage N1:
     Computes exact paired seed CIs and p-values for Delta_R^LC, Delta_R^HC, Delta_C^HR, Delta_C^LR.
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
from statsmodels.formula.api import ols
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
from experiments.diagnostic_utility_gap.run_stage_n1_orthogonalized_utility import train_ensemble_two_stream


def run_stage_n1v_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    n_seeds_per_pair: int = 5,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE N1-V: CLUSTER-BLOCKED VALIDATION & OCCUPANCY AUDIT (HOPPER-v4)        ")
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

    # Base Reference Model M_0
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

    x_eval, y_eval = [], []
    for tr in eval_trajs:
        for t in range(len(tr["obs"])):
            x_eval.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_eval.append(tr["next_obs"][t] - tr["obs"][t])
    x_eval = np.array(x_eval, dtype=np.float32)
    y_eval = np.array(y_eval, dtype=np.float32)

    base_eval_metrics = evaluate_stratified_errors(base_ensemble, eval_trajs, horizon=5)
    base_h5 = base_eval_metrics["e_h5"]
    print(f"-> Base Dataset D_0: {len(x_base)} samples | Candidate Pool: {n_pool} states | Eval Set: {len(x_eval)} states.")
    print(f"-> Reference Model M_0 Baseline H=5 Error: {base_h5:.6f}")

    # Extract Candidate Pool Action-Gradient Risk Scores
    print("\n[Step 2] Extracting Candidate Pool Action-Gradient Risk Scores...")
    critic = HopperHeuristicCritic(ensemble=base_ensemble)
    sigma_grad_q_list = []
    for i in range(n_pool):
        s = x_pool[i][:11]
        a = x_pool[i][11:]
        grads_k = critic.estimate_action_gradient(s, a)
        sigma_grad_q_list.append(float(np.mean(np.std(grads_k, axis=0, ddof=1))))
    scores_pool = np.array(sigma_grad_q_list)

    # -------------------------------------------------------------------------
    # 2. SUB-EXPERIMENT 2: EVALUATION OCCUPANCY & BASELINE ERROR AUDIT
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   SUB-EXPERIMENT 2: EVALUATION OCCUPANCY & BASELINE ERROR AUDIT (10 CLUSTERS)  ")
    print("================================================================================")

    scaler = StandardScaler()
    scaler.fit(np.concatenate([x_pool, x_eval], axis=0))
    x_pool_norm = scaler.transform(x_pool)
    x_eval_norm = scaler.transform(x_eval)

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(x_pool_norm)
    pool_labels = kmeans.labels_
    eval_labels = kmeans.predict(x_eval_norm)

    cluster_audit_records = []
    for c in range(n_clusters):
        c_pool_idx = np.where(pool_labels == c)[0]
        c_eval_idx = np.where(eval_labels == c)[0]

        n_pool_c = len(c_pool_idx)
        w_c = len(c_eval_idx) / len(x_eval)  # Evaluation occupancy mass
        mean_risk_c = float(np.mean(scores_pool[c_pool_idx])) if n_pool_c > 0 else 0.0

        # Baseline model prediction MSE on cluster c
        if n_pool_c > 0:
            c_preds = [
                np.mean(base_ensemble.predict_all(x_pool[i][:11], x_pool[i][11:]), axis=0)
                for i in c_pool_idx
            ]
            c_targets = [x_pool[i][:11] + y_pool[i] for i in c_pool_idx]
            err_c = float(np.mean([np.mean((p - t) ** 2) for p, t in zip(c_preds, c_targets)]))
        else:
            err_c = 0.0

        cluster_audit_records.append({
            "cluster_id": c,
            "n_pool_samples": n_pool_c,
            "eval_occupancy_wc": w_c,
            "mean_risk_rc": mean_risk_c,
            "baseline_error_ec": err_c,
        })
        print(f"  Cluster {c:>2}: Pool Size = {n_pool_c:>3} | Eval Occupancy w_c = {w_c*100:>5.1f}% | Mean Risk = {mean_risk_c:.4f} | Baseline MSE = {err_c:.6f}")

    df_cluster_audit = pd.DataFrame(cluster_audit_records)

    corr_risk_occ = stats.pearsonr(df_cluster_audit["mean_risk_rc"], df_cluster_audit["eval_occupancy_wc"])
    corr_risk_err = stats.pearsonr(df_cluster_audit["mean_risk_rc"], df_cluster_audit["baseline_error_ec"])
    print(f"\n  Cluster Audit Correlations:")
    print(f"    Mean Risk vs Eval Occupancy w_c : r = {corr_risk_occ.statistic:+.4f} (p = {corr_risk_occ.pvalue:.4e})")
    print(f"    Mean Risk vs Baseline Error E_c : r = {corr_risk_err.statistic:+.4f} (p = {corr_risk_err.pvalue:.4e})")

    # -------------------------------------------------------------------------
    # 3. SUB-EXPERIMENT 1: CLUSTER-BLOCKED RISK COMPARISON ACROSS K = 5 PAIRS
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   SUB-EXPERIMENT 1: CLUSTER-BLOCKED RISK COMPARISON (K = 5 CLUSTER PAIRS)     ")
    print("================================================================================")

    # Rank clusters by sample count and select 5 distinct pairs with sufficient size
    sorted_clusters = np.argsort(-df_cluster_audit["n_pool_samples"].to_numpy())
    cluster_pairs = [
        (int(sorted_clusters[0]), int(sorted_clusters[1])),
        (int(sorted_clusters[2]), int(sorted_clusters[3])),
        (int(sorted_clusters[4]), int(sorted_clusters[5])),
        (int(sorted_clusters[6]), int(sorted_clusters[7])),
        (int(sorted_clusters[0]), int(sorted_clusters[2])),
    ]

    # Baseline D0-only reference (matched T_steps=650, Two-Stream)
    d0_h5_seeds = []
    for s_idx in range(n_seeds_per_pair):
        seed_val = seed + s_idx * 13
        ens = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
        train_ensemble_two_stream(ens, x_base, y_base, None, None, total_steps=650, lr=0.003, seed=seed_val)
        ev = evaluate_stratified_errors(ens, eval_trajs, horizon=5)
        d0_h5_seeds.append(ev["e_h5"])
    mean_d0_h5 = float(np.mean(d0_h5_seeds))
    print(f"-> D0-Only Matched-Step Baseline (Two-Stream, T=650): H=5 Error = {mean_d0_h5:.6f}")

    pair_blocked_results = []
    pair_summary = []

    for p_idx, (c1, c2) in enumerate(cluster_pairs):
        pair_name = f"P{p_idx+1}_(C{c1}+C{c2})"
        c1_idx = np.where(pool_labels == c1)[0]
        c2_idx = np.where(pool_labels == c2)[0]
        comb_idx = np.concatenate([c1_idx, c2_idx])
        comb_scores = scores_pool[comb_idx]

        # Construct High-Risk subset (Top 200 in P_k) and Low-Risk subset (Bottom 200 in P_k)
        # Allocate proportionally between c1 and c2
        budget = 200
        n1_take = int(budget * len(c1_idx) / (len(c1_idx) + len(c2_idx)))
        n2_take = budget - n1_take

        # High-Risk from P_k
        top_c1 = c1_idx[np.argsort(-scores_pool[c1_idx])[:n1_take]]
        top_c2 = c2_idx[np.argsort(-scores_pool[c2_idx])[:n2_take]]
        idx_hr_pk = np.concatenate([top_c1, top_c2])

        # Low-Risk from P_k
        bot_c1 = c1_idx[np.argsort(scores_pool[c1_idx])[:n1_take]]
        bot_c2 = c2_idx[np.argsort(scores_pool[c2_idx])[:n2_take]]
        idx_lr_pk = np.concatenate([bot_c1, bot_c2])

        mean_hr_score = float(np.mean(scores_pool[idx_hr_pk]))
        mean_lr_score = float(np.mean(scores_pool[idx_lr_pk]))

        print(f"\n  --- Testing Cluster-Pair {pair_name} ---")
        print(f"      HR Risk Score = {mean_hr_score:.4f} | LR Risk Score = {mean_lr_score:.4f} | Combined Pool Size = {len(comb_idx)}")

        hr_h5_list = []
        lr_h5_list = []

        for s_idx in range(n_seeds_per_pair):
            seed_val = seed + s_idx * 37 + p_idx * 100

            # 1. Train on HR/LC_{P_k}
            ens_hr = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
            train_ensemble_two_stream(
                ens_hr, x_base, y_base, x_pool[idx_hr_pk], y_pool[idx_hr_pk],
                total_steps=650, batch_d0=51, batch_s=13, lr=0.003, seed=seed_val
            )
            ev_hr = evaluate_stratified_errors(ens_hr, eval_trajs, horizon=5)
            hr_h5 = ev_hr["e_h5"]
            hr_h5_list.append(hr_h5)

            # 2. Train on LR/LC_{P_k}
            ens_lr = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed_val)
            train_ensemble_two_stream(
                ens_lr, x_base, y_base, x_pool[idx_lr_pk], y_pool[idx_lr_pk],
                total_steps=650, batch_d0=51, batch_s=13, lr=0.003, seed=seed_val
            )
            ev_lr = evaluate_stratified_errors(ens_lr, eval_trajs, horizon=5)
            lr_h5 = ev_lr["e_h5"]
            lr_h5_list.append(lr_h5)

            # Utility U(S) = -(H5 - mean_d0_h5)
            u_hr = -(hr_h5 - mean_d0_h5)
            u_lr = -(lr_h5 - mean_d0_h5)

            pair_blocked_results.append({
                "pair_id": p_idx + 1,
                "pair_name": pair_name,
                "seed_idx": s_idx,
                "hr_h5": hr_h5,
                "lr_h5": lr_h5,
                "hr_utility": u_hr,
                "lr_utility": u_lr,
                "delta_u_hr_minus_lr": u_hr - u_lr,
            })

        mean_hr_h5 = float(np.mean(hr_h5_list))
        mean_lr_h5 = float(np.mean(lr_h5_list))
        mean_u_hr = -(mean_hr_h5 - mean_d0_h5)
        mean_u_lr = -(mean_lr_h5 - mean_d0_h5)
        delta_u_pair = mean_u_hr - mean_u_lr
        t_pair, p_pair = stats.ttest_rel(hr_h5_list, lr_h5_list)

        print(f"      HR/LC Mean H=5 = {mean_hr_h5:.6f} (Utility = {mean_u_hr:+.6f})")
        print(f"      LR/LC Mean H=5 = {mean_lr_h5:.6f} (Utility = {mean_u_lr:+.6f})")
        print(f"      Paired Gain (HR - LR) = {delta_u_pair:+.6f} (t = {t_pair:.3f}, p = {p_pair:.4e})")

        pair_summary.append({
            "pair_name": pair_name,
            "mean_hr_h5": mean_hr_h5,
            "mean_lr_h5": mean_lr_h5,
            "mean_hr_utility": mean_u_hr,
            "mean_lr_utility": mean_u_lr,
            "delta_utility": delta_u_pair,
            "t_stat": float(t_pair),
            "p_val": float(p_pair),
        })

    # Overall Mixed-Effects / Blocked Paired Test across all 25 paired runs
    df_blocked = pd.DataFrame(pair_blocked_results)
    all_hr_u = df_blocked["hr_utility"].to_numpy()
    all_lr_u = df_blocked["lr_utility"].to_numpy()
    overall_t, overall_p = stats.ttest_rel(all_hr_u, all_lr_u)
    overall_delta_u = float(np.mean(all_hr_u - all_lr_u))
    ci_overall = stats.t.interval(0.95, df=len(all_hr_u)-1, loc=overall_delta_u, scale=stats.sem(all_hr_u - all_lr_u))

    print("\n================================================================================")
    print("   OVERALL CLUSTER-BLOCKED HYPOTHESIS TEST RESULTS (N = 25 PAIRED RUNS)        ")
    print("================================================================================")
    print(f"  Mean Utility Gain (HR/LC - LR/LC across all blocked pairs): {overall_delta_u:+.6f}")
    print(f"  95% Confidence Interval: [{ci_overall[0]:+.6f}, {ci_overall[1]:+.6f}]")
    print(f"  Paired t-statistic: {overall_t:.3f} | Two-Sided p-value: {overall_p:.4e}")

    # -------------------------------------------------------------------------
    # 4. SUB-EXPERIMENT 3: EXACT CONTRAST AUDIT FROM STAGE N1
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   SUB-EXPERIMENT 3: STAGE N1 PRE-REGISTERED SIMPLE CONTRASTS AUDIT            ")
    print("================================================================================")
    # Read Stage N1 report
    n1_report_path = os.path.join(output_dir, "stage_n1_orthogonalized_utility_report.json")
    if os.path.exists(n1_report_path):
        with open(n1_report_path, "r") as f:
            n1_json = json.load(f)
        reg_n1 = n1_json["regression_results"]
        print(f"  Stage N1 Contrast Breakdown:")
        print(f"    Delta_R^LC (HR/LC - LR/LC) : +{reg_n1['beta_risk']:.6f} [95% CI: {reg_n1['ci_risk'][0]:+.6f}, {reg_n1['ci_risk'][1]:+.6f}], p = {reg_n1['p_risk']:.4e}")
        delta_r_hc = reg_n1['beta_risk'] + reg_n1['beta_risk_x_cov']
        print(f"    Delta_R^HC (HR/HC - LR/HC) : {delta_r_hc:+.6f}")
        print(f"    Delta_C^HR (HR/HC - HR/LC) : {reg_n1['beta_coverage'] + reg_n1['beta_risk_x_cov']:+.6f}")
        print(f"    Delta_C^LR (LR/HC - LR/LC) : {reg_n1['beta_coverage']:+.6f} [95% CI: {reg_n1['ci_coverage'][0]:+.6f}, {reg_n1['ci_coverage'][1]:+.6f}], p = {reg_n1['p_coverage']:.4e}")

    # Final Gate Verdict for N1-V
    if overall_delta_u > 0 and overall_p < 0.05:
        verdict = "N1-V_PASS: CONDITIONAL_RISK_UTILITY_SURVIVES_CLUSTER_BLOCKING"
    else:
        verdict = "N1-V_FAIL: RISK_EFFECT_EXPLAINED_BY_CLUSTER_IDENTITY_CONFOUND"

    print(f"\n================================================================================")
    print(f"   STAGE N1-V GATE VERDICT: [{verdict}]")
    print(f"================================================================================")

    # Generate Figures
    generate_stage_n1v_figures(df_cluster_audit, df_blocked, pair_summary, figure_dir)

    out_json = {
        "verdict": verdict,
        "base_reference_h5": base_h5,
        "d0_two_stream_h5": mean_d0_h5,
        "cluster_audit": cluster_audit_records,
        "cluster_audit_correlations": {
            "risk_vs_occupancy": {"r": float(corr_risk_occ.statistic), "p": float(corr_risk_occ.pvalue)},
            "risk_vs_baseline_err": {"r": float(corr_risk_err.statistic), "p": float(corr_risk_err.pvalue)},
        },
        "blocked_pairs_summary": pair_summary,
        "overall_blocked_test": {
            "mean_delta_utility": overall_delta_u,
            "ci_95": [float(ci_overall[0]), float(ci_overall[1])],
            "t_stat": float(overall_t),
            "p_val": float(overall_p),
        },
    }

    report_json_path = os.path.join(output_dir, "stage_n1v_cluster_blocked_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage N1-V Report to {report_json_path}")

    env.close()
    return out_json


def generate_stage_n1v_figures(
    df_cluster_audit: pd.DataFrame,
    df_blocked: pd.DataFrame,
    pair_summary: List[Dict],
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Cluster Audit: Mean Risk vs Evaluation Occupancy
    axes[0].scatter(
        df_cluster_audit["eval_occupancy_wc"] * 100,
        df_cluster_audit["mean_risk_rc"],
        s=80,
        color="#1f77b4",
        edgecolors="black",
        alpha=0.85,
    )
    for _, row in df_cluster_audit.iterrows():
        axes[0].annotate(f"C{int(row['cluster_id'])}", (row["eval_occupancy_wc"] * 100 + 0.3, row["mean_risk_rc"]))

    axes[0].set_xlabel("Evaluation Occupancy $w_c$ (%)")
    axes[0].set_ylabel("Mean Decision Risk $\\bar{r}_c$ (Action-Gradient)")
    axes[0].set_title("(a) Cluster Risk vs Evaluation Occupancy Audit")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Cluster-Blocked Comparison across 5 Pairs
    pair_names = [p["pair_name"] for p in pair_summary]
    hr_utils = [p["mean_hr_utility"] for p in pair_summary]
    lr_utils = [p["mean_lr_utility"] for p in pair_summary]

    x = np.arange(len(pair_names))
    width = 0.35

    axes[1].bar(x - width/2, hr_utils, width, label="HR/LC (High Risk)", color="#d62728", alpha=0.85, edgecolor="black")
    axes[1].bar(x + width/2, lr_utils, width, label="LR/LC (Low Risk)", color="#1f77b4", alpha=0.85, edgecolor="black")
    axes[1].axhline(0, color="black", linestyle="--", lw=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_names, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Realized Utility $U(S) = -\\Delta H=5$")
    axes[1].set_title("(b) Cluster-Blocked Risk Comparison (Matched Pairs)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_n1v_cluster_blocked_validation.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage N1-V Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_n1v_benchmark()
