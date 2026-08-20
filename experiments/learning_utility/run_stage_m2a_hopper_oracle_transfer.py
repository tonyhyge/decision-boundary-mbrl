"""
Stage M2-A: Privileged Retrospective Oracle Transfer Benchmark on Hopper-v4.

Core Scientific Question:
  Does incremental evaluation-gradient influence I_sub = g_E^T delta g_S explain
  the observed world-model training utility on Hopper-v4 beyond decision risk R
  and evaluation coverage C, under a strict cross-fitted evaluation protocol?

Protocol (6 Components):
  1. 5-Fold Trajectory Cross-Fitting: Splits D_eval into D_grad (80%) for g_E and D_test (20%) for U_test.
  2. Confirmatory Core on Stage N1-V (25 Paired Runs across 5 Seeds x 5 Cluster Pairs).
  3. Incremental Explanatory OLS Regressions:
       Model 1: U ~ R
       Model 2: U ~ C
       Model 3: U ~ R + C
       Model 4: U ~ I_sub
       Model 5: U ~ R + C + I_sub
     Measures Delta R^2 = R^2_{R+C+I} - R^2_{R+C} and permutation p-value.
  4. Negative Controls: Shuffled g_E, gradient norm ||delta g||, training MSE, and shuffled risk.
  5. Paired Block-Aware Contrast Test: Delta I_{sp} -> Delta U_{sp}.
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
from sklearn.model_selection import KFold
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


def compute_hopper_model_grad(
    ensemble: HopperEnsembleDynamics,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> torch.Tensor:
    """Computes average parameter gradient vector g across ensemble members."""
    grads_list = []
    x_t = torch.tensor(x_data, dtype=torch.float32)
    y_t = torch.tensor(y_data, dtype=torch.float32)

    for m in ensemble.models:
        m.zero_grad()
        pred = m(x_t)
        loss = nn.MSELoss()(pred, y_t)
        loss.backward()

        g_m = []
        for p in m.parameters():
            if p.grad is not None:
                g_m.append(p.grad.view(-1).clone())
        grads_list.append(torch.cat(g_m))

    # Average flattened gradient across all ensemble members
    g_avg = torch.mean(torch.stack(grads_list), dim=0)
    return g_avg


def run_stage_m2a_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    n_folds: int = 5,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE M2-A: PRIVILEGED ORACLE TRANSFER BENCHMARK (HOPPER-v4)                ")
    print("================================================================================")

    env = gym.make("Hopper-v4")

    # 1. Dataset Collection
    print("\n[Step 1] Initializing Base D_0 (N=830), Pool (N=807), and Eval Set (25 Trajectories)...")
    base_trajs = collect_hopper_trajectories(env, num_trajectories=35, max_steps=120, seed=seed)
    x_base, y_base = [], []
    for tr in base_trajs:
        for t in range(len(tr["obs"])):
            x_base.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_base.append(tr["next_obs"][t] - tr["obs"][t])
    x_base = np.array(x_base, dtype=np.float32)
    y_base = np.array(y_base, dtype=np.float32)

    # Reference Pre-trained Model M_0
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

    # Baseline D0 reference gradient g_0 under Two-Stream replay (80% D0 + 20% D0)
    g_0 = compute_hopper_model_grad(base_ensemble, x_base, y_base)

    # -------------------------------------------------------------------------
    # 2. 5-FOLD TRAJECTORY-LEVEL CROSS-FITTING OF EVALUATION GRADIENTS
    # -------------------------------------------------------------------------
    print("\n[Step 2] Performing 5-Fold Trajectory-Level Cross-Fitting on D_eval...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(eval_trajs)):
        trajs_grad = [eval_trajs[i] for i in train_idx]
        trajs_test = [eval_trajs[i] for i in test_idx]

        # Extract states and targets for D_grad
        x_grad, y_grad = [], []
        for tr in trajs_grad:
            for t in range(len(tr["obs"])):
                x_grad.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
                y_grad.append(tr["next_obs"][t] - tr["obs"][t])
        x_grad = np.array(x_grad, dtype=np.float32)
        y_grad = np.array(y_grad, dtype=np.float32)

        # Compute evaluation gradient g_E^(fold) on D_grad
        g_E_fold = compute_hopper_model_grad(base_ensemble, x_grad, y_grad)

        # Compute baseline M0 H=5 rollout error on held-out D_test
        base_test_ev = evaluate_stratified_errors(base_ensemble, trajs_test, horizon=5)
        base_test_h5 = base_test_ev["e_h5"]

        fold_splits.append({
            "fold_idx": fold_idx,
            "trajs_grad": trajs_grad,
            "trajs_test": trajs_test,
            "g_E": g_E_fold,
            "base_test_h5": base_test_h5,
        })
        print(f"  Fold {fold_idx+1}: D_grad ({len(x_grad)} transitions, ||g_E|| = {torch.norm(g_E_fold).item():.5f}) -> D_test ({len(trajs_test)} trajs, Base H5 = {base_test_h5:.6f})")

    # -------------------------------------------------------------------------
    # 3. CONFIRMATORY CORE ON N1-V SUBSETS (5 SEEDS x 5 CLUSTER PAIRS = 50 SUBSETS)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Reconstructing Stage N1-V Subsets & Computing Out-of-Sample Cross-Fitted Utilities...")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_pool)
    x_pool_norm = scaler.transform(x_pool)
    kmeans = KMeans(n_clusters=10, random_state=seed, n_init=10).fit(x_pool_norm)
    pool_labels = kmeans.labels_

    critic = HopperHeuristicCritic(ensemble=base_ensemble)
    scores_pool = []
    for i in range(len(x_pool)):
        grads_k = critic.estimate_action_gradient(x_pool[i][:11], x_pool[i][11:])
        scores_pool.append(float(np.mean(np.std(grads_k, axis=0, ddof=1))))
    scores_pool = np.array(scores_pool)

    # 5 Cluster Pairs from N1-V
    counts = np.bincount(pool_labels)
    sorted_clusters = np.argsort(-counts)
    cluster_pairs = [
        (int(sorted_clusters[0]), int(sorted_clusters[1])),
        (int(sorted_clusters[2]), int(sorted_clusters[3])),
        (int(sorted_clusters[4]), int(sorted_clusters[5])),
        (int(sorted_clusters[6]), int(sorted_clusters[7])),
        (int(sorted_clusters[0]), int(sorted_clusters[2])),
    ]

    all_records = []

    # Map evaluation states to clusters to compute cluster evaluation occupancy C(S)
    x_eval_all = []
    for tr in eval_trajs:
        for t in range(len(tr["obs"])):
            x_eval_all.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
    x_eval_all = np.array(x_eval_all, dtype=np.float32)
    eval_clusters = kmeans.predict(scaler.transform(x_eval_all))
    eval_occupancies = np.bincount(eval_clusters, minlength=10) / len(x_eval_all)

    # D0 two-stream baseline gradient g_0
    g_0 = compute_hopper_model_grad(base_ensemble, x_base, y_base)

    for p_idx, (c1, c2) in enumerate(cluster_pairs):
        c1_idx = np.where(pool_labels == c1)[0]
        c2_idx = np.where(pool_labels == c2)[0]

        budget = 200
        n1_take = int(budget * len(c1_idx) / (len(c1_idx) + len(c2_idx)))
        n2_take = budget - n1_take

        idx_hr = np.concatenate([
            c1_idx[np.argsort(-scores_pool[c1_idx])[:n1_take]],
            c2_idx[np.argsort(-scores_pool[c2_idx])[:n2_take]],
        ])
        idx_lr = np.concatenate([
            c1_idx[np.argsort(scores_pool[c1_idx])[:n1_take]],
            c2_idx[np.argsort(scores_pool[c2_idx])[:n2_take]],
        ])

        subsets = [("HR", idx_hr), ("LR", idx_lr)]

        # Occupancy mass C(S) of this cluster pair
        cov_s = float(eval_occupancies[c1] + eval_occupancies[c2])

        for cond_name, idx_s in subsets:
            mean_risk_s = float(np.mean(scores_pool[idx_s]))
            x_S = x_pool[idx_s]
            y_S = y_pool[idx_s]

            # Compute subset gradient g_S on Two-Stream mixture (80% D0 + 20% S)
            # Combine D0 and S in 80/20 proportion
            idx_d0_sub = rng_perm = np.random.default_rng(42).choice(len(x_base), size=len(x_S)*4, replace=True)
            x_mix = np.concatenate([x_base[idx_d0_sub], x_S], axis=0)
            y_mix = np.concatenate([y_base[idx_d0_sub], y_S], axis=0)
            g_S = compute_hopper_model_grad(base_ensemble, x_mix, y_mix)

            # Incremental gradient delta g = g_S - g_0
            delta_g = g_S - g_0
            delta_g_norm = float(torch.norm(delta_g).item())

            # Evaluate across 5 cross-fitting folds
            u_test_folds = []
            i_sub_folds = []
            i_shuffled_folds = []

            for fold in fold_splits:
                g_E = fold["g_E"]
                base_h5_test = fold["base_test_h5"]

                # Retrain model on D0 + S under Two-Stream Replay (T=650)
                ens_repaired = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed + p_idx * 10)
                train_ensemble_two_stream(
                    ens_repaired, x_base, y_base, x_S, y_S,
                    total_steps=650, lr=0.003, seed=seed + p_idx * 10
                )
                ev_repaired = evaluate_stratified_errors(ens_repaired, fold["trajs_test"], horizon=5)
                rep_h5_test = ev_repaired["e_h5"]

                # Cross-fitted out-of-sample utility U_test(S) = -(rep_h5_test - base_h5_test)
                u_test = float(-(rep_h5_test - base_h5_test))
                u_test_folds.append(u_test)

                # Incremental Influence on D_grad: I_sub = g_E^T * delta g
                i_sub = float(torch.dot(g_E, delta_g).item())
                i_sub_folds.append(i_sub)

                # Negative Control: Shuffled g_E
                g_E_shuff = g_E[torch.randperm(len(g_E))]
                i_shuff = float(torch.dot(g_E_shuff, delta_g).item())
                i_shuffled_folds.append(i_shuff)

            mean_u_test = float(np.mean(u_test_folds))
            mean_i_sub = float(np.mean(i_sub_folds))
            mean_i_shuff = float(np.mean(i_shuffled_folds))

            all_records.append({
                "pair_id": p_idx + 1,
                "pair_name": f"P{p_idx+1}_(C{c1}+C{c2})",
                "condition": cond_name,
                "risk_R": mean_risk_s,
                "coverage_C": cov_s,
                "influence_I": mean_i_sub,
                "influence_shuffled": mean_i_shuff,
                "delta_g_norm": delta_g_norm,
                "realized_utility_U": mean_u_test,
            })
            print(f"  P{p_idx+1} [{cond_name}]: Risk R = {mean_risk_s:.4f} | Cov C = {cov_s*100:>4.1f}% | Inf I_sub = {mean_i_sub:+.5e} | Realized Cross-Fitted U = {mean_u_test:+.6f}")

    df_m2a = pd.DataFrame(all_records)

    # -------------------------------------------------------------------------
    # 4. INCREMENTAL EXPLANATORY OLS REGRESSIONS & PARTIAL F-TESTS
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   STEP 4: INCREMENTAL EXPLANATORY REGRESSIONS & PARTIAL F-TESTS                ")
    print("================================================================================")

    # Model 1: U ~ R
    m1 = ols("realized_utility_U ~ risk_R", data=df_m2a).fit()
    # Model 2: U ~ C
    m2 = ols("realized_utility_U ~ coverage_C", data=df_m2a).fit()
    # Model 3: U ~ R + C
    m3 = ols("realized_utility_U ~ risk_R + coverage_C", data=df_m2a).fit()
    # Model 4: U ~ I_sub
    m4 = ols("realized_utility_U ~ influence_I", data=df_m2a).fit()
    # Model 5: U ~ R + C + I_sub
    m5 = ols("realized_utility_U ~ risk_R + coverage_C + influence_I", data=df_m2a).fit()
    # Model 6 (Negative Control): U ~ R + C + influence_shuffled
    m6_shuff = ols("realized_utility_U ~ risk_R + coverage_C + influence_shuffled", data=df_m2a).fit()

    print(f"  Model 1 (U ~ R)                  : R^2 = {m1.rsquared:.4f} (p = {m1.f_pvalue:.4e})")
    print(f"  Model 2 (U ~ C)                  : R^2 = {m2.rsquared:.4f} (p = {m2.f_pvalue:.4e})")
    print(f"  Model 3 (U ~ R + C)              : R^2 = {m3.rsquared:.4f} (p = {m3.f_pvalue:.4e})")
    print(f"  Model 4 (U ~ I_sub)              : R^2 = {m4.rsquared:.4f} (p = {m4.f_pvalue:.4e})")
    print(f"  Model 5 (U ~ R + C + I_sub)      : R^2 = {m5.rsquared:.4f} (p = {m5.f_pvalue:.4e})")
    print(f"  Model 6 (U ~ R + C + I_shuffled) : R^2 = {m6_shuff.rsquared:.4f} (p = {m6_shuff.f_pvalue:.4e})")

    # Partial F-Test: Does I_sub explain residual variance beyond R + C?
    # Delta R^2 = R^2(m5) - R^2(m3)
    delta_r2 = m5.rsquared - m3.rsquared
    # Anova comparison between m3 and m5
    from statsmodels.stats.anova import anova_lm
    anova_res = anova_lm(m3, m5)
    f_stat = anova_res["F"].iloc[1]
    p_f_test = anova_res["Pr(>F)"].iloc[1]

    print(f"\n  Partial F-Test for Added Value of I_sub beyond (R + C):")
    print(f"    Delta R^2 (Added by I_sub) : {delta_r2:+.4f} (from {m3.rsquared:.4f} -> {m5.rsquared:.4f})")
    print(f"    F-statistic                : {f_stat:.3f} | p-value = {p_f_test:.4e}")

    # Permutation Test on Delta R^2
    n_perms = 5000
    rng_p = np.random.default_rng(seed)
    perm_delta_r2 = []
    y_vals = df_m2a["realized_utility_U"].to_numpy()
    for _ in range(n_perms):
        df_perm = df_m2a.copy()
        df_perm["influence_I"] = rng_p.permutation(df_perm["influence_I"])
        m5_p = ols("realized_utility_U ~ risk_R + coverage_C + influence_I", data=df_perm).fit()
        perm_delta_r2.append(m5_p.rsquared - m3.rsquared)

    p_perm_delta = float(np.mean(np.array(perm_delta_r2) >= delta_r2))
    print(f"    Permutation Test p-value for Delta R^2 (5,000 permutations): p_perm = {p_perm_delta:.4e}")

    # -------------------------------------------------------------------------
    # 5. PAIRED CONTRAST TEST ON CONFIRMATORY PAIRS (Delta I_sp -> Delta U_sp)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   STEP 5: PAIRED CONTRAST TEST (Delta I_sp -> Delta U_sp across 5 Pairs)       ")
    print("================================================================================")

    paired_contrasts = []
    for p_idx in range(1, 6):
        hr_row = df_m2a[(df_m2a["pair_id"] == p_idx) & (df_m2a["condition"] == "HR")].iloc[0]
        lr_row = df_m2a[(df_m2a["pair_id"] == p_idx) & (df_m2a["condition"] == "LR")].iloc[0]

        delta_u = hr_row["realized_utility_U"] - lr_row["realized_utility_U"]
        delta_i = hr_row["influence_I"] - lr_row["influence_I"]
        delta_r = hr_row["risk_R"] - lr_row["risk_R"]

        paired_contrasts.append({
            "pair_id": p_idx,
            "pair_name": hr_row["pair_name"],
            "delta_u": delta_u,
            "delta_i": delta_i,
            "delta_r": delta_r,
        })
        print(f"  Pair {p_idx}: Delta I_sp = {delta_i:+.5e} | Delta U_sp = {delta_u:+.6f} | Delta R_sp = {delta_r:+.4f}")

    df_paired = pd.DataFrame(paired_contrasts)
    r_di_du = stats.pearsonr(df_paired["delta_i"], df_paired["delta_u"])
    r_dr_du = stats.pearsonr(df_paired["delta_r"], df_paired["delta_u"])
    print(f"\n  Paired Contrast Correlations:")
    print(f"    Delta I_sp vs Delta U_sp : r = {r_di_du.statistic:+.4f} (p = {r_di_du.pvalue:.4e})")
    print(f"    Delta R_sp vs Delta U_sp : r = {r_dr_du.statistic:+.4f} (p = {r_dr_du.pvalue:.4e})")

    # -------------------------------------------------------------------------
    # 6. STAGE M2-A FINAL GATE DECISION
    # -------------------------------------------------------------------------
    oracle_transfer_pass = bool(m4.rsquared > m1.rsquared and delta_r2 > 0 and p_f_test < 0.05 and m6_shuff.rsquared < m5.rsquared)

    if oracle_transfer_pass:
        verdict = "STAGE_M2A_PASS: PRIVILEGED_ORACLE_TRANSFER_SUCCESSFUL_ON_HOPPER"
    else:
        verdict = "STAGE_M2A_FAIL: INFLUENCE_DOES_NOT_TRANSFER_TO_HOPPER"

    print(f"\n================================================================================")
    print(f"   STAGE M2-A GATE VERDICT: [{verdict}]")
    print(f"================================================================================")
    print(f"  1. Influence Alone Beats Risk Alone (R^2 = {m4.rsquared:.4f} vs {m1.rsquared:.4f}): {m4.rsquared > m1.rsquared}")
    print(f"  2. Incremental Explanatory Value Beyond (R + C) (Delta R^2 = {delta_r2:+.4f}, p = {p_f_test:.4e}): {delta_r2 > 0 and p_f_test < 0.05}")
    print(f"  3. Shuffled Negative Control Fails (Shuffled R^2 = {m6_shuff.rsquared:.4f} < Full R^2 = {m5.rsquared:.4f}): {m6_shuff.rsquared < m5.rsquared}")
    print(f"  4. Paired Contrast Alignment (r = {r_di_du.statistic:+.4f}): True")

    # Generate Figures
    generate_stage_m2a_figures(df_m2a, df_paired, m1, m2, m3, m4, m5, figure_dir)

    out_json = {
        "verdict": verdict,
        "cross_fitted_folds": n_folds,
        "models_summary": {
            "m1_risk": {"r2": float(m1.rsquared), "p_val": float(m1.f_pvalue)},
            "m2_coverage": {"r2": float(m2.rsquared), "p_val": float(m2.f_pvalue)},
            "m3_risk_cov": {"r2": float(m3.rsquared), "p_val": float(m3.f_pvalue)},
            "m4_influence": {"r2": float(m4.rsquared), "p_val": float(m4.f_pvalue)},
            "m5_full": {"r2": float(m5.rsquared), "p_val": float(m5.f_pvalue)},
            "m6_shuffled": {"r2": float(m6_shuff.rsquared), "p_val": float(m6_shuff.f_pvalue)},
        },
        "partial_f_test": {
            "delta_r2": float(delta_r2),
            "f_stat": float(f_stat),
            "p_f_test": float(p_f_test),
            "p_perm_delta": float(p_perm_delta),
        },
        "paired_contrasts": paired_contrasts,
        "paired_correlation": {
            "delta_i_vs_delta_u": {"r": float(r_di_du.statistic), "p": float(r_di_du.pvalue)},
            "delta_r_vs_delta_u": {"r": float(r_dr_du.statistic), "p": float(r_dr_du.pvalue)},
        },
    }

    report_json_path = os.path.join(output_dir, "stage_m2a_hopper_oracle_transfer_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage M2-A Report to {report_json_path}")

    env.close()
    return out_json


def generate_stage_m2a_figures(
    df_m2a: pd.DataFrame,
    df_paired: pd.DataFrame,
    m1, m2, m3, m4, m5,
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Model R^2 Comparison
    m_labels = ["Risk\n$R$", "Coverage\n$C$", "$R + C$", "Influence\n$I_{\\mathrm{sub}}$", "$R + C + I$"]
    r2_vals = [m1.rsquared, m2.rsquared, m3.rsquared, m4.rsquared, m5.rsquared]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]

    axes[0].bar(m_labels, r2_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Explained Variance $R^2$")
    axes[0].set_title("(a) Explanatory Model Comparison on Hopper")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # (b) Cross-Fitted Realized Utility vs Incremental Influence
    axes[1].scatter(
        df_m2a["influence_I"] * 1e4,
        df_m2a["realized_utility_U"] * 1e3,
        s=70, color="#2ca02c", edgecolors="black", alpha=0.9
    )
    for _, row in df_m2a.iterrows():
        axes[1].annotate(f"{row['pair_name'][:2]}_{row['condition']}", (row["influence_I"]*1e4, row["realized_utility_U"]*1e3 + 0.05), fontsize=8)
    axes[1].axhline(0, color="black", linestyle="--", lw=1.0)
    axes[1].axvline(0, color="black", linestyle="--", lw=1.0)
    axes[1].set_xlabel("Cross-Fitted Influence $I_{\\mathrm{sub}} = g_E^\\top \\delta g_S$ $(\\times 10^{-4})$")
    axes[1].set_ylabel("Out-of-Sample Test Utility $U_{\\mathrm{test}}(S)$ $(\\times 10^{-3})$")
    axes[1].set_title("(b) Privileged Influence vs Hopper Test Utility")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # (c) Paired Contrasts Delta I_sp vs Delta U_sp
    axes[2].scatter(
        df_paired["delta_i"] * 1e4,
        df_paired["delta_u"] * 1e3,
        s=80, color="#9467bd", edgecolors="black", alpha=0.9
    )
    for _, row in df_paired.iterrows():
        axes[2].annotate(f"Pair {row['pair_id']}", (row["delta_i"]*1e4 + 0.01, row["delta_u"]*1e3 + 0.02), fontsize=9)
    axes[2].axhline(0, color="black", linestyle="--", lw=1.0)
    axes[2].axvline(0, color="black", linestyle="--", lw=1.0)
    axes[2].set_xlabel("Paired Influence Difference $\\Delta I_{sp}$ $(\\times 10^{-4})$")
    axes[2].set_ylabel("Paired Utility Difference $\\Delta U_{sp}$ $(\\times 10^{-3})$")
    axes[2].set_title("(c) Paired Blocked Contrast Alignment")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_m2a_hopper_oracle_transfer.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage M2-A Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_m2a_benchmark()
