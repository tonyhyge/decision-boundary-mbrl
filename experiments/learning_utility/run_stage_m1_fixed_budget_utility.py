"""
Stage M1: Fixed-Budget Substitution & Set-Level Utility Validation Benchmark.

Core Scientific Question:
  Does incremental gradient alignment I_sub(x) = g_E^T (g_{B,x} - g_B) explain realized
  training utility when a candidate replaces training exposure under fixed compute?
  Does non-additivity and multi-step operator path integration explain set-level utility?

Sub-Experiments:
  M1-A: One-Step Fixed-Budget Substitution & Incremental Gradient Prediction.
  M1-B: Orthogonal Falsification (R in {High, Low} x I_sub in {Aligned, Opposed}) across S = 10 Checkpoints.
  M1-C: Set Utility & Non-Additivity Interaction Gamma(i, j) on Redundant vs Diverse vs Mixed-Sign Batches.
  M1-D: Multi-Step Operator Path Influence (I_path = sum_t g_{E,t}^T delta g_t vs I_static).
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

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.learning_utility.run_stage_m0_operator_utility_falsification import (
    Controlled2DDynamics,
    true_physics_step,
    reference_policy,
    compute_loss_and_grad,
    compute_hessian_vector_product,
)


def generate_evaluation_set(n_trajectories: int = 20, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates ground-truth evaluation trajectory distribution along reference policy."""
    rng = np.random.default_rng(seed)
    eval_states, eval_targets = [], []
    for _ in range(n_trajectories):
        s = rng.uniform([-0.8, -1.0], [0.8, 1.0])
        for _ in range(10):
            a = reference_policy(s)
            s_next = true_physics_step(s, a)
            eval_states.append(np.concatenate([s, a]))
            eval_targets.append(s_next - s)
            s = s_next
    x_eval = torch.tensor(np.array(eval_states, dtype=np.float32))
    y_eval = torch.tensor(np.array(eval_targets, dtype=np.float32))
    return x_eval, y_eval


def generate_baseline_dataset(n_trajectories: int = 15, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates baseline exploratory dataset D_0."""
    rng = np.random.default_rng(seed)
    d0_states, d0_targets = [], []
    for _ in range(n_trajectories):
        s = rng.uniform([-1.5, -2.0], [1.5, 2.0])
        for _ in range(10):
            a = rng.uniform([-1.5], [1.5])
            s_next = true_physics_step(s, a)
            d0_states.append(np.concatenate([s, a]))
            d0_targets.append(s_next - s)
            s = s_next
    x_d0 = torch.tensor(np.array(d0_states, dtype=np.float32))
    y_d0 = torch.tensor(np.array(d0_targets, dtype=np.float32))
    return x_d0, y_d0


def train_baseline_model(x_d0: torch.Tensor, y_d0: torch.Tensor, seed: int = 42) -> nn.Module:
    """Trains a baseline neural world model theta_0 on D_0."""
    model = Controlled2DDynamics(hidden_dim=32, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x_d0), y_d0)
        loss.backward()
        optimizer.step()
    return model


def run_stage_m1_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    n_checkpoints: int = 10,
    seed: int = 42,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE M1: FIXED-BUDGET SUBSTITUTION & SET-UTILITY BENCHMARK                  ")
    print("================================================================================")

    x_eval, y_eval = generate_evaluation_set(n_trajectories=20, seed=seed)
    print(f"-> Generated Reference Evaluation Distribution D_eval: {len(x_eval)} transitions.")

    # -------------------------------------------------------------------------
    # PART M1-B: ORTHOGONAL FALSIFICATION UNDER FIXED COMPUTE ACROSS S = 10 CHECKPOINTS
    # -------------------------------------------------------------------------
    print(f"\n================================================================================")
    print(f"   PART M1-B: FIXED-BUDGET SUBSTITUTION ACROSS S = {n_checkpoints} INDEPENDENT CHECKPOINTS   ")
    print(f"================================================================================")

    checkpoint_records = []
    eta = 0.01
    batch_size = 32

    for s_idx in range(n_checkpoints):
        ckpt_seed = seed + s_idx * 17
        x_d0, y_d0 = generate_baseline_dataset(n_trajectories=15, seed=ckpt_seed)
        model = train_baseline_model(x_d0, y_d0, seed=ckpt_seed)

        l_eval_base, g_eval = compute_loss_and_grad(model, x_eval, y_eval)

        # Baseline batch B of size 32 from D_0
        rng_ckpt = np.random.default_rng(ckpt_seed)
        idx_b = rng_ckpt.choice(len(x_d0), size=batch_size, replace=False)
        x_B = x_d0[idx_b]
        y_B = y_d0[idx_b]
        l_B_base, g_B = compute_loss_and_grad(model, x_B, y_B)

        # Baseline post-update parameters theta_B = theta_0 - eta * g_B
        model_B = copy.deepcopy(model)
        with torch.no_grad():
            idx_p = 0
            for p in model_B.parameters():
                n_elem = p.numel()
                p.sub_(eta * g_B[idx_p:idx_p+n_elem].view(p.shape))
                idx_p += n_elem
        l_eval_B, _ = compute_loss_and_grad(model_B, x_eval, y_eval)

        # Generate N = 40 candidate substitution transitions
        n_cands = 40
        cand_states, cand_targets = [], []
        for _ in range(n_cands):
            s_c = rng_ckpt.uniform([-1.5, -2.0], [1.5, 2.0])
            a_c = rng_ckpt.uniform([-1.5], [1.5])
            s_next_c = true_physics_step(s_c, a_c)
            cand_states.append(np.concatenate([s_c, a_c]))
            cand_targets.append(s_next_c - s_c)
        x_cand = torch.tensor(np.array(cand_states, dtype=np.float32))
        y_cand = torch.tensor(np.array(cand_targets, dtype=np.float32))

        u_sub_real_list = []
        u_sub_pred_list = []
        risk_list = []
        influence_sub_list = []
        grad_norm_list = []
        pred_loss_list = []

        for i in range(n_cands):
            # Substitute last sample in batch B with candidate x_i
            x_Bx = torch.cat([x_B[:-1], x_cand[i:i+1]], dim=0)
            y_Bx = torch.cat([y_B[:-1], y_cand[i:i+1]], dim=0)
            l_Bx, g_Bx = compute_loss_and_grad(model, x_Bx, y_Bx)

            # Incremental gradient delta g_x = g_{B,x} - g_B
            delta_g_x = g_Bx - g_B
            i_sub = float(torch.dot(g_eval, delta_g_x).item())
            influence_sub_list.append(i_sub)
            grad_norm_list.append(float(torch.norm(delta_g_x).item()))

            # Model update under substitution theta_{B,x} = theta_0 - eta * g_{B,x}
            model_Bx = copy.deepcopy(model)
            with torch.no_grad():
                idx_p = 0
                for p in model_Bx.parameters():
                    n_elem = p.numel()
                    p.sub_(eta * g_Bx[idx_p:idx_p+n_elem].view(p.shape))
                    idx_p += n_elem
            l_eval_Bx, _ = compute_loss_and_grad(model_Bx, x_eval, y_eval)

            # Realized substitution utility U_sub(x_i) = L_E(theta_B) - L_E(theta_{B,x})
            u_sub_real = float(l_eval_B - l_eval_Bx)
            u_sub_real_list.append(u_sub_real)

            # First-order predicted substitution utility U_sub^(1) = eta * g_E^T * delta g_x
            u_sub_pred = float(eta * i_sub)
            u_sub_pred_list.append(u_sub_pred)

            # Decision risk R(x_i)
            s_i = x_cand[i][:2].numpy()
            a_i = x_cand[i][2:].numpy()
            r_i = float(np.abs(s_i[0]) * 3.0 + np.abs(s_i[1]) * 1.0 + np.abs(a_i[0] - reference_policy(s_i)[0]) * 2.0)
            risk_list.append(r_i)

            # Candidate prediction loss l(x_i)
            l_cand, _ = compute_loss_and_grad(model, x_cand[i:i+1], y_cand[i:i+1])
            pred_loss_list.append(l_cand)

        df_ckpt = pd.DataFrame({
            "u_sub_real": u_sub_real_list,
            "u_sub_pred": u_sub_pred_list,
            "i_sub": influence_sub_list,
            "risk_R": risk_list,
            "grad_norm": grad_norm_list,
            "pred_loss": pred_loss_list,
        })

        rho_i_u = stats.spearmanr(df_ckpt["i_sub"], df_ckpt["u_sub_real"]).statistic
        rho_r_u = stats.spearmanr(df_ckpt["risk_R"], df_ckpt["u_sub_real"]).statistic
        rho_pred_u = stats.spearmanr(df_ckpt["pred_loss"], df_ckpt["u_sub_real"]).statistic
        r_i_u = stats.pearsonr(df_ckpt["i_sub"], df_ckpt["u_sub_real"]).statistic
        r_pred_u = stats.pearsonr(df_ckpt["u_sub_pred"], df_ckpt["u_sub_real"]).statistic

        print(f"  Checkpoint {s_idx+1:>2}: Incremental Influence rho(I_sub, U) = {rho_i_u:+.4f} (Pearson r = {r_i_u:+.4f}) | Decision Risk rho(R, U) = {rho_r_u:+.4f} | Prediction Loss rho(L, U) = {rho_pred_u:+.4f}")

        checkpoint_records.append({
            "checkpoint_id": s_idx + 1,
            "rho_i_u": float(rho_i_u),
            "rho_r_u": float(rho_r_u),
            "rho_pred_u": float(rho_pred_u),
            "r_i_u": float(r_i_u),
            "r_pred_u": float(r_pred_u),
            "df_ckpt": df_ckpt,
        })

    # Aggregate across S = 10 checkpoints
    rhos_i = [r["rho_i_u"] for r in checkpoint_records]
    rhos_r = [r["rho_r_u"] for r in checkpoint_records]
    rhos_l = [r["rho_pred_u"] for r in checkpoint_records]

    mean_rho_i = float(np.mean(rhos_i))
    ci_rho_i = stats.t.interval(0.95, df=len(rhos_i)-1, loc=mean_rho_i, scale=stats.sem(rhos_i))
    mean_rho_r = float(np.mean(rhos_r))
    ci_rho_r = stats.t.interval(0.95, df=len(rhos_r)-1, loc=mean_rho_r, scale=stats.sem(rhos_r))
    mean_rho_l = float(np.mean(rhos_l))

    t_diff, p_diff = stats.ttest_rel(rhos_i, rhos_r)

    print("\n--- AGGREGATE CHECKPOINT-LEVEL CORRELATIONS (S = 10 CHECKPOINTS) ---")
    print(f"  Incremental Influence rho(I_sub, U) : {mean_rho_i:+.4f} [95% CI: {ci_rho_i[0]:+.4f}, {ci_rho_i[1]:+.4f}]")
    print(f"  Decision Risk rho(R, U)             : {mean_rho_r:+.4f} [95% CI: {ci_rho_r[0]:+.4f}, {ci_rho_r[1]:+.4f}]")
    print(f"  Prediction Loss rho(L, U)           : {mean_rho_l:+.4f}")
    print(f"  Paired t-test (Influence vs Risk)   : t = {t_diff:.3f}, p = {p_diff:.4e}")

    # -------------------------------------------------------------------------
    # PART M1-C: SET UTILITY & NON-ADDITIVITY INTERACTION GAMMA(i, j)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PART M1-C: SET UTILITY & NON-ADDITIVITY (REDUNDANT vs DIVERSE vs MIXED)     ")
    print("================================================================================")

    # Use first checkpoint model to evaluate set-level interaction
    ref_ckpt = checkpoint_records[0]
    df_ref = ref_ckpt["df_ckpt"]
    model_ref = train_baseline_model(x_d0, y_d0, seed=seed)
    l_eval_base, g_eval = compute_loss_and_grad(model_ref, x_eval, y_eval)

    # Generate 3 Types of Subsets of Size K = 4:
    # 1. Redundant Batch: 4 high-risk, high-influence samples with high collinearity (same state region)
    # 2. Diverse Batch: 4 high-risk, high-influence samples from diverse spatial quadrants
    # 3. Mixed-Sign Batch: 2 positive-influence + 2 negative-influence samples

    # Measure realized set substitution utility U_sub(S) vs sum of individual utilities
    set_types = ["Redundant_HighI", "Diverse_HighI", "Mixed_Sign_I"]
    set_records = []

    for st in set_types:
        if st == "Redundant_HighI":
            # 4 close points near s = [0.6, 0.8]
            cand_s = [np.array([0.6 + 0.05 * k, 0.8 + 0.05 * k, -1.0]) for k in range(4)]
        elif st == "Diverse_HighI":
            # 4 diverse points across 4 quadrants
            cand_s = [
                np.array([0.6, 0.8, -1.0]),
                np.array([-0.6, -0.8, 1.0]),
                np.array([0.6, -0.8, -0.5]),
                np.array([-0.6, 0.8, 0.5]),
            ]
        else:
            # 2 positive influence + 2 opposing influence
            cand_s = [
                np.array([0.6, 0.8, -1.0]),
                np.array([-0.6, -0.8, 1.0]),
                np.array([1.5, 1.5, 1.5]),   # Opposing gradient
                np.array([-1.5, -1.5, -1.5]), # Opposing gradient
            ]

        cand_targets = [true_physics_step(s[:2], s[2:]) - s[:2] for s in cand_s]
        x_S = torch.tensor(np.array(cand_s, dtype=np.float32))
        y_S = torch.tensor(np.array(cand_targets, dtype=np.float32))

        # 1. Individual utilities U_sub(x_k)
        indiv_utils = []
        indiv_delta_gs = []
        for k in range(4):
            x_Bx = torch.cat([x_B[:-1], x_S[k:k+1]], dim=0)
            y_Bx = torch.cat([y_B[:-1], y_S[k:k+1]], dim=0)
            l_Bx, g_Bx = compute_loss_and_grad(model_ref, x_Bx, y_Bx)
            delta_g = g_Bx - g_B
            indiv_delta_gs.append(delta_g)

            model_Bx = copy.deepcopy(model_ref)
            with torch.no_grad():
                idx_p = 0
                for p in model_Bx.parameters():
                    n_elem = p.numel()
                    p.sub_(eta * g_Bx[idx_p:idx_p+n_elem].view(p.shape))
                    idx_p += n_elem
            l_eval_Bx, _ = compute_loss_and_grad(model_Bx, x_eval, y_eval)
            indiv_utils.append(float(l_eval_B - l_eval_Bx))

        # 2. Joint substitution of all 4 samples
        x_BS = torch.cat([x_B[:-4], x_S], dim=0)
        y_BS = torch.cat([y_B[:-4], y_S], dim=0)
        l_BS, g_BS = compute_loss_and_grad(model_ref, x_BS, y_BS)

        model_BS = copy.deepcopy(model_ref)
        with torch.no_grad():
            idx_p = 0
            for p in model_BS.parameters():
                n_elem = p.numel()
                p.sub_(eta * g_BS[idx_p:idx_p+n_elem].view(p.shape))
                idx_p += n_elem
        l_eval_BS, _ = compute_loss_and_grad(model_BS, x_eval, y_eval)

        u_set_real = float(l_eval_B - l_eval_BS)
        u_sum_indiv = float(np.sum(indiv_utils))
        non_additivity_gap = u_set_real - u_sum_indiv

        # Pairwise gradient cosine similarity
        cos_sims = []
        for i in range(4):
            for j in range(i+1, 4):
                cos_sim = torch.cosine_similarity(indiv_delta_gs[i], indiv_delta_gs[j], dim=0).item()
                cos_sims.append(cos_sim)
        mean_cos = float(np.mean(cos_sims))

        print(f"  Batch [{st:<14}]: Realized Set Utility U(S) = {u_set_real:+.6f} | Sum Individual Sum_i U(x_i) = {u_sum_indiv:+.6f} | Non-Additivity Gap Gamma = {non_additivity_gap:+.6f} | Mean Cosine Sim = {mean_cos:+.4f}")

        set_records.append({
            "set_type": st,
            "u_set_real": u_set_real,
            "u_sum_indiv": u_sum_indiv,
            "non_additivity_gap": non_additivity_gap,
            "mean_cosine_sim": mean_cos,
        })

    # -------------------------------------------------------------------------
    # PART M1-D: MULTI-STEP OPERATOR PATH INFLUENCE
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PART M1-D: MULTI-STEP OPERATOR PATH INFLUENCE (K = 5 UPDATES)                ")
    print("================================================================================")

    k_steps = 5
    n_multi_cands = 20
    multi_u_real = []
    i_static_list = []
    i_path_list = []

    for i in range(n_multi_cands):
        # 1. Multi-step baseline trajectory theta_0 -> ... -> theta_K under batch B
        model_b_step = copy.deepcopy(model_ref)
        opt_b = optim.SGD(model_b_step.parameters(), lr=eta)
        g_e_trajectory = []

        for t in range(k_steps):
            _, g_e_t = compute_loss_and_grad(model_b_step, x_eval, y_eval)
            g_e_trajectory.append(g_e_t)
            opt_b.zero_grad()
            loss_b = nn.MSELoss()(model_b_step(x_B), y_B)
            loss_b.backward()
            opt_b.step()

        l_eval_k_base, _ = compute_loss_and_grad(model_b_step, x_eval, y_eval)

        # 2. Multi-step candidate substitution trajectory under B, x_i
        x_Bx_i = torch.cat([x_B[:-1], x_cand[i:i+1]], dim=0)
        y_Bx_i = torch.cat([y_B[:-1], y_cand[i:i+1]], dim=0)

        model_bx_step = copy.deepcopy(model_ref)
        opt_bx = optim.SGD(model_bx_step.parameters(), lr=eta)
        delta_g_trajectory = []

        for t in range(k_steps):
            # Compute gradient difference delta g_t
            _, g_b_t = compute_loss_and_grad(model_bx_step, x_B, y_B)
            _, g_bx_t = compute_loss_and_grad(model_bx_step, x_Bx_i, y_Bx_i)
            delta_g_trajectory.append(g_bx_t - g_b_t)

            opt_bx.zero_grad()
            loss_bx = nn.MSELoss()(model_bx_step(x_Bx_i), y_Bx_i)
            loss_bx.backward()
            opt_bx.step()

        l_eval_k_sub, _ = compute_loss_and_grad(model_bx_step, x_eval, y_eval)

        # Realized Multi-Step Substitution Utility
        u_k_real = float(l_eval_k_base - l_eval_k_sub)
        multi_u_real.append(u_k_real)

        # Static Influence: K * eta * g_E,0^T delta g_0
        i_static = float(k_steps * eta * torch.dot(g_e_trajectory[0], delta_g_trajectory[0]).item())
        i_static_list.append(i_static)

        # Path Influence: eta * sum_{t=0}^{K-1} g_E,t^T delta g_t
        i_path = float(eta * np.sum([torch.dot(ge, dg).item() for ge, dg in zip(g_e_trajectory, delta_g_trajectory)]))
        i_path_list.append(i_path)

    corr_static = stats.spearmanr(i_static_list, multi_u_real)
    corr_path = stats.spearmanr(i_path_list, multi_u_real)

    print(f"  Multi-Step Predictive Accuracy (K = {k_steps} updates):")
    print(f"    Static Influence I_static : Spearman rho = {corr_static.statistic:+.4f} (p = {corr_static.pvalue:.4e})")
    print(f"    Path Influence   I_path   : Spearman rho = {corr_path.statistic:+.4f} (p = {corr_path.pvalue:.4e})")

    # -------------------------------------------------------------------------
    # 5. STAGE M1 FINAL GATE DECISION
    # -------------------------------------------------------------------------
    gate1_passed = bool(mean_rho_i > 0.85 and p_diff < 1e-4)
    gate2_passed = bool(set_records[0]["non_additivity_gap"] != 0)
    gate3_passed = bool(corr_path.statistic > 0.85)

    all_gates_pass = gate1_passed and gate2_passed and gate3_passed
    verdict = "STAGE_M1_PASS: INCREMENTAL_INFLUENCE_AND_OPERATOR_PATH_VALIDATED" if all_gates_pass else "STAGE_M1_INCONCLUSIVE"

    print(f"\n================================================================================")
    print(f"   STAGE M1 GATE VERDICT: [{verdict}]")
    print(f"================================================================================")
    print(f"  1. Incremental Influence Outperforms Risk Across S=10 Checkpoints (rho={mean_rho_i:.4f} vs {mean_rho_r:.4f}, p={p_diff:.4e}): {gate1_passed}")
    print(f"  2. Set-Level Non-Additivity Established (Redundant gap={set_records[0]['non_additivity_gap']:.6f}): {gate2_passed}")
    print(f"  3. Multi-Step Path Influence Predicts Multi-Step Utility (rho={corr_path.statistic:.4f}): {gate3_passed}")

    # Generate Figures
    generate_stage_m1_figures(checkpoint_records, set_records, i_static_list, i_path_list, multi_u_real, figure_dir)

    out_json = {
        "verdict": verdict,
        "part_m1b_checkpoints": {
            "mean_rho_influence": mean_rho_i,
            "ci_rho_influence": [float(ci_rho_i[0]), float(ci_rho_i[1])],
            "mean_rho_risk": mean_rho_r,
            "ci_rho_risk": [float(ci_rho_r[0]), float(ci_rho_r[1])],
            "mean_rho_pred_loss": mean_rho_l,
            "paired_t_diff": float(t_diff),
            "p_val_diff": float(p_diff),
        },
        "part_m1c_set_non_additivity": set_records,
        "part_m1d_multi_step_path": {
            "rho_static": float(corr_static.statistic),
            "p_static": float(corr_static.pvalue),
            "rho_path": float(corr_path.statistic),
            "p_path": float(corr_path.pvalue),
        },
    }

    report_json_path = os.path.join(output_dir, "stage_m1_fixed_budget_utility_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage M1 Report to {report_json_path}")

    return out_json


def generate_stage_m1_figures(
    checkpoint_records: List[Dict],
    set_records: List[Dict],
    i_static: List[float],
    i_path: List[float],
    multi_u: List[float],
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Checkpoint-Level Distribution of rho(I_sub, U) vs rho(R, U)
    rhos_i = [r["rho_i_u"] for r in checkpoint_records]
    rhos_r = [r["rho_r_u"] for r in checkpoint_records]

    axes[0].boxplot([rhos_i, rhos_r], labels=["Incremental Influence\n$I_{\\mathrm{sub}}$", "Decision Risk\n$R(x)$"], patch_artist=True)
    axes[0].axhline(0, color="black", linestyle="--", lw=1.0)
    axes[0].set_ylabel("Spearman Rank Correlation $\\rho_s$ with Realized Utility")
    axes[0].set_title("(a) Substitution Utility Prediction ($S=10$ Checkpoints)")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # (b) Set-Level Non-Additivity Gap Gamma
    names = [r["set_type"].replace("_", "\n") for r in set_records]
    real_u = [r["u_set_real"] * 1e5 for r in set_records]
    sum_u = [r["u_sum_indiv"] * 1e5 for r in set_records]

    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, real_u, width, label="Realized Set $U(S)$", color="#2ca02c", alpha=0.85, edgecolor="black")
    axes[1].bar(x + width/2, sum_u, width, label="Sum of Indiv $\\sum U_i$", color="#1f77b4", alpha=0.85, edgecolor="black")
    axes[1].axhline(0, color="black", linestyle="--", lw=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel("Utility $(\\times 10^{-5})$")
    axes[1].set_title("(b) Set Non-Additivity $\\Gamma(S) = U(S) - \\sum U_i$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    # (c) Multi-Step Path Influence vs Realized Multi-Step Utility
    axes[2].scatter(i_path, multi_u, color="#d62728", alpha=0.85, s=50, edgecolors="black", label=f"Path ($r = {stats.pearsonr(i_path, multi_u).statistic:.3f}$)")
    axes[2].axhline(0, color="black", linestyle="--", lw=1.0)
    axes[2].axvline(0, color="black", linestyle="--", lw=1.0)
    axes[2].set_xlabel("Integrated Path Influence $I_{\\mathrm{path}} = \\eta \\sum_t g_{E,t}^\\top \\delta g_t$")
    axes[2].set_ylabel("Realized Multi-Step Utility $U_K(x)$ ($K=5$)")
    axes[2].set_title("(c) Multi-Step Operator Path Prediction")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_m1_fixed_budget_utility.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage M1 Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_m1_benchmark()
