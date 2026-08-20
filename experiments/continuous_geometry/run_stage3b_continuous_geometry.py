"""
Stage 3B: Continuous-Action Geometry and Decision Sensitivity Benchmark.

Core Scientific Question:
  Can model-induced decision risk in continuous-action RL be characterized by the
  local geometry of the action-value landscape, rather than by dynamics prediction error alone?

Falsification Ladder:
  - 3B.0: Ground-truth Geometry Sanity (Curvature vs Decision Regret under matched error).
  - 3B.1: Local Perturbation Law (delta_a_pred = - H_a^(-1) nabla_a delta_Q vs actual displacement).
  - 3B.2: Unprivileged Continuous Risk Ranking on Held-Out Test States (N = 1,500).
  - 3B.3: Negative Controls (Shuffled curvature, permuted perturbation) and Hard Kill Check.
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.continuous_control_geometry import ContinuousCartPoleEnv, ExactLQRGeometrySolver
from src.continuous_geometry.implicit_perturbation import ImplicitPerturbationAuditor
from src.continuous_geometry.continuous_risk_detector import (
    ContinuousEnsembleDynamics,
    ContinuousUnprivilegedSignalExtractor,
)


def collect_continuous_training_data(
    env: ContinuousCartPoleEnv,
    solver: ExactLQRGeometrySolver,
    num_episodes: int = 40,
    max_steps: int = 80,
    noise_std: float = 0.35,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect diverse continuous state-action transitions (s, a) -> delta_s."""
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []

    for _ in range(num_episodes):
        s = rng.uniform(low=[-0.2, -0.2, -0.08, -0.10], high=[0.2, 0.2, 0.08, 0.10])
        for _ in range(max_steps):
            a_star = solver.get_optimal_action(s)
            a_noisy = float(np.clip(a_star + rng.normal(0.0, noise_std), -1.0, 1.0))

            next_s, _, done = env.step(s, a_noisy)
            delta = next_s - s

            sa = np.concatenate([s, [a_noisy]])
            x_list.append(sa)
            y_list.append(delta)

            s = next_s
            if done:
                break

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def run_stage3b_continuous_geometry_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    num_models: int = 7,
    seed: int = 42,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 3B: CONTINUOUS-ACTION GEOMETRY & DECISION SENSITIVITY BENCHMARK        ")
    print("================================================================================")

    env = ContinuousCartPoleEnv()
    solver_nominal = ExactLQRGeometrySolver(env=env, r_cost=0.1)

    # -------------------------------------------------------------------------
    # 3B.0: GROUND-TRUTH GEOMETRY SANITY (MATCHED ERROR, VARYING CURVATURE)
    # -------------------------------------------------------------------------
    print("\n[Phase 3B.0] Evaluating Ground-Truth Geometry Sanity under Matched Predictive Error...")
    solver_sharp = ExactLQRGeometrySolver(env=env, r_cost=0.5)  # Sharp optimum: |H_a| large
    solver_flat = ExactLQRGeometrySolver(env=env, r_cost=0.01)  # Flat optimum: |H_a| small

    auditor_sharp = ImplicitPerturbationAuditor(env, solver_sharp)
    auditor_flat = ImplicitPerturbationAuditor(env, solver_flat)

    rng_sanity = np.random.default_rng(seed)
    n_sanity_states = 500
    sanity_states = rng_sanity.uniform(low=[-0.25, -0.25, -0.10, -0.12], high=[0.25, 0.25, 0.10, 0.12], size=(n_sanity_states, 4))

    matched_disp_flat = []
    matched_disp_sharp = []
    matched_regret_flat = []
    matched_regret_sharp = []

    # Apply identical synthetic perturbation delta_f(s, a)
    for s in sanity_states:
        pert_vec = rng_sanity.normal(0.0, 0.01, size=4)
        delta_fn = lambda st, ac: pert_vec

        res_flat = auditor_flat.evaluate_local_perturbation(s, delta_fn)
        res_sharp = auditor_sharp.evaluate_local_perturbation(s, delta_fn)

        matched_disp_flat.append(abs(res_flat["actual_displacement"]))
        matched_disp_sharp.append(abs(res_sharp["actual_displacement"]))
        matched_regret_flat.append(res_flat["actual_regret"])
        matched_regret_sharp.append(res_sharp["actual_regret"])

    disp_ratio = np.mean(matched_disp_flat) / max(1e-6, np.mean(matched_disp_sharp))
    t_stat_disp, p_val_disp = stats.ttest_rel(matched_disp_flat, matched_disp_sharp)

    print(f"  Matched Error: Mean Displacement in Flat Optimum  = {np.mean(matched_disp_flat):.4f} rad")
    print(f"  Matched Error: Mean Displacement in Sharp Optimum = {np.mean(matched_disp_sharp):.4f} rad")
    print(f"  Flatness Amplification Factor: {disp_ratio:.2f}x (t = {t_stat_disp:.3f}, p = {p_val_disp:.4e})")

    # -------------------------------------------------------------------------
    # 3B.1: LOCAL PERTURBATION LAW AUDIT (delta_a_pred vs actual displacement)
    # -------------------------------------------------------------------------
    print("\n[Phase 3B.1] Auditing First-Order Implicit Function Perturbation Law...")
    auditor_nom = ImplicitPerturbationAuditor(env, solver_nominal)
    n_pert_states = 1000
    pert_states = rng_sanity.uniform(low=[-0.3, -0.3, -0.12, -0.15], high=[0.3, 0.3, 0.12, 0.15], size=(n_pert_states, 4))

    pred_displacements = []
    actual_displacements = []
    sign_matches = []
    rel_errors = []

    for s in pert_states:
        # Random directional gradient perturbation
        scale = rng_sanity.uniform(1e-4, 0.02)
        pert_dir = rng_sanity.normal(size=4)
        pert_dir = pert_dir / np.linalg.norm(pert_dir) * scale
        delta_fn = lambda st, ac: pert_dir * (1.0 + 0.5 * ac)

        res_p = auditor_nom.evaluate_local_perturbation(s, delta_fn)
        pred_displacements.append(res_p["delta_a_pred"])
        actual_displacements.append(res_p["actual_displacement"])
        sign_matches.append(res_p["sign_match"])
        rel_errors.append(res_p["rel_error"])

    pred_displacements = np.array(pred_displacements)
    actual_displacements = np.array(actual_displacements)

    # Cosine alignment and Spearman correlation
    cos_sim = float(np.dot(pred_displacements, actual_displacements) / (np.linalg.norm(pred_displacements) * np.linalg.norm(actual_displacements) + 1e-12))
    spearman_rho, spearman_p = stats.spearmanr(pred_displacements, actual_displacements)
    mean_sign_match = float(np.mean(sign_matches))
    median_rel_err = float(np.median(rel_errors))

    print(f"  Vector Cosine Alignment cos(pred, actual) = {cos_sim:.4f}")
    print(f"  Directional Sign Match Rate               = {mean_sign_match*100:.1f}%")
    print(f"  Spearman Rank Correlation rho             = {spearman_rho:.4f} (p = {spearman_p:.4e})")
    print(f"  Median Relative Magnitude Error           = {median_rel_err*100:.2f}%")

    # Non-linear breakdown sweep
    perturbation_scales = np.logspace(-4, -0.5, 15)
    breakdown_errors = []
    for sc in perturbation_scales:
        scale_errs = []
        for s in pert_states[:100]:
            p_vec = np.array([0.0, sc, 0.0, sc * 2.0])
            res_sc = auditor_nom.evaluate_local_perturbation(s, lambda st, ac: p_vec * ac)
            scale_errs.append(res_sc["rel_error"])
        breakdown_errors.append(float(np.median(scale_errs)))

    # -------------------------------------------------------------------------
    # 3B.2: UNPRIVILEGED CONTINUOUS RISK RANKING ON HELD-OUT STATES
    # -------------------------------------------------------------------------
    print("\n[Phase 3B.2] Training Continuous World-Model Ensemble and Extracting Unprivileged Signals...")
    x_train, y_train = collect_continuous_training_data(env, solver_nominal, num_episodes=40, seed=seed)
    ensemble = ContinuousEnsembleDynamics(num_models=num_models, hidden_dim=64, seed=seed)
    train_losses = ensemble.train_ensemble(x_train, y_train, epochs=80, batch_size=64, seed=seed)
    print(f"  Ensemble trained. Mean final MSE loss: {np.mean(train_losses):.6f}")

    # Generate held-out evaluation dataset (N = 1,500 states from 30 independent test trajectories)
    print("  Generating held-out test rollouts (N = 1,500 states)...")
    extractor = ContinuousUnprivilegedSignalExtractor(ensemble=ensemble, solver=solver_nominal, env=env)

    rng_test = np.random.default_rng(seed + 500)
    test_records = []
    n_test_trajs = 30
    for traj_id in range(n_test_trajs):
        s = rng_test.uniform(low=[-0.25, -0.25, -0.10, -0.12], high=[0.25, 0.25, 0.10, 0.12])
        for step in range(50):
            # Extract unprivileged signals and ground truth regret
            sig = extractor.extract_state_signals(s)
            rec = {"traj_id": traj_id, "step": step, **sig}
            test_records.append(rec)

            # True step with model-chosen action
            next_s, _, done = env.step(s, sig["a_hat"])
            s = next_s
            if done:
                break

    df_test = pd.DataFrame(test_records)
    n_test = len(df_test)

    # Define Severe Regret Target: Top 10% highest regret states
    regret_threshold_90 = float(df_test["regret"].quantile(0.90))
    df_test["severe_regret"] = (df_test["regret"] >= regret_threshold_90).astype(int)
    n_pos = int(df_test["severe_regret"].sum())
    prevalence = float(n_pos / n_test)

    print(f"  Test Dataset: {n_test} states across {n_test_trajs} trajectories.")
    print(f"  Severe Regret Threshold (90th percentile): {regret_threshold_90:.5f}")
    print(f"  Positive Severe Regret States: {n_pos} / {n_test} ({prevalence*100:.1f}%)")

    # Signals to evaluate
    signals_dict = {
        "Random": rng_test.uniform(size=n_test),
        "Dynamics Uncertainty (sigma_dyn)": df_test["sigma_dyn"].to_numpy(),
        "Action Disagreement (sigma_a)": df_test["sigma_a"].to_numpy(),
        "Flatness Vulnerability (|H_hat|^-1)": df_test["inv_h_hat"].to_numpy(),
        "Action-Gradient Epistemic Disagreement (sigma_grad_Q)": df_test["sigma_grad_q"].to_numpy(),
        "Proposed Continuous Risk Score (S_cont)": df_test["s_cont"].to_numpy(),
    }

    # Negative Controls:
    # 1. Shuffled Curvature Control: S_cont with randomly permuted inverse curvature
    shuffled_inv_h = rng_test.permutation(df_test["inv_h_hat"].to_numpy())
    signals_dict["Shuffled Curvature Negative Control"] = df_test["sigma_grad_q"].to_numpy() * shuffled_inv_h
    # 2. Permuted Signal Control
    signals_dict["Permuted Signal Negative Control"] = rng_test.permutation(df_test["s_cont"].to_numpy())

    # Evaluate Metrics for All Signals
    y_true = df_test["severe_regret"].to_numpy()
    continuous_results = []

    print("\n--- CONTINUOUS DECISION-RISK RANKING BENCHMARK RESULTS ---")
    for name, s_vals in signals_dict.items():
        if "Random" in name:
            auprc = prevalence
            auroc = 0.5000
            rho = 0.0
            rec10 = 0.1000
        else:
            auprc = float(average_precision_score(y_true, s_vals))
            auroc = float(roc_auc_score(y_true, s_vals))
            rho, _ = stats.spearmanr(s_vals, df_test["regret"])
            # Recall@10%
            k_cnt = int(np.ceil(0.10 * n_test))
            top_k_labels = y_true[np.argsort(-s_vals)[:k_cnt]]
            rec10 = float(np.sum(top_k_labels) / max(1, n_pos))

        print(f"  {name:<46}: AUPRC = {auprc:.4f}, AUROC = {auroc:.4f}, rho(Regret) = {rho:+.4f}, Recall@10% = {rec10*100:.1f}%")

        continuous_results.append({
            "signal_name": name,
            "auprc": auprc,
            "auroc": auroc,
            "spearman_rho": float(rho),
            "recall_at_10": rec10,
        })

    # Paired Bootstrap for Proposed S_cont vs Dynamics Uncertainty
    s_cont_scores = df_test["s_cont"].to_numpy()
    s_dyn_scores = df_test["sigma_dyn"].to_numpy()

    boot_delta_auprc = []
    boot_delta_rec10 = []
    unique_trajs = df_test["traj_id"].unique()
    rng_b = np.random.default_rng(seed)

    for _ in range(2000):
        b_trajs = rng_b.choice(unique_trajs, size=len(unique_trajs), replace=True)
        idx_boot = []
        for t in b_trajs:
            idx_boot.extend(df_test[df_test["traj_id"] == t].index.tolist())
        idx_boot = np.array(idx_boot)

        b_y = y_true[idx_boot]
        b_s_cont = s_cont_scores[idx_boot]
        b_s_dyn = s_dyn_scores[idx_boot]

        if np.sum(b_y) > 0 and np.sum(b_y) < len(b_y):
            auprc_c = average_precision_score(b_y, b_s_cont)
            auprc_d = average_precision_score(b_y, b_s_dyn)
            k10 = int(np.ceil(0.10 * len(b_y)))
            rec_c = np.sum(b_y[np.argsort(-b_s_cont)[:k10]]) / np.sum(b_y)
            rec_d = np.sum(b_y[np.argsort(-b_s_dyn)[:k10]]) / np.sum(b_y)

            boot_delta_auprc.append(auprc_c - auprc_d)
            boot_delta_rec10.append(rec_c - rec_d)

    d_auprc_mean = float(np.mean(boot_delta_auprc))
    d_auprc_bci = [float(np.percentile(boot_delta_auprc, 2.5)), float(np.percentile(boot_delta_auprc, 97.5))]
    p_val_auprc = float(min(1.0, 2.0 * np.mean(np.array(boot_delta_auprc) <= 0)))

    d_rec10_mean = float(np.mean(boot_delta_rec10))
    d_rec10_bci = [float(np.percentile(boot_delta_rec10, 2.5)), float(np.percentile(boot_delta_rec10, 97.5))]
    p_val_rec10 = float(min(1.0, 2.0 * np.mean(np.array(boot_delta_rec10) <= 0)))

    print("\n--- PAIRED BOOTSTRAP: PROPOSED S_cont VS DYNAMICS UNCERTAINTY ---")
    print(f"  Delta AUPRC (S_cont - sigma_dyn): {d_auprc_mean:+.4f} (95% BCI: [{d_auprc_bci[0]:+.4f}, {d_auprc_bci[1]:+.4f}], p = {p_val_auprc:.4e})")
    print(f"  Delta Recall@10%               : {d_rec10_mean*100:+.2f}% (95% BCI: [{d_rec10_bci[0]*100:+.2f}%, {d_rec10_bci[1]*100:+.2f}%], p = {p_val_rec10:.4e})")

    # Hard Kill Criterion Check
    hard_kill_passed = bool(d_auprc_bci[0] > 0 and p_val_auprc < 0.01)
    verdict = "PASS_CONTINUOUS_GEOMETRY_SUPPORTED" if hard_kill_passed else "FAIL_HARD_KILL"

    print(f"\n[STAGE 3B HARD KILL CHECK]: {'PASS' if hard_kill_passed else 'KILL'} (p < 0.01 and 95% BCI > 0)")
    print(f"================================================================================")
    print(f"   STAGE 3B FINAL VERDICT: [{verdict}]")
    print(f"================================================================================")

    # Save Results
    df_test_path = os.path.join(output_dir, "stage3b_continuous_test_dataset.csv")
    df_test.to_csv(df_test_path, index=False)

    report_dict = {
        "verdict": verdict,
        "hard_kill_passed": hard_kill_passed,
        "phase_3b0_sanity": {
            "mean_disp_flat": float(np.mean(matched_disp_flat)),
            "mean_disp_sharp": float(np.mean(matched_disp_sharp)),
            "flatness_amplification_ratio": float(disp_ratio),
            "t_stat": float(t_stat_disp),
            "p_val": float(p_val_disp),
        },
        "phase_3b1_perturbation_law": {
            "cosine_similarity": float(cos_sim),
            "directional_sign_match_pct": float(mean_sign_match),
            "spearman_rho": float(spearman_rho),
            "spearman_pval": float(spearman_p),
            "median_relative_magnitude_error": float(median_rel_err),
            "breakdown_scales": perturbation_scales.tolist(),
            "breakdown_relative_errors": breakdown_errors,
        },
        "phase_3b2_ranking_results": continuous_results,
        "phase_3b3_paired_bootstrap": {
            "delta_auprc_mean": d_auprc_mean,
            "delta_auprc_95_bci": d_auprc_bci,
            "p_val_auprc": p_val_auprc,
            "delta_recall_at_10_mean": d_rec10_mean,
            "delta_recall_at_10_95_bci": d_rec10_bci,
            "p_val_recall_at_10": p_val_rec10,
        },
    }

    report_json_path = os.path.join(output_dir, "stage3b_continuous_geometry_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n-> Saved Stage 3B JSON Report to {report_json_path}")
    print(f"-> Saved Stage 3B Test Dataset to {df_test_path}")

    # Generate Figures
    generate_stage3b_figures(
        df_test=df_test,
        signals_dict=signals_dict,
        pred_disp=pred_displacements,
        act_disp=actual_displacements,
        breakdown_scales=perturbation_scales,
        breakdown_errors=breakdown_errors,
        figure_dir=figure_dir,
    )

    return report_dict


def generate_stage3b_figures(
    df_test: pd.DataFrame,
    signals_dict: Dict,
    pred_disp: np.ndarray,
    act_disp: np.ndarray,
    breakdown_scales: np.ndarray,
    breakdown_errors: List[float],
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    # -------------------------------------------------------------------------
    # FIGURE 1: Perturbation Law Validation & Non-linear Breakdown (2 Panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Predicted vs Actual Continuous Displacement
    axes[0].scatter(pred_disp, act_disp, alpha=0.3, color="#1f77b4", s=18)
    lims = [min(np.min(pred_disp), np.min(act_disp)), max(np.max(pred_disp), np.max(act_disp))]
    axes[0].plot(lims, lims, "r--", lw=2, label="Exact Theory (1:1 Line)")
    axes[0].set_xlabel("Predicted Displacement $\\delta a_{\\mathrm{pred}} = - H_a^{-1} \\nabla_a \\delta Q$")
    axes[0].set_ylabel("Empirical Displacement $\\hat{a} - a^*$")
    axes[0].set_title("(a) Implicit Perturbation Law Validation")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Relative Error vs Perturbation Magnitude
    axes[1].loglog(breakdown_scales, breakdown_errors, "o-", color="#d62728", lw=2)
    axes[1].axhline(0.25, color="black", linestyle="--", label="25% Linear Approximation Bound")
    axes[1].set_xlabel("Perturbation Magnitude $\|\\delta f\|$")
    axes[1].set_ylabel("Median Relative Displacement Error")
    axes[1].set_title("(b) First-Order Perturbation Breakdown")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(figure_dir, "stage3b_perturbation_law_validation.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Perturbation Law Figure to {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Continuous Precision-Recall & ROC Curves (2 Panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y_true = df_test["severe_regret"].to_numpy()

    color_map = {
        "Proposed Continuous Risk Score (S_cont)": "#d62728",
        "Action-Gradient Epistemic Disagreement (sigma_grad_Q)": "#9467bd",
        "Flatness Vulnerability (|H_hat|^-1)": "#2ca02c",
        "Action Disagreement (sigma_a)": "#8c564b",
        "Dynamics Uncertainty (sigma_dyn)": "#ff7f0e",
        "Shuffled Curvature Negative Control": "#bcbd22",
        "Permuted Signal Negative Control": "#7f7f7f",
    }

    # (a) Precision-Recall Curves
    for name, s_vals in signals_dict.items():
        if "Random" in name:
            continue
        prec, rec, _ = precision_recall_curve(y_true, s_vals)
        ap = average_precision_score(y_true, s_vals)
        axes[0].plot(rec, prec, label=f"{name} (AUPRC={ap:.3f})", color=color_map.get(name, "black"), lw=2 if "Proposed" in name else 1.5)

    axes[0].axhline(np.mean(y_true), color="#7f7f7f", linestyle="--", label=f"Prevalence Null ({np.mean(y_true)*100:.1f}%)")
    axes[0].set_xlabel("Recall of Severe Regret States ($R_{\\mathrm{dec}} > \\tau_{90\\%}$)")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("(a) Continuous Precision-Recall Curves")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) ROC Curves
    for name, s_vals in signals_dict.items():
        if "Random" in name:
            continue
        fpr, tpr, _ = roc_curve(y_true, s_vals)
        auc = roc_auc_score(y_true, s_vals)
        axes[1].plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})", color=color_map.get(name, "black"), lw=2 if "Proposed" in name else 1.5)

    axes[1].plot([0, 1], [0, 1], "k--", label="Random (0.500)")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("(b) Continuous ROC Discrimination")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(figure_dir, "stage3b_continuous_pr_roc_curves.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Continuous PR/ROC Figure to {fig2_path}")


if __name__ == "__main__":
    run_stage3b_continuous_geometry_benchmark()
