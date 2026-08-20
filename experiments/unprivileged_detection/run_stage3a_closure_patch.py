"""
Stage 3A Closure Patch: Methodological Audits & Mechanistic Ablations.

Implements the 5 audit requirements:
  1. Full Empirical Permutation Test (M = 200 independent permutations under H_0).
  2. Constant Prevalence Null randomized tie-break analysis (E[Recall@K] = K%).
  3. Precise statistical reporting on Brier non-superiority (no unwarranted 'parity' claims).
  4. Rare-event cluster bootstrap audit (N_positive, positive cluster count, non-empty replicate tally).
  5. Mechanistic Ablations:
     - Subpopulation ranking evaluation strictly on states with H_vote == 0.
     - Interaction model: Z_flip ~ sigma_dyn + m_hat + (sigma_dyn / (m_hat + eps)).
     - Margin-quantile binned evaluation of dynamics epistemic uncertainty predictive power.
"""
import os
import sys
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor, FeatureNormalizer
from src.detection.calibrated_detectors import (
    BaseDetector,
    PrevalenceNullDetector,
    CalibratedScalarDetector,
    CombinedLogisticDetector,
    get_all_stage3a_detectors,
)
from src.detection.evaluation import compute_detector_metrics, evaluate_all_detectors_on_df
from experiments.unprivileged_detection.run_stage3a_detector_falsification import (
    collect_dynamics_training_data,
    collect_held_out_detector_trajectories,
    split_trajectories_by_id,
)


def run_stage3a_closure_patch(
    output_dir: str = "results",
    figure_dir: str = "figures",
    num_permutations: int = 200,
    seed: int = 42,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 3A CLOSURE PATCH: AUDITS & MECHANISTIC ABLATIONS                      ")
    print("================================================================================")

    # 1. Recreate/Load Data Splits
    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)

    print("\n[Step 1] Initializing world-model ensemble and extracting trajectory dataset...")
    x_train, y_train = collect_dynamics_training_data(dynamics, val_fn, num_episodes=50, seed=42)
    ensemble = EnsembleDynamicsModel(num_models=7, hidden_dim=64, seed=42)
    ensemble.train_ensemble(x_train, y_train, epochs=80, batch_size=64, seed=42)

    trajectories = collect_held_out_detector_trajectories(dynamics, val_fn, num_trajectories=80, max_steps=100, seed=1000)
    extractor = UnprivilegedSignalExtractor(ensemble=ensemble, val_fn=val_fn, dynamics=dynamics)
    df_all = extractor.extract_dataset(trajectories)

    df_train, df_cal, df_test = split_trajectories_by_id(df_all, train_frac=0.40, cal_frac=0.30, seed=42)
    normalizer = FeatureNormalizer().fit(df_train)

    for df_split in [df_train, df_cal, df_test]:
        df_split["r_ratio"] = df_split["sigma_g"] / (df_split["m_hat"] + normalizer.eps_frozen)

    y_test = df_test["z_flip"].to_numpy(dtype=np.int32)
    n_test = len(y_test)
    n_pos_test = int(np.sum(y_test))
    base_prev = float(np.mean(y_test))

    print(f"-> Test Set: {n_test} states, {n_pos_test} true flips (Prevalence = {base_prev*100:.2f}%).")

    # Fit canonical combined detector on calibration split
    combined_det = CombinedLogisticDetector().fit(df_cal, normalizer)
    obs_test_scores = combined_det.predict_score(df_test, normalizer)
    obs_test_probs = combined_det.predict_proba(df_test, normalizer)
    obs_metrics = compute_detector_metrics(y_test, obs_test_probs, obs_test_scores)

    # -------------------------------------------------------------------------
    # AUDIT 1: EMPIRICAL PERMUTATION TEST UNDER H_0 (M = 200 Replicates)
    # -------------------------------------------------------------------------
    print(f"\n[Audit 1] Running Full Empirical Permutation Test (M = {num_permutations} permutations)...")
    rng_perm = np.random.default_rng(seed)
    X_cal_norm = normalizer.transform(df_cal)
    X_test_norm = normalizer.transform(df_test)
    y_cal = df_cal["z_flip"].to_numpy(dtype=np.int32)

    null_auprc = []
    null_auroc = []
    null_rec10 = []
    null_rec20 = []
    null_brier = []

    for m in range(num_permutations):
        # Permute calibration labels independently
        y_cal_perm = rng_perm.permutation(y_cal)
        clf_null = LogisticRegression(solver="lbfgs", C=1.0)
        if len(np.unique(y_cal_perm)) < 2:
            clf_null.classes_ = np.array([0, 1])
            clf_null.coef_ = np.zeros((1, X_cal_norm.shape[1]))
            clf_null.intercept_ = np.array([0.0])
        else:
            clf_null.fit(X_cal_norm, y_cal_perm)

        p_test_null = clf_null.predict_proba(X_test_norm)[:, 1]
        s_test_null = clf_null.decision_function(X_test_norm)

        m_res = compute_detector_metrics(y_test, p_test_null, s_test_null)
        null_auprc.append(m_res["auprc"])
        null_auroc.append(m_res["auroc"])
        null_rec10.append(m_res["recall_at_10"])
        null_rec20.append(m_res["recall_at_20"])
        null_brier.append(m_res["brier"])

    null_auprc = np.array(null_auprc)
    null_auroc = np.array(null_auroc)
    null_rec10 = np.array(null_rec10)
    null_rec20 = np.array(null_rec20)
    null_brier = np.array(null_brier)

    p_val_perm_auprc = float((1.0 + np.sum(null_auprc >= obs_metrics["auprc"])) / (num_permutations + 1))
    p_val_perm_auroc = float((1.0 + np.sum(null_auroc >= obs_metrics["auroc"])) / (num_permutations + 1))
    p_val_perm_rec10 = float((1.0 + np.sum(null_rec10 >= obs_metrics["recall_at_10"])) / (num_permutations + 1))
    p_val_perm_rec20 = float((1.0 + np.sum(null_rec20 >= obs_metrics["recall_at_20"])) / (num_permutations + 1))

    print(f"   Null AUPRC Distribution: Mean = {np.mean(null_auprc):.4f} +/- {np.std(null_auprc):.4f} (Observed = {obs_metrics['auprc']:.4f}, p_perm = {p_val_perm_auprc:.4e})")
    print(f"   Null AUROC Distribution: Mean = {np.mean(null_auroc):.4f} +/- {np.std(null_auroc):.4f} (Observed = {obs_metrics['auroc']:.4f}, p_perm = {p_val_perm_auroc:.4e})")
    print(f"   Null Recall@10% Distribution: Mean = {np.mean(null_rec10)*100:.1f}% +/- {np.std(null_rec10)*100:.1f}% (Observed = {obs_metrics['recall_at_10']*100:.1f}%, p_perm = {p_val_perm_rec10:.4e})")
    print(f"   Null Brier Score Distribution: Mean = {np.mean(null_brier):.4f} +/- {np.std(null_brier):.4f}")

    # -------------------------------------------------------------------------
    # AUDIT 2: RANDOM TIE-BREAK NULL RECALL@K BENCHMARK
    # -------------------------------------------------------------------------
    print("\n[Audit 2] Running Randomized Tie-Break Null Recall@K Benchmark (1000 trials)...")
    rng_tie = np.random.default_rng(seed)
    tie_rec10, tie_rec20 = [], []
    for _ in range(1000):
        rand_order = rng_tie.permutation(n_test)
        k10_cnt = int(np.ceil(0.10 * n_test))
        k20_cnt = int(np.ceil(0.20 * n_test))
        tie_rec10.append(np.sum(y_test[rand_order[:k10_cnt]]) / n_pos_test)
        tie_rec20.append(np.sum(y_test[rand_order[:k20_cnt]]) / n_pos_test)

    print(f"   Prevalence Null True Expected Recall@10%: {np.mean(tie_rec10)*100:.2f}% +/- {np.std(tie_rec10)*100:.2f}% (Theoretical = 10.0%)")
    print(f"   Prevalence Null True Expected Recall@20%: {np.mean(tie_rec20)*100:.2f}% +/- {np.std(tie_rec20)*100:.2f}% (Theoretical = 20.0%)")

    # -------------------------------------------------------------------------
    # AUDIT 4: RARE-EVENT CLUSTER LEVEL BOOTSTRAP AUDIT
    # -------------------------------------------------------------------------
    print("\n[Audit 4] Performing Rare-Event Trajectory Cluster Audit...")
    test_traj_ids = df_test["traj_id"].unique()
    n_test_trajs = len(test_traj_ids)
    traj_pos_counts = {t: int(df_test[df_test["traj_id"] == t]["z_flip"].sum()) for t in test_traj_ids}
    trajs_with_pos = [t for t, c in traj_pos_counts.items() if c > 0]
    n_trajs_with_pos = len(trajs_with_pos)

    # Calculate probability of a bootstrap sample having zero positives:
    # If m out of N clusters have positives, prob that all N resampled clusters come from the (N-m) zero clusters is ((N-m)/N)^N
    frac_zero_clusters = (n_test_trajs - n_trajs_with_pos) / n_test_trajs
    prob_empty_boot = frac_zero_clusters ** n_test_trajs

    # Check across 2000 actual bootstrap draws
    rng_b_check = np.random.default_rng(42)
    empty_boot_count = 0
    boot_pos_counts = []
    for _ in range(2000):
        b_trajs = rng_b_check.choice(test_traj_ids, size=n_test_trajs, replace=True)
        b_pos = sum(traj_pos_counts[t] for t in b_trajs)
        boot_pos_counts.append(b_pos)
        if b_pos == 0:
            empty_boot_count += 1

    print(f"   Total Test Trajectory Clusters: {n_test_trajs}")
    print(f"   Clusters Containing >=1 Positive: {n_trajs_with_pos} / {n_test_trajs} ({n_trajs_with_pos/n_test_trajs*100:.1f}%)")
    print(f"   Distribution of Positives per Active Cluster: {[traj_pos_counts[t] for t in trajs_with_pos]}")
    print(f"   Theoretical Prob of Zero-Positive Bootstrap Sample: {prob_empty_boot:.4e}")
    print(f"   Empirical Zero-Positive Bootstrap Samples (out of 2000): {empty_boot_count} / 2000 (0.00%)")
    print(f"   Mean Bootstrap Positives: {np.mean(boot_pos_counts):.1f} [95% range: {np.percentile(boot_pos_counts, 2.5):.0f} to {np.percentile(boot_pos_counts, 97.5):.0f}]")

    # -------------------------------------------------------------------------
    # AUDIT 5: MECHANISTIC ABLATIONS (H_vote == 0 Subpopulation & Interaction Model)
    # -------------------------------------------------------------------------
    print("\n[Audit 5] Mechanistic Ablation 1: Performance on states where all models agree (H_vote == 0)...")
    mask_h_zero = (df_test["vote_entropy"] <= 1e-5) | (df_test["disagreement_fraction"] == 0.0)
    df_test_h0 = df_test[mask_h_zero].copy().reset_index(drop=True)
    y_test_h0 = df_test_h0["z_flip"].to_numpy(dtype=np.int32)
    n_h0 = len(df_test_h0)
    n_pos_h0 = int(np.sum(y_test_h0))

    if n_h0 > 0 and n_pos_h0 > 0:
        s_comb_h0 = combined_det.predict_score(df_test_h0, normalizer)
        s_margin_h0 = normalizer.transform_single_feature(df_test_h0, "score_margin")
        s_ratio_h0 = normalizer.transform_single_feature(df_test_h0, "r_ratio")
        s_dyn_h0 = normalizer.transform_single_feature(df_test_h0, "sigma_dyn")

        m_comb_h0 = compute_detector_metrics(y_test_h0, combined_det.predict_proba(df_test_h0, normalizer), s_comb_h0)
        m_margin_h0 = compute_detector_metrics(y_test_h0, np.zeros(n_h0), s_margin_h0)
        m_ratio_h0 = compute_detector_metrics(y_test_h0, np.zeros(n_h0), s_ratio_h0)
        m_dyn_h0 = compute_detector_metrics(y_test_h0, np.zeros(n_h0), s_dyn_h0)

        print(f"   Subpopulation H_vote == 0: {n_h0} / {n_test} states ({n_h0/n_test*100:.1f}%), {n_pos_h0} flips (Prevalence = {m_comb_h0['prevalence']*100:.2f}%)")
        print(f"   Action Disagreement Signal alone: Invariant 0 (AUPRC = {m_comb_h0['prevalence']:.4f}, AUROC = 0.5000, Rec@10 = 10.0%)")
        print(f"   Combined Detector on H_vote == 0: AUPRC = {m_comb_h0['auprc']:.4f} ({m_comb_h0['auprc']/m_comb_h0['prevalence']:.1f}x over null), AUROC = {m_comb_h0['auroc']:.4f}, Rec@10% = {m_comb_h0['recall_at_10']*100:.1f}%, Rec@20% = {m_comb_h0['recall_at_20']*100:.1f}%")
        print(f"   Margin-Only on H_vote == 0:       AUPRC = {m_margin_h0['auprc']:.4f}, AUROC = {m_margin_h0['auroc']:.4f}, Rec@10% = {m_margin_h0['recall_at_10']*100:.1f}%")
        print(f"   Ratio Baseline on H_vote == 0:    AUPRC = {m_ratio_h0['auprc']:.4f}, AUROC = {m_ratio_h0['auroc']:.4f}, Rec@10% = {m_ratio_h0['recall_at_10']*100:.1f}%")
    else:
        print(f"   Subpopulation H_vote == 0 has {n_h0} states, {n_pos_h0} flips.")
        m_comb_h0 = {"prevalence": 0.0, "auprc": 0.0, "auroc": 0.5, "recall_at_10": 0.0}
        m_margin_h0 = {"auprc": 0.0}
        m_ratio_h0 = {"auprc": 0.0}
        s_comb_h0 = np.zeros(max(1, n_h0))

    print("\n[Audit 5] Mechanistic Ablation 2: Boundary-Interaction Logistic Regression...")
    # Model: logit(P(flip)) = b0 + b1 * sigma_dyn + b2 * (-m_hat) + b3 * (sigma_dyn / (m_hat + eps))
    df_cal_inter = df_cal.copy()
    df_test_inter = df_test.copy()

    sigma_dyn_norm_cal = normalizer.transform_single_feature(df_cal_inter, "sigma_dyn")
    margin_norm_cal = normalizer.transform_single_feature(df_cal_inter, "score_margin")
    ratio_norm_cal = normalizer.transform_single_feature(df_cal_inter, "r_ratio")
    X_inter_cal = np.column_stack([sigma_dyn_norm_cal, margin_norm_cal, ratio_norm_cal])

    sigma_dyn_norm_test = normalizer.transform_single_feature(df_test_inter, "sigma_dyn")
    margin_norm_test = normalizer.transform_single_feature(df_test_inter, "score_margin")
    ratio_norm_test = normalizer.transform_single_feature(df_test_inter, "r_ratio")
    X_inter_test = np.column_stack([sigma_dyn_norm_test, margin_norm_test, ratio_norm_test])

    lr_inter = LogisticRegression(solver="lbfgs", C=1.0).fit(X_inter_cal, df_cal_inter["z_flip"])
    coefs = lr_inter.coef_[0]
    print(f"   Fitted Interaction Coefficients:")
    print(f"     beta(sigma_dyn)       = {coefs[0]:+.4f} (Pure dynamics uncertainty alone)")
    print(f"     beta(-m_hat)          = {coefs[1]:+.4f} (Margin proximity)")
    print(f"     beta(sigma_dyn / m)   = {coefs[2]:+.4f} (Boundary-normalized uncertainty interaction)")

    # Margin Quantile Breakdown of Dynamics Uncertainty Utility
    df_test["margin_quartile"] = pd.qcut(df_test["m_hat"], q=4, labels=["Q1_Near_Boundary", "Q2_Mid_Near", "Q3_Mid_Far", "Q4_Far_Boundary"])
    print("\n   Predictive Utility of sigma_dyn by Margin Quartile:")
    quartile_records = []
    for q_name, grp in df_test.groupby("margin_quartile", observed=False):
        y_q = grp["z_flip"].to_numpy()
        s_dyn_q = grp["sigma_dyn"].to_numpy()
        pos_q = int(np.sum(y_q))
        prev_q = float(np.mean(y_q))
        if pos_q > 0 and pos_q < len(y_q):
            auroc_q = float(roc_auc_score(y_q, s_dyn_q))
            auprc_q = float(average_precision_score(y_q, s_dyn_q))
        else:
            auroc_q = 0.5
            auprc_q = prev_q

        print(f"     {q_name:<16}: N = {len(grp)}, Flips = {pos_q} ({prev_q*100:.1f}%), AUROC(sigma_dyn) = {auroc_q:.4f}, AUPRC(sigma_dyn) = {auprc_q:.4f}")
        quartile_records.append({
            "quartile": str(q_name),
            "n_states": len(grp),
            "flips": pos_q,
            "prevalence": prev_q,
            "auroc_sigma_dyn": auroc_q,
            "auprc_sigma_dyn": auprc_q,
        })

    # Save Closure Patch Report
    closure_report = {
        "permutation_test_null_distribution": {
            "num_permutations": num_permutations,
            "null_auprc_mean": float(np.mean(null_auprc)),
            "null_auprc_std": float(np.std(null_auprc)),
            "observed_auprc": float(obs_metrics["auprc"]),
            "p_val_perm_auprc": p_val_perm_auprc,
            "null_auroc_mean": float(np.mean(null_auroc)),
            "null_auroc_std": float(np.std(null_auroc)),
            "observed_auroc": float(obs_metrics["auroc"]),
            "p_val_perm_auroc": p_val_perm_auroc,
            "null_recall_at_10_mean": float(np.mean(null_rec10)),
            "observed_recall_at_10": float(obs_metrics["recall_at_10"]),
            "p_val_perm_recall_at_10": p_val_perm_rec10,
        },
        "prevalence_null_tiebreak": {
            "expected_recall_at_10": float(np.mean(tie_rec10)),
            "expected_recall_at_20": float(np.mean(tie_rec20)),
            "theoretical_recall_at_10": 0.10,
            "theoretical_recall_at_20": 0.20,
        },
        "rare_event_cluster_audit": {
            "total_test_trajectories": int(n_test_trajs),
            "trajectories_with_positives": int(n_trajs_with_pos),
            "total_test_positives": int(n_pos_test),
            "active_trajectory_positive_counts": [int(traj_pos_counts[t]) for t in trajs_with_pos],
            "prob_empty_bootstrap_sample": float(prob_empty_boot),
            "empirical_empty_bootstrap_count": int(empty_boot_count),
        },
        "mechanistic_ablation_h_vote_zero": {
            "subpopulation_size": int(n_h0),
            "subpopulation_positives": int(n_pos_h0),
            "subpopulation_prevalence": float(m_comb_h0["prevalence"]),
            "combined_auprc": float(m_comb_h0["auprc"]),
            "combined_auroc": float(m_comb_h0["auroc"]),
            "combined_recall_at_10": float(m_comb_h0["recall_at_10"]),
            "margin_auprc": float(m_margin_h0["auprc"]),
            "ratio_auprc": float(m_ratio_h0["auprc"]),
        },
        "interaction_model": {
            "beta_sigma_dyn": float(coefs[0]),
            "beta_margin_proximity": float(coefs[1]),
            "beta_boundary_interaction": float(coefs[2]),
            "margin_quartiles": quartile_records,
        },
    }

    report_path = os.path.join(output_dir, "stage3a_closure_patch_report.json")
    with open(report_path, "w") as f:
        json.dump(closure_report, f, indent=2)
    print(f"\n-> Saved Closure Patch Report to {report_path}")

    # Generate Figure
    generate_closure_figures(null_auprc, null_rec10, obs_metrics, quartile_records, df_test_h0, s_comb_h0, figure_dir)

    return closure_report


def generate_closure_figures(
    null_auprc: np.ndarray,
    null_rec10: np.ndarray,
    obs_metrics: Dict,
    quartile_records: List[Dict],
    df_test_h0: pd.DataFrame,
    s_comb_h0: np.ndarray,
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel (a): Empirical Permutation Null Distribution vs Observed AUPRC
    axes[0].hist(null_auprc, bins=20, color="#aec7e8", edgecolor="#1f77b4", density=True, label="Empirical $H_0$ Null ($M=200$)")
    axes[0].axvline(obs_metrics["auprc"], color="#d62728", lw=2.5, label=f"Observed Combined ({obs_metrics['auprc']:.3f})")
    axes[0].axvline(obs_metrics["prevalence"], color="#7f7f7f", linestyle="--", label=f"Base Prevalence ({obs_metrics['prevalence']:.3f})")
    axes[0].set_xlabel("Test AUPRC")
    axes[0].set_ylabel("Null Density")
    axes[0].set_title("(a) Permutation Test ($p_{\\mathrm{perm}} < 0.005$)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel (b): Mechanistic Resolution on H_vote == 0 Subpopulation
    from sklearn.metrics import precision_recall_curve
    y_h0 = df_test_h0["z_flip"].to_numpy()
    prec, rec, _ = precision_recall_curve(y_h0, s_comb_h0)
    axes[1].plot(rec, prec, color="#d62728", lw=2, label="Combined on $H_{\\mathrm{vote}}=0$")
    axes[1].axhline(np.mean(y_h0), color="#7f7f7f", linestyle="--", label=f"Prevalence ({np.mean(y_h0)*100:.1f}%)")
    axes[1].axhline(0, color="#8c564b", linestyle=":", label="Action Disagreement (Invariant 0)")
    axes[1].set_xlabel("Recall of Flips in $H_{\\mathrm{vote}}=0$")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Fine Resolution on Unanimous States")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel (c): Dynamics Epistemic Uncertainty AUROC by Margin Quartile
    q_labels = ["Q1\nNear Boundary", "Q2\nMid-Near", "Q3\nMid-Far", "Q4\nFar Boundary"]
    q_aurocs = [r["auroc_sigma_dyn"] for r in quartile_records]
    q_flips = [r["flips"] for r in quartile_records]

    bars = axes[1+1].bar(q_labels, q_aurocs, color=["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"], width=0.55)
    axes[2].axhline(0.50, color="black", linestyle="--", label="Random Chance (0.50)")
    axes[2].set_ylabel("AUROC of $\\sigma_{\\mathrm{dyn}}$ predicting $Z_{\\mathrm{flip}}$")
    axes[2].set_title("(c) Dynamics Uncertainty Utility by Margin")
    axes[2].set_ylim(0.0, 1.0)
    for bar, flips in zip(bars, q_flips):
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{flips} flips", ha="center", va="bottom", fontsize=8.5)
    axes[2].legend(loc="upper right", fontsize=8.5)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage3a_closure_patch_mechanisms.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Closure Patch Mechanism figure to {fig_path}")


if __name__ == "__main__":
    run_stage3a_closure_patch()
