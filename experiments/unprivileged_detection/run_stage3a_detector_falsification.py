"""
Stage 3A: Unprivileged Risk Detector Falsification Experiment.

Core Scientific Question:
  Can model-internal signals predict when world-model error alters the agent's decision?
  rho_hat(s) = P(Z_flip(s) = 1 | I_Mhat(s))

Experimental Protocol:
  1. Freeze world-model ensemble trained on trajectory data.
  2. Collect held-out trajectory rollouts across independent seeds.
  3. Strictly split by trajectory into Train (40%), Calibration (30%), Final Test (30%).
  4. Extract 5 unprivileged signal families (Margin, Dynamics Disagreement, Value Gap Disagreement,
     Action Disagreement, Ratio Baseline) and ground-truth flip labels (true-dynamics reference-action disagreement).
  5. Fit calibration models (Platt scaling, Logistic Regression) strictly on Calibration split.
  6. Identify strongest single-signal baseline on Calibration split and freeze as comparator.
  7. Final Test evaluation (opened ONCE) with trajectory-level clustered paired bootstrap (2000 replicates).
  8. Mandatory Permuted-Label Negative Control and Prevalence Null Baseline.
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

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
    PermutedLabelControlDetector,
    get_all_stage3a_detectors,
)
from src.detection.evaluation import (
    compute_detector_metrics,
    evaluate_all_detectors_on_df,
    run_clustered_paired_bootstrap,
)


def collect_dynamics_training_data(
    dynamics: CartPoleDynamics,
    val_fn: CompetentCartPoleValueFunction,
    num_episodes: int = 50,
    max_steps: int = 100,
    epsilon: float = 0.35,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect exploratory transitions to train the world-model ensemble."""
    rng = np.random.default_rng(seed)
    x_list, y_list = [], []

    for _ in range(num_episodes):
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


def collect_held_out_detector_trajectories(
    dynamics: CartPoleDynamics,
    val_fn: CompetentCartPoleValueFunction,
    num_trajectories: int = 80,
    max_steps: int = 100,
    exploration_epsilon: float = 0.25,
    seed: int = 1000,
) -> List[List[Dict]]:
    """
    Collect held-out trajectory rollouts across independent seeds to form the detector dataset.
    These trajectories are NEVER used to fit the dynamics models.
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    for traj_idx in range(num_trajectories):
        traj = []
        # Perturbation across continuous state space
        s = rng.uniform(low=[-0.2, -0.2, -0.08, -0.10], high=[0.2, 0.2, 0.08, 0.10])
        for step in range(max_steps):
            if rng.uniform() < exploration_epsilon:
                a = int(rng.choice(2))
            else:
                a = val_fn.get_optimal_action(s)

            next_s, r, done = dynamics.step(s, a)
            traj.append({
                "state": s.copy(),
                "action": a,
                "reward": r,
                "next_state": next_s.copy(),
                "done": done,
            })
            s = next_s
            if done:
                break

        trajectories.append(traj)

    return trajectories


def split_trajectories_by_id(
    df: pd.DataFrame,
    train_frac: float = 0.40,
    cal_frac: float = 0.30,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict trajectory-level splitting into Train, Calibration, and Final Test splits.
    Guarantees no state leakage across trajectory clusters.
    """
    rng = np.random.default_rng(seed)
    unique_trajs = np.array(sorted(df["traj_id"].unique()))
    n_trajs = len(unique_trajs)

    perm = rng.permutation(unique_trajs)
    n_train = int(np.round(train_frac * n_trajs))
    n_cal = int(np.round(cal_frac * n_trajs))

    train_ids = set(perm[:n_train])
    cal_ids = set(perm[n_train : n_train + n_cal])
    test_ids = set(perm[n_train + n_cal :])

    df_train = df[df["traj_id"].isin(train_ids)].copy().reset_index(drop=True)
    df_cal = df[df["traj_id"].isin(cal_ids)].copy().reset_index(drop=True)
    df_test = df[df["traj_id"].isin(test_ids)].copy().reset_index(drop=True)

    return df_train, df_cal, df_test


def run_stage3a_detector_falsification(
    num_models: int = 7,
    num_trajectories: int = 80,
    train_seed: int = 42,
    test_seed: int = 1000,
    output_dir: str = "results",
    figure_dir: str = "figures",
) -> Dict:
    """
    Execute the full Stage 3A Unprivileged Detector Falsification Benchmark.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 3A: UNPRIVILEGED RISK DETECTOR FALSIFICATION EXPERIMENT                ")
    print("================================================================================")

    # 1. Initialize Environment and Value Reference
    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)

    # 2. Train World-Model Ensemble on Dynamics Training Data
    print("\n[Step 1] Training World-Model Ensemble (K = 7 Bootstrapped Gaussian/MLP Models)...")
    x_train, y_train = collect_dynamics_training_data(dynamics, val_fn, num_episodes=50, seed=train_seed)
    ensemble = EnsembleDynamicsModel(num_models=num_models, hidden_dim=64, seed=train_seed)
    train_losses = ensemble.train_ensemble(x_train, y_train, epochs=80, batch_size=64, seed=train_seed)
    print(f"-> Ensemble training complete. Final MSE losses: mean = {np.mean(train_losses):.6f}, std = {np.std(train_losses):.6f}")

    # 3. Collect Held-out Trajectories for Detector Falsification
    print(f"\n[Step 2] Collecting {num_trajectories} held-out trajectory rollouts...")
    trajectories = collect_held_out_detector_trajectories(
        dynamics, val_fn, num_trajectories=num_trajectories, max_steps=100, seed=test_seed
    )
    total_states = sum(len(t) for t in trajectories)
    print(f"-> Collected {len(trajectories)} trajectories with {total_states} total states.")

    # 4. Extract Internal Signals & Evaluation Labels
    print("\n[Step 3] Extracting unprivileged model-internal signals & true flip labels...")
    extractor = UnprivilegedSignalExtractor(ensemble=ensemble, val_fn=val_fn, dynamics=dynamics)
    df_all = extractor.extract_dataset(trajectories)
    print(f"-> Full dataset extracted: {len(df_all)} states. Base flip prevalence: {df_all['z_flip'].mean()*100:.2f}%")

    # 5. Strict Trajectory-Level Partitioning
    print("\n[Step 4] Trajectory-level split: Train (40%), Calibration (30%), Final Test (30%)...")
    df_train, df_cal, df_test = split_trajectories_by_id(df_all, train_frac=0.40, cal_frac=0.30, seed=42)
    print(f"   Train set:       {len(df_train)} states ({len(df_train['traj_id'].unique())} trajectories), Flip Rate = {df_train['z_flip'].mean()*100:.2f}%")
    print(f"   Calibration set: {len(df_cal)} states ({len(df_cal['traj_id'].unique())} trajectories), Flip Rate = {df_cal['z_flip'].mean()*100:.2f}%")
    print(f"   Final Test set:  {len(df_test)} states ({len(df_test['traj_id'].unique())} trajectories), Flip Rate = {df_test['z_flip'].mean()*100:.2f}%")

    # 6. Feature Normalization & Ratio Epsilon Freezing (Train split ONLY)
    print("\n[Step 5] Normalizing features & freezing epsilon strictly from training split...")
    normalizer = FeatureNormalizer().fit(df_train)
    print(f"-> Frozen epsilon for ratio signal: {normalizer.eps_frozen:.6f}")
    print(f"-> Training feature means: {normalizer.means}")
    print(f"-> Training feature stds:  {normalizer.stds}")

    # Recompute ratio signal with frozen epsilon across all splits
    for df_split in [df_train, df_cal, df_test]:
        df_split["r_ratio"] = df_split["sigma_g"] / (df_split["m_hat"] + normalizer.eps_frozen)

    # 7. Instantiate and Fit All Detectors
    print("\n[Step 6] Fitting calibrated detectors strictly on Calibration split...")
    detectors = get_all_stage3a_detectors()

    # Fit Prevalence Null on Train split
    detectors["prevalence_null"].fit_prevalence(df_train)

    # Fit all other detectors on Calibration split
    rng_fit = np.random.default_rng(42)
    for name, det in detectors.items():
        if name != "prevalence_null":
            det.fit(df_cal, normalizer, rng=rng_fit)

    # 8. Pre-Registered Validation Gate: Select Strongest Single-Signal Comparator on Calibration Split
    print("\n[Step 7] Evaluating Validation Gate on Calibration Split...")
    cal_metrics = evaluate_all_detectors_on_df(detectors, df_cal, normalizer)
    print("\n--- CALIBRATION SPLIT VALIDATION METRICS ---")
    cal_summary_rows = []
    for name, m in cal_metrics.items():
        print(f"   {name:<24}: Brier = {m['brier']:.4f}, ECE = {m['ece']:.4f}, AUPRC = {m['auprc']:.4f}, AUROC = {m['auroc']:.4f}, Rec@10 = {m['recall_at_10']*100:.1f}%, Rec@20 = {m['recall_at_20']*100:.1f}%")
        cal_summary_rows.append({"detector": name, **m})

    # Identify strongest single-signal baseline (excluding null, combined, permuted)
    single_signal_keys = ["margin_only", "dynamics_only", "gap_variance_only", "action_disagreement_only", "ratio_only"]
    strongest_brier_comp = min(single_signal_keys, key=lambda k: cal_metrics[k]["brier"])
    strongest_auprc_comp = max(single_signal_keys, key=lambda k: cal_metrics[k]["auprc"])

    # Primary comparator chosen by Brier on validation
    frozen_comparator = strongest_brier_comp
    print(f"\n-> PRE-REGISTERED FROZEN COMPARATOR: '{frozen_comparator}' (Best Val Brier = {cal_metrics[frozen_comparator]['brier']:.4f}, AUPRC = {cal_metrics[frozen_comparator]['auprc']:.4f})")

    # 9. Final Test Set Evaluation (Opened ONCE) & Clustered Bootstrap Inference
    print("\n[Step 8] Opening Final Test Split for Decisive Falsification Evaluation (2000 Clustered Bootstraps)...")
    test_metrics = evaluate_all_detectors_on_df(detectors, df_test, normalizer)

    boot_summary = run_clustered_paired_bootstrap(
        detectors=detectors,
        df_test=df_test,
        normalizer=normalizer,
        comparator_name=frozen_comparator,
        cluster_col="traj_id",
        n_boot=2000,
        seed=42,
    )

    print("\n================================================================================")
    print(f"   FINAL TEST RESULTS & PAIRED BOOTSTRAP VS '{frozen_comparator}' (N_boot=2000) ")
    print("================================================================================")

    test_table_rows = []
    for name, det_res in boot_summary.items():
        pm = det_res["point_metrics"]
        paired = det_res["paired_to_comparator"]
        d_brier = paired["delta_brier"]
        d_auprc = paired["delta_auprc"]
        d_rec10 = paired["delta_recall_at_10"]
        d_rec20 = paired["delta_recall_at_20"]

        print(f"\nDetector: {name}")
        print(f"  Point Metrics: Brier = {pm['brier']:.4f} (95% BCI: [{det_res['brier_95_bci'][0]:.4f}, {det_res['brier_95_bci'][1]:.4f}]), ECE = {pm['ece']:.4f}")
        print(f"                 AUPRC = {pm['auprc']:.4f} (95% BCI: [{det_res['auprc_95_bci'][0]:.4f}, {det_res['auprc_95_bci'][1]:.4f}]), AUROC = {pm['auroc']:.4f}")
        print(f"                 Recall@10% = {pm['recall_at_10']*100:.1f}%, Recall@20% = {pm['recall_at_20']*100:.1f}%, Precision@10% = {pm['precision_at_10']*100:.1f}%")
        print(f"  Paired Delta vs Comparator ({frozen_comparator}):")
        print(f"    Delta Brier:      {d_brier['observed']:+.5f} (95% BCI: [{d_brier['ci_95'][0]:+.5f}, {d_brier['ci_95'][1]:+.5f}], p = {d_brier['p_val']:.4e})")
        print(f"    Delta AUPRC:      {d_auprc['observed']:+.5f} (95% BCI: [{d_auprc['ci_95'][0]:+.5f}, {d_auprc['ci_95'][1]:+.5f}], p = {d_auprc['p_val']:.4e})")
        print(f"    Delta Recall@10%: {d_rec10['observed']*100:+.2f}% (95% BCI: [{d_rec10['ci_95'][0]*100:+.2f}%, {d_rec10['ci_95'][1]*100:+.2f}%], p = {d_rec10['p_val']:.4e})")
        print(f"    Delta Recall@20%: {d_rec20['observed']*100:+.2f}% (95% BCI: [{d_rec20['ci_95'][0]*100:+.2f}%, {d_rec20['ci_95'][1]*100:+.2f}%], p = {d_rec20['p_val']:.4e})")

        test_table_rows.append({
            "detector": name,
            "brier": pm["brier"],
            "brier_ci_lower": det_res["brier_95_bci"][0],
            "brier_ci_upper": det_res["brier_95_bci"][1],
            "ece": pm["ece"],
            "auprc": pm["auprc"],
            "auprc_ci_lower": det_res["auprc_95_bci"][0],
            "auprc_ci_upper": det_res["auprc_95_bci"][1],
            "auroc": pm["auroc"],
            "auroc_ci_lower": det_res["auroc_95_bci"][0],
            "auroc_ci_upper": det_res["auroc_95_bci"][1],
            "recall_at_10": pm["recall_at_10"],
            "recall_at_20": pm["recall_at_20"],
            "precision_at_10": pm["precision_at_10"],
            "precision_at_20": pm["precision_at_20"],
            "delta_brier_obs": d_brier["observed"],
            "delta_brier_ci_lower": d_brier["ci_95"][0],
            "delta_brier_ci_upper": d_brier["ci_95"][1],
            "delta_brier_pval": d_brier["p_val"],
            "delta_auprc_obs": d_auprc["observed"],
            "delta_auprc_ci_lower": d_auprc["ci_95"][0],
            "delta_auprc_ci_upper": d_auprc["ci_95"][1],
            "delta_auprc_pval": d_auprc["p_val"],
            "delta_rec10_obs": d_rec10["observed"],
            "delta_rec10_ci_lower": d_rec10["ci_95"][0],
            "delta_rec10_ci_upper": d_rec10["ci_95"][1],
            "delta_rec10_pval": d_rec10["p_val"],
        })

    # 10. Audit Gate & Hard Falsification Evaluation
    comb_paired = boot_summary["combined"]["paired_to_comparator"]
    perm_metrics = boot_summary["permuted_control"]["point_metrics"]
    null_metrics = boot_summary["prevalence_null"]["point_metrics"]

    # Falsification Checks:
    # 1. Negative Control Check: Permuted control must have AUPRC approx prevalence and Brier approx Null
    perm_auprc_diff = abs(perm_metrics["auprc"] - null_metrics["prevalence"])
    perm_control_passed = bool(perm_auprc_diff <= 0.05)

    # 2. Null Calibration Check: Combined detector must beat Prevalence Null in Brier score
    comb_vs_null_brier = boot_summary["combined"]["point_metrics"]["brier"] - null_metrics["brier"]
    null_brier_passed = bool(comb_vs_null_brier < 0)

    # 3. Strongest Single Comparator Check:
    comb_d_brier = comb_paired["delta_brier"]["observed"]
    comb_brier_ci_upper = comb_paired["delta_brier"]["ci_95"][1]
    comb_d_auprc = comb_paired["delta_auprc"]["observed"]
    comb_auprc_ci_lower = comb_paired["delta_auprc"]["ci_95"][0]
    comb_d_rec10 = comb_paired["delta_recall_at_10"]["observed"]

    # Scientific Verdict Determination:
    if comb_d_brier < 0 and comb_brier_ci_upper < 0 and (comb_d_auprc > 0 or comb_d_rec10 > 0):
        verdict = "PASS_STRONG_CALIBRATED_FUSION"
        verdict_desc = "Combined unprivileged detector achieves statistically significant improvements in calibrated Brier score and ranking over all single-signal baselines."
    elif comb_d_auprc > 0 and comb_auprc_ci_lower > 0:
        verdict = "PASS_UNPRIVILEGED_RISK_RANKER"
        verdict_desc = "Combined detector significantly improves ranking/AUPRC, but calibration Brier score does not surpass the strongest single-signal baseline. Claim restricted to unprivileged risk ranker."
    elif cal_metrics[frozen_comparator]["auprc"] > null_metrics["prevalence"] + 0.10:
        verdict = "PASS_SINGLE_SIGNAL_DOMINANCE"
        verdict_desc = f"Single signal '{frozen_comparator}' provides decisive out-of-sample risk discrimination, but multi-signal linear fusion does not add incremental value."
    else:
        verdict = "FAIL_NO_SIGNAL_DISCRIMINATION"
        verdict_desc = "No unprivileged internal signal provided out-of-sample decision flip discrimination. Stage 3A is falsified."

    print("\n================================================================================")
    print(f"   STAGE 3A SCIENTIFIC VERDICT: [{verdict}]")
    print(f"   {verdict_desc}")
    print("================================================================================")

    # 11. Save Results & Datasets
    test_df_path = os.path.join(output_dir, "stage3a_detector_test_dataset.csv")
    df_test.to_csv(test_df_path, index=False)

    report_data = {
        "verdict": verdict,
        "verdict_description": verdict_desc,
        "frozen_comparator": frozen_comparator,
        "negative_control_passed": perm_control_passed,
        "null_brier_passed": null_brier_passed,
        "dataset_statistics": {
            "num_trajectories_total": num_trajectories,
            "total_states": total_states,
            "train_states": len(df_train),
            "cal_states": len(df_cal),
            "test_states": len(df_test),
            "train_flip_prevalence": float(df_train["z_flip"].mean()),
            "cal_flip_prevalence": float(df_cal["z_flip"].mean()),
            "test_flip_prevalence": float(df_test["z_flip"].mean()),
            "frozen_ratio_eps": float(normalizer.eps_frozen),
        },
        "calibration_split_metrics": cal_summary_rows,
        "test_split_bootstrap_results": boot_summary,
        "test_table_rows": test_table_rows,
    }

    report_json_path = os.path.join(output_dir, "stage3a_detector_falsification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n-> Saved JSON Report to {report_json_path}")
    print(f"-> Saved Test Dataset to {test_df_path}")

    # 12. Generate Publication Figures
    generate_stage3a_figures(
        df_test=df_test,
        detectors=detectors,
        normalizer=normalizer,
        boot_summary=boot_summary,
        frozen_comparator=frozen_comparator,
        figure_dir=figure_dir,
    )

    return report_data


def generate_stage3a_figures(
    df_test: pd.DataFrame,
    detectors: Dict[str, BaseDetector],
    normalizer: FeatureNormalizer,
    boot_summary: Dict,
    frozen_comparator: str,
    figure_dir: str,
) -> None:
    """Generate high-resolution publication figures for Stage 3A."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    y_true = df_test["z_flip"].to_numpy(dtype=np.int32)
    color_map = {
        "combined": "#d62728",
        "ratio_only": "#1f77b4",
        "margin_only": "#2ca02c",
        "gap_variance_only": "#9467bd",
        "dynamics_only": "#ff7f0e",
        "action_disagreement_only": "#8c564b",
        "prevalence_null": "#7f7f7f",
        "permuted_control": "#17becf",
    }

    # -------------------------------------------------------------------------
    # FIGURE 1: Precision-Recall Curves & ROC Curves (2 panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Precision-Recall Curve
    for name, det in detectors.items():
        if name == "prevalence_null":
            continue
        scores = det.predict_score(df_test, normalizer)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        auprc_val = boot_summary[name]["point_metrics"]["auprc"]
        lbl = f"{name} (AUPRC = {auprc_val:.3f})"
        axes[0].plot(rec, prec, label=lbl, color=color_map.get(name, "black"), lw=2 if "combined" in name else 1.5)

    base_prev = float(np.mean(y_true))
    axes[0].axhline(base_prev, color="#7f7f7f", linestyle="--", label=f"Prevalence Null ({base_prev*100:.1f}%)")
    axes[0].set_xlabel("Recall of Decision Flips $Z_{\\mathrm{flip}}$")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("(a) Precision-Recall Curves (Held-out Test)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) ROC Curve
    for name, det in detectors.items():
        if name == "prevalence_null":
            continue
        scores = det.predict_score(df_test, normalizer)
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc_val = boot_summary[name]["point_metrics"]["auroc"]
        lbl = f"{name} (AUROC = {auroc_val:.3f})"
        axes[1].plot(fpr, tpr, label=lbl, color=color_map.get(name, "black"), lw=2 if "combined" in name else 1.5)

    axes[1].plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", label="Random Chance (0.500)")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("(b) ROC Discrimination Curves")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pr_roc_path = os.path.join(figure_dir, "stage3a_pr_roc_curves.png")
    fig.savefig(pr_roc_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved PR/ROC curves figure to {pr_roc_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Reliability Diagram (Calibration Curves) & Risk-Coverage (2 panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Calibration Reliability Curve
    n_bins = 8
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for name in ["combined", frozen_comparator, "prevalence_null", "permuted_control"]:
        if name not in detectors:
            continue
        det = detectors[name]
        probs = det.predict_proba(df_test, normalizer)
        bin_confs, bin_accs = [], []
        for b in range(n_bins):
            in_b = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1] if b < n_bins - 1 else probs <= bin_edges[b + 1])
            if np.sum(in_b) >= 5:
                bin_confs.append(np.mean(probs[in_b]))
                bin_accs.append(np.mean(y_true[in_b]))

        brier_val = boot_summary[name]["point_metrics"]["brier"]
        axes[0].plot(bin_confs, bin_accs, marker="o", label=f"{name} (Brier = {brier_val:.4f})", color=color_map.get(name, "black"))

    axes[0].set_xlabel("Predicted Flip Probability $\\hat{\\rho}(s)$")
    axes[0].set_ylabel("Observed Empirical Flip Rate")
    axes[0].set_title("(a) Calibration Reliability Diagram")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Selective Acquisition Risk-Coverage Curve (Recall@K vs Budget Fraction K)
    budget_fractions = np.linspace(0.05, 0.50, 19)
    total_positives = int(np.sum(y_true))

    for name in ["combined", "ratio_only", "margin_only", "gap_variance_only", "permuted_control"]:
        if name not in detectors:
            continue
        det = detectors[name]
        scores = det.predict_score(df_test, normalizer)
        order = np.argsort(-scores)
        recalls = []
        for k_frac in budget_fractions:
            k_count = max(1, int(np.ceil(k_frac * len(df_test))))
            pos_k = int(np.sum(y_true[order[:k_count]]))
            recalls.append(pos_k / max(1, total_positives))

        axes[1].plot(budget_fractions * 100, np.array(recalls) * 100, marker="s", ms=4, label=name, color=color_map.get(name, "black"), lw=2 if name == "combined" else 1.5)

    # Diagonal uniform acquisition line
    axes[1].plot(budget_fractions * 100, budget_fractions * 100, "k--", label="Random Acquisition")
    axes[1].set_xlabel("Inspection / Acquisition Budget $K$ (% of States)")
    axes[1].set_ylabel("Cumulative Flip Capture (Recall@K %)")
    axes[1].set_title("(b) Selective Risk-Coverage Budget Curves")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    cal_cov_path = os.path.join(figure_dir, "stage3a_calibration_risk_coverage.png")
    fig.savefig(cal_cov_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Calibration and Risk-Coverage figure to {cal_cov_path}")


if __name__ == "__main__":
    run_stage3a_detector_falsification()
