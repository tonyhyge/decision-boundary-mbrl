"""
Stage 4A: Rank-Based Active Acquisition Falsification Benchmark.

Core Scientific Question:
  Does decision-risk ranking improve model repair per acquired true transition?

Decoupled Falsification Ladder:
  - Phase 1 (Gate G4A.1): Oracle Repairability Gate at B = 10% (Is targeted model repair viable?).
  - Phase 2 (Gate G4A.2): Combined vs Random Gate at B = 10% (Does unprivileged ranking beat random sampling?).
  - Phase 3 (Gate G4A.3): Full 7-Strategy Comparator Ladder over Budget Sweep B in {1%, 5%, 10%, 20%}.
  - Phase 4 (Gate G4A.4): Downstream Policy Return and Boundary-Local Error Audit.
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor, FeatureNormalizer
from src.detection.calibrated_detectors import CombinedLogisticDetector
from src.acquisition.strategies import get_all_stage4a_strategies, BaseAcquisitionStrategy
from src.acquisition.repair_trainer import ModelRepairTrainer
from experiments.unprivileged_detection.run_stage3a_detector_falsification import (
    collect_dynamics_training_data,
    collect_held_out_detector_trajectories,
)


def run_stage4a_active_acquisition_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    num_models: int = 7,
    n_seeds_phase1: int = 10,
    n_seeds_sweep: int = 5,
    base_seed: int = 42,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE 4A: RANK-BASED ACTIVE ACQUISITION FALSIFICATION BENCHMARK              ")
    print("================================================================================")

    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)
    trainer = ModelRepairTrainer(dynamics=dynamics, val_fn=val_fn, epochs=50, lr=0.005)

    # 1. Collect Baseline Initial Dataset D_0
    print("\n[Step 1] Initializing baseline world-model training dataset D_0 (N_0 = 1,500)...")
    x_base, y_base = collect_dynamics_training_data(
        dynamics, val_fn, num_episodes=25, max_steps=60, epsilon=0.40, seed=base_seed
    )
    print(f"-> Base training transitions D_0: {len(x_base)} samples.")

    # 2. Train Initial World Model Ensemble M_0
    print("[Step 2] Training Initial World-Model Ensemble M_0...")
    base_ensemble = EnsembleDynamicsModel(num_models=num_models, hidden_dim=64, seed=base_seed)
    base_ensemble.train_ensemble(x_base, y_base, epochs=60, batch_size=64, seed=base_seed)

    # 3. Collect Candidate Pool D_pool and Held-Out Eval Set D_eval
    print("\n[Step 3] Collecting Candidate Pool D_pool (N=2,000) and Evaluation Set D_eval (N=1,500)...")
    pool_trajs = collect_held_out_detector_trajectories(
        dynamics, val_fn, num_trajectories=30, max_steps=70, exploration_epsilon=0.35, seed=base_seed + 100
    )
    eval_trajs = collect_held_out_detector_trajectories(
        dynamics, val_fn, num_trajectories=25, max_steps=60, exploration_epsilon=0.25, seed=base_seed + 200
    )

    extractor = UnprivilegedSignalExtractor(ensemble=base_ensemble, val_fn=val_fn, dynamics=dynamics)
    df_pool = extractor.extract_dataset(pool_trajs)
    df_eval = extractor.extract_dataset(eval_trajs)

    n_pool = len(df_pool)
    n_eval = len(df_eval)
    n_pos_pool = int(df_pool["z_flip"].sum())
    n_pos_eval = int(df_eval["z_flip"].sum())

    print(f"-> Candidate Pool D_pool: {n_pool} states ({n_pos_pool} flips, {n_pos_pool/n_pool*100:.2f}% prevalence).")
    print(f"-> Evaluation Set D_eval: {n_eval} states ({n_pos_eval} flips, {n_pos_eval/n_eval*100:.2f}% prevalence).")

    # Fit Normalizer & Unprivileged Combined Detector on D_pool
    normalizer = FeatureNormalizer().fit(df_pool)
    for df_split in [df_pool, df_eval]:
        df_split["r_ratio"] = df_split["sigma_g"] / (df_split["m_hat"] + normalizer.eps_frozen)

    detector = CombinedLogisticDetector().fit(df_pool, normalizer)

    # Baseline Evaluation before repair
    base_eval_metrics = trainer.evaluate_flips_on_eval_set(base_ensemble, df_eval)
    base_return_metrics = trainer.evaluate_downstream_policy_return(base_ensemble, num_episodes=50, seed=base_seed)
    n_flips_before = base_eval_metrics["n_flips"]
    flip_rate_before = base_eval_metrics["flip_rate"]

    print(f"\n-> Baseline Model M_0 Performance on D_eval:")
    print(f"   Flips Before Repair: {n_flips_before} / {n_eval} ({flip_rate_before*100:.2f}%)")
    print(f"   Mean Dynamics MSE:   {base_eval_metrics['mean_mse']:.6f} (Near Boundary: {base_eval_metrics['near_boundary_mse']:.6f})")
    print(f"   Mean Episode Return: {base_return_metrics['mean_return']:.2f} +/- {base_return_metrics['std_return']:.2f}")

    strategies = get_all_stage4a_strategies()

    # -------------------------------------------------------------------------
    # PHASE 1: GATE G4A.1 — ORACLE REPAIRABILITY GATE (B = 10%, N = 10 SEEDS)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PHASE 1: GATE G4A.1 — ORACLE REPAIRABILITY GATE (B = 10%, N = 10 Seeds)      ")
    print("================================================================================")

    b_10pct = max(1, int(np.round(0.10 * n_pool)))  # 10% budget
    oracle_flips_after = []
    oracle_etas = []
    random_flips_after_p1 = []
    random_etas_p1 = []

    for seed_idx in range(n_seeds_phase1):
        rng_seed = np.random.default_rng(base_seed + seed_idx * 13)

        # 1. Oracle Acquisition
        q_oracle = strategies["oracle"].select_queries(
            df_pool, budget=b_10pct, rng=rng_seed, detector=detector, normalizer=normalizer
        )
        # Construct acquired dataset (s, a, f_P(s, a))
        x_q_oracle = []
        y_q_oracle = []
        for idx in q_oracle:
            row = df_pool.iloc[idx]
            s = np.array([row["x"], row["x_dot"], row["theta"], row["theta_dot"]], dtype=np.float32)
            a = int(row["a_model_ens"])
            next_s, _, _ = dynamics.step(s, a)
            sa = np.concatenate([s, [1.0 if a == 1 else -1.0]])
            x_q_oracle.append(sa)
            y_q_oracle.append(next_s - s)

        x_aug_oracle = np.concatenate([x_base, np.array(x_q_oracle, dtype=np.float32)], axis=0)
        y_aug_oracle = np.concatenate([y_base, np.array(y_q_oracle, dtype=np.float32)], axis=0)

        ens_oracle = trainer.retrain_or_finetune_ensemble(base_ensemble, x_aug_oracle, y_aug_oracle, finetune=True, seed=base_seed + seed_idx)
        eval_oracle = trainer.evaluate_flips_on_eval_set(ens_oracle, df_eval)
        eta_oracle = (n_flips_before - eval_oracle["n_flips"]) / b_10pct
        oracle_flips_after.append(eval_oracle["n_flips"])
        oracle_etas.append(eta_oracle)

        # 2. Random Acquisition
        q_rand = strategies["random"].select_queries(
            df_pool, budget=b_10pct, rng=rng_seed, detector=detector, normalizer=normalizer
        )
        x_q_rand = []
        y_q_rand = []
        for idx in q_rand:
            row = df_pool.iloc[idx]
            s = np.array([row["x"], row["x_dot"], row["theta"], row["theta_dot"]], dtype=np.float32)
            a = int(row["a_model_ens"])
            next_s, _, _ = dynamics.step(s, a)
            sa = np.concatenate([s, [1.0 if a == 1 else -1.0]])
            x_q_rand.append(sa)
            y_q_rand.append(next_s - s)

        x_aug_rand = np.concatenate([x_base, np.array(x_q_rand, dtype=np.float32)], axis=0)
        y_aug_rand = np.concatenate([y_base, np.array(y_q_rand, dtype=np.float32)], axis=0)

        ens_rand = trainer.retrain_or_finetune_ensemble(base_ensemble, x_aug_rand, y_aug_rand, finetune=True, seed=base_seed + seed_idx)
        eval_rand = trainer.evaluate_flips_on_eval_set(ens_rand, df_eval)
        eta_rand = (n_flips_before - eval_rand["n_flips"]) / b_10pct
        random_flips_after_p1.append(eval_rand["n_flips"])
        random_etas_p1.append(eta_rand)

    mean_oracle_flips = float(np.mean(oracle_flips_after))
    mean_oracle_eta = float(np.mean(oracle_etas))
    mean_rand_flips_p1 = float(np.mean(random_flips_after_p1))
    mean_rand_eta_p1 = float(np.mean(random_etas_p1))

    delta_flips_oracle = mean_oracle_flips - n_flips_before
    t_stat_p1, p_val_p1 = stats.ttest_rel(oracle_etas, random_etas_p1)

    print(f"-> Oracle Flips After Repair: {mean_oracle_flips:.1f} +/- {np.std(oracle_flips_after):.1f} (Before: {n_flips_before}, Delta = {delta_flips_oracle:+.1f})")
    print(f"-> Oracle Repair Efficiency eta: {mean_oracle_eta:.4f} flips reduced per query")
    print(f"-> Random Flips After Repair: {mean_rand_flips_p1:.1f} +/- {np.std(random_flips_after_p1):.1f} (eta = {mean_rand_eta_p1:.4f})")
    print(f"-> Oracle vs Random Paired t-test: t({n_seeds_phase1-1}) = {t_stat_p1:.3f}, p = {p_val_p1:.4e}")

    if delta_flips_oracle >= 0:
        print("\n[CRITICAL HARD KILL] Gate G4A.1 FAILED: Oracle acquisition does NOT reduce decision flips.")
        print("Stopping Stage 4A immediately per pre-registered protocol (Repair Mechanism Failure).")
        return {"gate_g4a1_status": "FAIL", "reason": "Oracle repair did not reduce decision flips."}
    else:
        print("\n[GATE G4A.1 PASSED]: Oracle repairability confirmed. Model retraining actively eliminates decision flips.")

    # -------------------------------------------------------------------------
    # PHASE 2: GATE G4A.2 — COMBINED VS RANDOM GATE (B = 10%, N = 10 SEEDS)
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PHASE 2: GATE G4A.2 — COMBINED VS RANDOM GATE (B = 10%, N = 10 Seeds)        ")
    print("================================================================================")

    comb_flips_after_p2 = []
    comb_etas_p2 = []

    for seed_idx in range(n_seeds_phase1):
        rng_seed = np.random.default_rng(base_seed + seed_idx * 17)
        q_comb = strategies["combined"].select_queries(
            df_pool, budget=b_10pct, rng=rng_seed, detector=detector, normalizer=normalizer
        )
        x_q_comb = []
        y_q_comb = []
        for idx in q_comb:
            row = df_pool.iloc[idx]
            s = np.array([row["x"], row["x_dot"], row["theta"], row["theta_dot"]], dtype=np.float32)
            a = int(row["a_model_ens"])
            next_s, _, _ = dynamics.step(s, a)
            sa = np.concatenate([s, [1.0 if a == 1 else -1.0]])
            x_q_comb.append(sa)
            y_q_comb.append(next_s - s)

        x_aug_comb = np.concatenate([x_base, np.array(x_q_comb, dtype=np.float32)], axis=0)
        y_aug_comb = np.concatenate([y_base, np.array(y_q_comb, dtype=np.float32)], axis=0)

        ens_comb = trainer.retrain_or_finetune_ensemble(base_ensemble, x_aug_comb, y_aug_comb, finetune=True, seed=base_seed + seed_idx)
        eval_comb = trainer.evaluate_flips_on_eval_set(ens_comb, df_eval)
        eta_comb = (n_flips_before - eval_comb["n_flips"]) / b_10pct
        comb_flips_after_p2.append(eval_comb["n_flips"])
        comb_etas_p2.append(eta_comb)

    mean_comb_flips = float(np.mean(comb_flips_after_p2))
    mean_comb_eta = float(np.mean(comb_etas_p2))
    delta_flips_comb = mean_comb_flips - n_flips_before

    paired_diff_p2 = np.array(comb_etas_p2) - np.array(random_etas_p1)
    t_stat_p2, p_val_p2 = stats.ttest_rel(comb_etas_p2, random_etas_p1)
    bci_p2 = [float(np.percentile(paired_diff_p2, 2.5)), float(np.percentile(paired_diff_p2, 97.5))]

    print(f"-> Combined Ranker Flips After Repair: {mean_comb_flips:.1f} +/- {np.std(comb_flips_after_p2):.1f} (Before: {n_flips_before}, Delta = {delta_flips_comb:+.1f})")
    print(f"-> Combined Repair Efficiency eta: {mean_comb_eta:.4f} flips reduced per query")
    print(f"-> Random Repair Efficiency eta:   {mean_rand_eta_p1:.4f} flips reduced per query")
    print(f"-> Paired Efficiency Gain (Combined - Random): {np.mean(paired_diff_p2):+.4f} (95% CI: [{bci_p2[0]:+.4f}, {bci_p2[1]:+.4f}])")
    print(f"-> Paired t-test: t({n_seeds_phase1-1}) = {t_stat_p2:.3f}, p = {p_val_p2:.4e}")

    gate_g4a2_pass = bool(np.mean(paired_diff_p2) > 0 and p_val_p2 < 0.05)
    print(f"\n[GATE G4A.2 VERDICT]: {'PASS' if gate_g4a2_pass else 'NEGATIVE'} (Combined vs Random at B=10%)")

    # -------------------------------------------------------------------------
    # PHASE 3: GATE G4A.3 — FULL 7-STRATEGY COMPARATOR LADDER & BUDGET SWEEP
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("   PHASE 3: GATE G4A.3 — FULL COMPARATOR LADDER & BUDGET SWEEP (1%, 5%, 10%, 20%)")
    print("================================================================================")

    budget_pcts = [0.01, 0.05, 0.10, 0.20]
    sweep_records = []
    strategy_names = ["random", "dynamics", "margin", "action_disagreement", "ratio", "combined", "oracle", "shuffled_control"]

    for b_pct in budget_pcts:
        budget = max(1, int(np.round(b_pct * n_pool)))
        print(f"\n--- Running Sweep for Budget B = {b_pct*100:.0f}% ({budget} queries, N = {n_seeds_sweep} seeds) ---")

        for strat_key in strategy_names:
            strat = strategies[strat_key]
            flips_list = []
            etas_list = []
            near_mse_list = []
            far_mse_list = []
            return_list = []

            for s_idx in range(n_seeds_sweep):
                seed_val = base_seed + s_idx * 23 + int(b_pct * 1000)
                rng_s = np.random.default_rng(seed_val)

                q_indices = strat.select_queries(
                    df_pool, budget=budget, rng=rng_s, detector=detector, normalizer=normalizer
                )

                # Collect transitions
                x_q, y_q = [], []
                for idx in q_indices:
                    row = df_pool.iloc[idx]
                    s = np.array([row["x"], row["x_dot"], row["theta"], row["theta_dot"]], dtype=np.float32)
                    a = int(row["a_model_ens"])
                    next_s, _, _ = dynamics.step(s, a)
                    sa = np.concatenate([s, [1.0 if a == 1 else -1.0]])
                    x_q.append(sa)
                    y_q.append(next_s - s)

                x_aug = np.concatenate([x_base, np.array(x_q, dtype=np.float32)], axis=0)
                y_aug = np.concatenate([y_base, np.array(y_q, dtype=np.float32)], axis=0)

                ens_retrained = trainer.retrain_or_finetune_ensemble(base_ensemble, x_aug, y_aug, finetune=True, seed=seed_val)
                ev_res = trainer.evaluate_flips_on_eval_set(ens_retrained, df_eval)
                ret_res = trainer.evaluate_downstream_policy_return(ens_retrained, num_episodes=40, seed=seed_val)

                eta = (n_flips_before - ev_res["n_flips"]) / budget
                flips_list.append(ev_res["n_flips"])
                etas_list.append(eta)
                near_mse_list.append(ev_res["near_boundary_mse"])
                far_mse_list.append(ev_res["far_boundary_mse"])
                return_list.append(ret_res["mean_return"])

            mean_f = float(np.mean(flips_list))
            std_f = float(np.std(flips_list))
            mean_e = float(np.mean(etas_list))
            mean_near = float(np.mean(near_mse_list))
            mean_far = float(np.mean(far_mse_list))
            mean_ret = float(np.mean(return_list))
            delta_f_pct = float((mean_f - n_flips_before) / n_flips_before * 100)

            print(f"  {strat.name:<32}: Flips = {mean_f:.1f} +/- {std_f:.1f} ({delta_f_pct:+.1f}%), eta = {mean_e:+.4f}, Near MSE = {mean_near:.6f}, Return = {mean_ret:.1f}")

            sweep_records.append({
                "budget_pct": b_pct,
                "budget_count": budget,
                "strategy_key": strat_key,
                "strategy_name": strat.name,
                "is_privileged": strat.is_privileged,
                "flips_before": n_flips_before,
                "flips_after_mean": mean_f,
                "flips_after_std": std_f,
                "delta_flips_pct": delta_f_pct,
                "repair_efficiency_eta_mean": mean_e,
                "repair_efficiency_eta_std": float(np.std(etas_list)),
                "near_boundary_mse": mean_near,
                "far_boundary_mse": mean_far,
                "policy_return_mean": mean_ret,
                "policy_return_std": float(np.std(return_list)),
            })

    df_sweep = pd.DataFrame(sweep_records)
    csv_sweep_path = os.path.join(output_dir, "stage4a_acquisition_sweep.csv")
    df_sweep.to_csv(csv_sweep_path, index=False)
    print(f"\n-> Saved Stage 4A Sweep Results to {csv_sweep_path}")

    # Compile Final Report
    report_data = {
        "gate_g4a1_oracle_repairability": {
            "status": "PASS" if delta_flips_oracle < 0 else "FAIL",
            "budget_10pct": b_10pct,
            "flips_before": n_flips_before,
            "oracle_flips_after": mean_oracle_flips,
            "oracle_eta": mean_oracle_eta,
            "random_flips_after": mean_rand_flips_p1,
            "random_eta": mean_rand_eta_p1,
            "t_stat": float(t_stat_p1),
            "p_val": float(p_val_p1),
        },
        "gate_g4a2_combined_vs_random": {
            "status": "PASS" if gate_g4a2_pass else "FAIL",
            "combined_flips_after": mean_comb_flips,
            "combined_eta": mean_comb_eta,
            "random_eta": mean_rand_eta_p1,
            "paired_diff_mean": float(np.mean(paired_diff_p2)),
            "paired_diff_95_bci": bci_p2,
            "t_stat": float(t_stat_p2),
            "p_val": float(p_val_p2),
        },
        "sweep_summary": sweep_records,
    }

    report_json_path = os.path.join(output_dir, "stage4a_active_acquisition_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"-> Saved Stage 4A JSON Report to {report_json_path}")

    # Generate Figures
    generate_stage4a_figures(df_sweep, figure_dir)

    return report_data


def generate_stage4a_figures(df_sweep: pd.DataFrame, figure_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    color_map = {
        "combined": "#d62728",
        "oracle": "#2ca02c",
        "ratio": "#1f77b4",
        "margin": "#9467bd",
        "dynamics": "#ff7f0e",
        "action_disagreement": "#8c564b",
        "random": "#7f7f7f",
        "shuffled_control": "#bcbd22",
    }

    # -------------------------------------------------------------------------
    # FIGURE 1: Decision Flip Reduction & Repair Efficiency Across Budgets (2 Panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Flip Reduction % vs Budget Fraction
    for strat_key, grp in df_sweep.groupby("strategy_key"):
        s_name = grp["strategy_name"].iloc[0]
        axes[0].plot(
            grp["budget_pct"] * 100,
            grp["delta_flips_pct"],
            marker="o",
            label=s_name,
            color=color_map.get(strat_key, "black"),
            lw=2.5 if strat_key in ["combined", "oracle"] else 1.5,
        )

    axes[0].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Acquisition Budget $B$ (% of Pool)")
    axes[0].set_ylabel("Decision Flip Reduction $\\Delta \\mathrm{FlipRate}$ (%)")
    axes[0].set_title("(a) Decision Flip Reduction vs Budget")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Repair Efficiency eta_B (Flips Reduced per Query)
    for strat_key, grp in df_sweep.groupby("strategy_key"):
        s_name = grp["strategy_name"].iloc[0]
        axes[1].plot(
            grp["budget_pct"] * 100,
            grp["repair_efficiency_eta_mean"],
            marker="s",
            label=s_name,
            color=color_map.get(strat_key, "black"),
            lw=2.5 if strat_key in ["combined", "oracle"] else 1.5,
        )

    axes[1].set_xlabel("Acquisition Budget $B$ (% of Pool)")
    axes[1].set_ylabel("Repair Efficiency $\\eta_B$ (Flips Reduced / Query)")
    axes[1].set_title("(b) Sample Efficiency of Active Repair")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(figure_dir, "stage4a_flip_reduction_efficiency.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage 4A Flip Reduction & Efficiency Figure to {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Boundary Error vs Downstream Return (2 Panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Near-Boundary MSE vs Budget
    for strat_key, grp in df_sweep.groupby("strategy_key"):
        s_name = grp["strategy_name"].iloc[0]
        axes[0].plot(
            grp["budget_pct"] * 100,
            grp["near_boundary_mse"],
            marker="o",
            label=s_name,
            color=color_map.get(strat_key, "black"),
            lw=2.5 if strat_key in ["combined", "oracle"] else 1.5,
        )

    axes[0].set_xlabel("Acquisition Budget $B$ (% of Pool)")
    axes[0].set_ylabel("Near-Boundary Prediction MSE")
    axes[0].set_title("(a) Near-Boundary Model Resolution")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Policy Return vs Budget
    for strat_key, grp in df_sweep.groupby("strategy_key"):
        s_name = grp["strategy_name"].iloc[0]
        axes[1].plot(
            grp["budget_pct"] * 100,
            grp["policy_return_mean"],
            marker="s",
            label=s_name,
            color=color_map.get(strat_key, "black"),
            lw=2.5 if strat_key in ["combined", "oracle"] else 1.5,
        )

    axes[1].set_xlabel("Acquisition Budget $B$ (% of Pool)")
    axes[1].set_ylabel("Downstream Policy Mean Return")
    axes[1].set_title("(b) Downstream Policy Recovery")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(figure_dir, "stage4a_boundary_error_return.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage 4A Boundary Error & Return Figure to {fig2_path}")


if __name__ == "__main__":
    run_stage4a_active_acquisition_benchmark()
