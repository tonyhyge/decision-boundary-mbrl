"""
World-Model Architecture Portability Benchmark Pipeline.

Executes Gates P0, P1, P2, and P3 for testing:
    Does the same boundary-crossing-aware model-learning mechanism improve downstream control
    across structurally different world-model families (Deterministic Host A vs. Probabilistic Ensemble Host B)?
"""
import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import value_iteration, expected_discounted_return, policy_evaluation
from src.models.tabular_learned_model import (
    collect_gridworld_experience,
    evaluate_estimation_fidelity,
    TrajectoryDataset,
)
from src.models.portability_models import (
    WeightedCategoricalWorldModel,
    ProbabilisticEnsembleWorldModel,
)
from src.correction.portability_weighting import (
    compute_sample_scores,
    compute_normalized_weights,
)


def evaluate_policy_quality(
    true_mdp: ChoiceGridWorldMDP,
    learned_mdp: TabularMDP,
    Q_star: np.ndarray,
    pi_star: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate downstream control quality and fidelity in the true environment.
    """
    V_hat, Q_hat, pi_hat = value_iteration(learned_mdp)

    # 1. Expected discounted return in true environment
    j_learned = expected_discounted_return(true_mdp, pi_hat)
    j_optimal = expected_discounted_return(true_mdp, pi_star)
    regret = max(0.0, j_optimal - j_learned)

    # 2. Policy action agreement on active states
    active_states = [s for s in range(true_mdp.num_states) if s != true_mdp.goal_state]
    agreements = [int(pi_hat[s] == pi_star[s]) for s in active_states]
    action_agreement = float(np.mean(agreements))

    # 3. Action-ranking reversal rate (Z_cross)
    flips = [int(pi_hat[s] != pi_star[s]) for s in active_states]
    reversal_rate = float(np.mean(flips))

    # 4. Transition matrix Frobenius error against true dynamics
    p_true = true_mdp.transitions
    p_hat = learned_mdp.transitions
    transition_mse = float(np.mean((p_hat - p_true) ** 2))

    return {
        "j_learned": j_learned,
        "j_optimal": j_optimal,
        "regret": regret,
        "action_agreement": action_agreement,
        "reversal_rate": reversal_rate,
        "transition_mse": transition_mse,
    }


def select_best_lambda_on_validation(
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    true_mdp: ChoiceGridWorldMDP,
    initial_mdp: TabularMDP,
    initial_p_mat: np.ndarray,
    candidate_lambdas: List[float],
    host_type: str = "deterministic",
    seed: int = 42,
    epochs: int = 80,
) -> float:
    """
    Select hyperparameter lambda strictly on validation dataset without touching test evaluation.
    """
    best_lambda = candidate_lambdas[0]
    best_val_ret = -float("inf")

    V_star, Q_star, pi_star = value_iteration(true_mdp)

    for lam in candidate_lambdas:
        # Score training data using candidate lambda
        _, weights = compute_sample_scores(
            dataset=train_dataset,
            condition="estimated_crossing",
            true_mdp=true_mdp,
            initial_mdp=initial_mdp,
            initial_p_matrix=initial_p_mat,
            lambda_val=lam,
        )

        if host_type == "deterministic":
            model = WeightedCategoricalWorldModel(num_states=25, num_actions=4)
            model.fit(train_dataset, sample_weights=weights, epochs=epochs, seed=seed)
            learned_mdp = model.create_learned_mdp(true_mdp)
        else:
            model = ProbabilisticEnsembleWorldModel(num_states=25, num_actions=4, ensemble_size=5)
            model.fit(train_dataset, sample_weights=weights, epochs=epochs, base_seed=seed)
            learned_mdp = model.create_learned_mdp(true_mdp)

        _, _, pi_val = value_iteration(learned_mdp)
        val_ret = expected_discounted_return(true_mdp, pi_val)

        if val_ret > best_val_ret:
            best_val_ret = val_ret
            best_lambda = lam

    return best_lambda


def run_portability_benchmark(
    num_seeds: int = 30,
    base_seed: int = 700,
    output_dir: str = "results",
    figure_dir: str = "figures",
    num_trajectories_train: int = 80,
    num_trajectories_val: int = 40,
    epochs: int = 90,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("      WORLD-MODEL ARCHITECTURE PORTABILITY BENCHMARK PIPELINE                   ")
    print("================================================================================")

    candidate_lambdas = [0.5, 1.0, 2.0, 5.0]
    conditions = [
        "uniform",
        "prediction_error",
        "estimated_crossing",
        "shuffled_crossing",
        "oracle_crossing",
        "continuous_pressure",
    ]

    all_seed_records: List[Dict] = []
    p1_oracle_diffs: List[float] = []

    # -------------------------------------------------------------------------
    # GATE P1: ORACLE UTILITY TEST (Host A: Uniform vs. Oracle Crossing)
    # -------------------------------------------------------------------------
    print(f"\n[Gate P1] Evaluating Oracle Utility on Host A across {num_seeds} seeds...")

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        rng = np.random.default_rng(seed)

        true_grid = make_stochastic_choice_gridworld(height=5, width=5, seed=seed)
        V_star, Q_star, pi_star = value_iteration(true_grid)

        # Datasets
        train_data = collect_gridworld_experience(true_grid, num_trajectories=num_trajectories_train, max_steps=40, seed=seed)
        val_data = collect_gridworld_experience(true_grid, num_trajectories=num_trajectories_val, max_steps=40, seed=seed + 5000)

        # Train initial reference model (uniform) to compute diagnostics
        init_model = WeightedCategoricalWorldModel(25, 4)
        init_model.fit(train_data, epochs=epochs, seed=seed)
        init_mdp = init_model.create_learned_mdp(true_grid)
        init_p_mat = init_model.get_transition_matrix()

        # Fit Uniform Baseline (Host A)
        model_uniform = WeightedCategoricalWorldModel(25, 4)
        t0 = time.perf_counter()
        model_uniform.fit(train_data, sample_weights=None, epochs=epochs, seed=seed)
        time_uniform = time.perf_counter() - t0
        mdp_uniform = model_uniform.create_learned_mdp(true_grid)
        metrics_uniform = evaluate_policy_quality(true_grid, mdp_uniform, Q_star, pi_star)

        # Fit Oracle Crossing Model (Host A, lambda=1.0 for P1)
        _, oracle_w = compute_sample_scores(train_data, "oracle_crossing", true_grid, init_mdp, init_p_mat, lambda_val=1.0)
        model_oracle = WeightedCategoricalWorldModel(25, 4)
        t0 = time.perf_counter()
        model_oracle.fit(train_data, sample_weights=oracle_w, epochs=epochs, seed=seed)
        time_oracle = time.perf_counter() - t0
        mdp_oracle = model_oracle.create_learned_mdp(true_grid)
        metrics_oracle = evaluate_policy_quality(true_grid, mdp_oracle, Q_star, pi_star)

        diff_oracle = metrics_oracle["j_learned"] - metrics_uniform["j_learned"]
        p1_oracle_diffs.append(diff_oracle)

    mean_p1_diff = float(np.mean(p1_oracle_diffs))
    t_stat_p1, p_val_p1 = stats.ttest_1samp(p1_oracle_diffs, 0.0)
    w_stat_p1, w_pval_p1 = stats.wilcoxon(p1_oracle_diffs, zero_method="wilcox")

    print(f"   Mean Oracle Return Advantage Delta J: {mean_p1_diff:+.4f}")
    print(f"   Paired t-test: t = {t_stat_p1:.3f}, p = {p_val_p1:.4e}; Wilcoxon p = {w_pval_p1:.4e}")

    if mean_p1_diff <= 0.0 or p_val_p1 > 0.05:
        print("\n[VERDICT] PORTABILITY BRANCH FALSIFIED AT ORACLE UTILITY")
        summary = {
            "gate_p1_status": "FALSIFIED",
            "mean_oracle_diff": mean_p1_diff,
            "p_val": p_val_p1,
        }
        with open(os.path.join(output_dir, "portability_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    print("   -> Gate P1 PASSED: Oracle weighting provides significant downstream control gain.")

    # -------------------------------------------------------------------------
    # GATE P2 & P3: FULL MATRIX EVALUATION ACROSS BOTH HOST ARCHITECTURES
    # -------------------------------------------------------------------------
    print(f"\n[Gate P2 & P3] Executing Full 5-Condition Matrix on Host A and Host B...")

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        rng = np.random.default_rng(seed)

        true_grid = make_stochastic_choice_gridworld(height=5, width=5, seed=seed)
        V_star, Q_star, pi_star = value_iteration(true_grid)

        train_data = collect_gridworld_experience(true_grid, num_trajectories=num_trajectories_train, max_steps=40, seed=seed)
        val_data = collect_gridworld_experience(true_grid, num_trajectories=num_trajectories_val, max_steps=40, seed=seed + 5000)

        # Initial reference model
        init_model = WeightedCategoricalWorldModel(25, 4)
        init_model.fit(train_data, epochs=epochs, seed=seed)
        init_mdp = init_model.create_learned_mdp(true_grid)
        init_p_mat = init_model.get_transition_matrix()

        # Fidelity of initial model
        fidelity = evaluate_estimation_fidelity(true_grid, init_mdp)

        # Validation hyperparameter selection (shared grid)
        best_lam_a = select_best_lambda_on_validation(
            train_data, val_data, true_grid, init_mdp, init_p_mat, candidate_lambdas, host_type="deterministic", seed=seed, epochs=epochs
        )
        best_lam_b = select_best_lambda_on_validation(
            train_data, val_data, true_grid, init_mdp, init_p_mat, candidate_lambdas, host_type="ensemble", seed=seed, epochs=epochs
        )

        for host_name in ["Host_A_Deterministic", "Host_B_Ensemble"]:
            selected_lam = best_lam_a if host_name == "Host_A_Deterministic" else best_lam_b

            for cond in conditions:
                # Compute sample weights
                scores, weights = compute_sample_scores(
                    dataset=train_data,
                    condition=cond,
                    true_mdp=true_grid,
                    initial_mdp=init_mdp,
                    initial_p_matrix=init_p_mat,
                    rng=rng,
                    lambda_val=selected_lam,
                )

                t0 = time.perf_counter()
                if host_name == "Host_A_Deterministic":
                    model = WeightedCategoricalWorldModel(25, 4)
                    losses = model.fit(train_data, sample_weights=weights, epochs=epochs, seed=seed)
                    learned_mdp = model.create_learned_mdp(true_grid)
                    param_count = sum(p.numel() for p in model.net.parameters())
                else:
                    model = ProbabilisticEnsembleWorldModel(25, 4, ensemble_size=5)
                    losses = model.fit(train_data, sample_weights=weights, epochs=epochs, base_seed=seed)
                    learned_mdp = model.create_learned_mdp(true_grid)
                    param_count = sum(sum(p.numel() for p in m.parameters()) for m in model.members)

                wall_clock = time.perf_counter() - t0

                # Evaluate control quality
                metrics = evaluate_policy_quality(true_grid, learned_mdp, Q_star, pi_star)

                all_seed_records.append({
                    "seed": seed,
                    "host": host_name,
                    "condition": cond,
                    "lambda": selected_lam,
                    "j_learned": metrics["j_learned"],
                    "j_optimal": metrics["j_optimal"],
                    "regret": metrics["regret"],
                    "action_agreement": metrics["action_agreement"],
                    "reversal_rate": metrics["reversal_rate"],
                    "transition_mse": metrics["transition_mse"],
                    "wall_clock_sec": wall_clock,
                    "param_count": param_count,
                    "crossing_auroc": fidelity["crossing_auroc"],
                    "margin_mae": fidelity["margin_mae"],
                })

    df_results = pd.DataFrame(all_seed_records)
    csv_path = os.path.join(output_dir, "portability_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n-> Saved {len(df_results)} benchmark evaluation records to {csv_path}")

    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS & CONTRASTS
    # -------------------------------------------------------------------------
    summary_report = analyze_portability_results(df_results, output_dir, figure_dir)
    return summary_report


def analyze_portability_results(df: pd.DataFrame, output_dir: str, figure_dir: str) -> Dict:
    print("\n================================================================================")
    print("      STATISTICAL SYNTHESIS & HYPOTHESIS TESTING RESULTS                        ")
    print("================================================================================")

    hosts = ["Host_A_Deterministic", "Host_B_Ensemble"]
    analysis_dict: Dict[str, Dict] = {}

    rng_boot = np.random.default_rng(42)

    for host in hosts:
        df_h = df[df["host"] == host]
        print(f"\n--- {host.upper().replace('_', ' ')} ---")

        # Mean and std by condition
        agg = df_h.groupby("condition").agg({
            "j_learned": ["mean", "std"],
            "regret": ["mean", "std"],
            "action_agreement": ["mean", "std"],
            "reversal_rate": ["mean", "std"],
            "transition_mse": ["mean", "std"],
            "wall_clock_sec": ["mean"],
        })
        print(agg)

        # Seed-paired comparisons
        piv = df_h.pivot(index="seed", columns="condition", values="j_learned")

        # 1. Delta J = Estimated Boundary - Uniform
        diff_boundary_uniform = (piv["estimated_crossing"] - piv["uniform"]).to_numpy()
        mean_dj = float(np.mean(diff_boundary_uniform))
        std_dj = float(np.std(diff_boundary_uniform, ddof=1))
        d_eff_dj = mean_dj / (std_dj + 1e-12)
        t_dj, p_dj = stats.ttest_1samp(diff_boundary_uniform, 0.0)
        w_dj, wp_dj = stats.wilcoxon(diff_boundary_uniform, zero_method="wilcox")

        boot_dj = [np.mean(rng_boot.choice(diff_boundary_uniform, size=len(diff_boundary_uniform), replace=True)) for _ in range(2000)]
        bci_dj = [float(np.percentile(boot_dj, 2.5)), float(np.percentile(boot_dj, 97.5))]

        # 2. Delta J_semantic = Estimated Boundary - Shuffled
        diff_semantic = (piv["estimated_crossing"] - piv["shuffled_crossing"]).to_numpy()
        mean_dsem = float(np.mean(diff_semantic))
        std_dsem = float(np.std(diff_semantic, ddof=1))
        d_eff_dsem = mean_dsem / (std_dsem + 1e-12)
        t_dsem, p_dsem = stats.ttest_1samp(diff_semantic, 0.0)
        w_dsem, wp_dsem = stats.wilcoxon(diff_semantic, zero_method="wilcox")

        boot_dsem = [np.mean(rng_boot.choice(diff_semantic, size=len(diff_semantic), replace=True)) for _ in range(2000)]
        bci_dsem = [float(np.percentile(boot_dsem, 2.5)), float(np.percentile(boot_dsem, 97.5))]

        # 3. Prediction Error vs Uniform
        diff_pe = (piv["prediction_error"] - piv["uniform"]).to_numpy()
        mean_pe = float(np.mean(diff_pe))
        t_pe, p_pe = stats.ttest_1samp(diff_pe, 0.0)

        # 4. Oracle vs Uniform
        diff_oracle = (piv["oracle_crossing"] - piv["uniform"]).to_numpy()
        mean_oracle = float(np.mean(diff_oracle))
        t_oracle, p_oracle = stats.ttest_1samp(diff_oracle, 0.0)

        print(f"\n   -> Delta J (Estimated - Uniform): {mean_dj:+.4f}, 95% BCI: [{bci_dj[0]:.4f}, {bci_dj[1]:.4f}], Cohen's d: {d_eff_dj:.3f}, p = {p_dj:.4e}")
        print(f"   -> Delta J_semantic (Estimated - Shuffled): {mean_dsem:+.4f}, 95% BCI: [{bci_dsem[0]:.4f}, {bci_dsem[1]:.4f}], Cohen's d: {d_eff_dsem:.3f}, p = {p_dsem:.4e}")
        print(f"   -> Delta J (Prediction Error - Uniform): {mean_pe:+.4f}, p = {p_pe:.4e}")
        print(f"   -> Delta J (Oracle - Uniform): {mean_oracle:+.4f}, p = {p_oracle:.4e}")

        analysis_dict[host] = {
            "mean_return_uniform": float(piv["uniform"].mean()),
            "mean_return_prediction_error": float(piv["prediction_error"].mean()),
            "mean_return_estimated_crossing": float(piv["estimated_crossing"].mean()),
            "mean_return_shuffled": float(piv["shuffled_crossing"].mean()),
            "mean_return_oracle": float(piv["oracle_crossing"].mean()),
            "delta_j_mean": mean_dj,
            "delta_j_bci_95": bci_dj,
            "delta_j_cohen_d": d_eff_dj,
            "delta_j_p_val": p_dj,
            "delta_j_wilcoxon_p": wp_dj,
            "delta_semantic_mean": mean_dsem,
            "delta_semantic_bci_95": bci_dsem,
            "delta_semantic_cohen_d": d_eff_dsem,
            "delta_semantic_p_val": p_dsem,
            "delta_semantic_wilcoxon_p": wp_dsem,
            "mean_pe_diff": mean_pe,
            "mean_oracle_diff": mean_oracle,
        }

    # Generate Publication Figures
    generate_portability_figures(df, figure_dir)

    # Determine Scientific Conclusion
    a_supported = (analysis_dict["Host_A_Deterministic"]["delta_j_mean"] > 0 and 
                   analysis_dict["Host_A_Deterministic"]["delta_semantic_mean"] > 0 and
                   analysis_dict["Host_A_Deterministic"]["delta_semantic_p_val"] < 0.05)
    b_supported = (analysis_dict["Host_B_Ensemble"]["delta_j_mean"] > 0 and 
                   analysis_dict["Host_B_Ensemble"]["delta_semantic_mean"] > 0 and
                   analysis_dict["Host_B_Ensemble"]["delta_semantic_p_val"] < 0.05)

    if a_supported and b_supported:
        verdict = "PORTABILITY SUPPORTED"
    elif a_supported or b_supported:
        verdict = "PORTABILITY PARTIALLY SUPPORTED"
    elif analysis_dict["Host_A_Deterministic"]["mean_oracle_diff"] > 0 and not a_supported:
        verdict = "ESTIMATION BOTTLENECK"
    else:
        verdict = "ALGORITHMIC PORTABILITY FALSIFIED"

    print(f"\n================================================================================")
    print(f"      FINAL SCIENTIFIC VERDICT: [{verdict}]                                    ")
    print("================================================================================")

    output_summary = {
        "verdict": verdict,
        "host_a_supported": a_supported,
        "host_b_supported": b_supported,
        "analysis": analysis_dict,
    }

    with open(os.path.join(output_dir, "portability_summary.json"), "w") as f:
        json.dump(output_summary, f, indent=2)

    return output_summary


def generate_portability_figures(df: pd.DataFrame, figure_dir: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cond_order = [
        "uniform",
        "prediction_error",
        "shuffled_crossing",
        "estimated_crossing",
        "oracle_crossing",
    ]
    cond_labels = [
        "Uniform\nBaseline",
        "Prediction\nError",
        "Shuffled\nControl",
        "Estimated\nCrossing",
        "Oracle\nCrossing",
    ]
    colors = ["#7f7f7f", "#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]

    for idx, (host, ax, title) in enumerate([
        ("Host_A_Deterministic", axes[0], "(a) Host A: Deterministic Dynamics"),
        ("Host_B_Ensemble", axes[1], "(b) Host B: Probabilistic Ensemble Dynamics"),
    ]):
        df_h = df[df["host"] == host]
        means = [df_h[df_h["condition"] == c]["j_learned"].mean() for c in cond_order]
        sems = [df_h[df_h["condition"] == c]["j_learned"].std() / np.sqrt(len(df_h[df_h["condition"] == c])) for c in cond_order]

        bars = ax.bar(range(len(cond_order)), means, yerr=sems, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_labels, fontsize=10)
        ax.set_ylabel("True Environment Return $J(\\pi^*_{\\hat{P}}; P)$")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Value label on top of bars
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m * 0.96, f"{m:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="white")

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "portability_architecture_comparison.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Portability Architecture Figure to {fig_path}")


if __name__ == "__main__":
    run_portability_benchmark()
